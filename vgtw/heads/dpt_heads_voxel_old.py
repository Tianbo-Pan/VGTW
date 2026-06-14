# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead_voxel(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        use_density: bool = True,
        use_semantic: bool = True,
        use_img: bool = True,
        use_3D_pos_embed: bool = True,
    ) -> None:
        super(DPTHead_voxel, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_density = use_density
        self.use_semantic = use_semantic
        self.use_img = use_img
        self.use_3D_pos_embed = use_3D_pos_embed

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(
            out_channels,
            features,
            expand=False,
        )

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

        if self.use_density:
            density_c = 1
            self.density_to_layer1 = nn.ConvTranspose2d(density_c, 256, kernel_size=4, stride=2, padding=1)
            self.density_to_layer2 = nn.Conv2d(density_c, 256, kernel_size=3, stride=1, padding=1)
            self.density_to_layer3 = nn.Conv2d(density_c, 256, kernel_size=3, stride=2, padding=1)
            self.density_to_layer4 = nn.Sequential(
                nn.Conv2d(density_c, 256, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1) 
            )

        if self.use_semantic:
            semantic_c = 256
            self.semantic_to_layer1 = nn.ConvTranspose2d(semantic_c, 256, kernel_size=4, stride=2, padding=1)
            self.semantic_to_layer2 = nn.Conv2d(semantic_c, 256, kernel_size=3, stride=1, padding=1)
            self.semantic_to_layer3 = nn.Conv2d(semantic_c, 256, kernel_size=3, stride=2, padding=1)
            self.semantic_to_layer4 = nn.Sequential(
                nn.Conv2d(semantic_c, 256, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1) 
            )
            
        if self.use_img:
            image_c = 3
            # 首先将图像从 [392, 518] 下采样到 [56, 74] 并转换为 256 个通道
            self.image_base = nn.Sequential(
                nn.AvgPool2d(kernel_size=7, stride=7),  # 下采样到 [56, 74]
                nn.Conv2d(image_c, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            # 转换为 layer1 的分辨率 [112, 148]
            self.image_to_layer1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
            # 转换为 layer2 的分辨率 [56, 74]（保持分辨率）
            self.image_to_layer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            # 转换为 layer3 的分辨率 [28, 37]
            self.image_to_layer3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            # 转换为 layer4 的分辨率 [14, 19]
            self.image_to_layer4 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            )
        
        if self.use_3D_pos_embed:
            self.get_embed_3D_pose  = nn.ModuleList(
            [   nn.Sequential(
                nn.Linear(3, oc),
                nn.LayerNorm(oc),
                nn.ReLU(),
                nn.Linear(oc, oc),
            ) for oc in out_channels]
            )

        # Calculate fusion_conv input channels dynamically
        fusion_in_channels = conv2_in_channels
        if self.use_density:
            fusion_in_channels += 32  # density_feature channels
        if self.use_semantic:
            fusion_in_channels += 32  # semantic_feature channels
        self.fusion_conv = nn.Conv2d(fusion_in_channels, 128, kernel_size=1)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        density_entropy: torch.Tensor, 
        patch_semantic: torch.Tensor,
        patch_xyz: torch.Tensor,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, density_entropy, patch_semantic, patch_xyz, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []
        all_conf = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, density_entropy, patch_semantic, patch_xyz, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds = self._forward_impl(
                    aggregated_tokens_list, images, density_entropy, patch_semantic, patch_xyz, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)

        # Concatenate results along the sequence dimension
        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        density_entropy: torch.Tensor,
        patch_semantic: torch.Tensor,
        patch_xyz: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        all_images = images.clone()
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
            if self.use_density and density_entropy is not None:
                density_entropy = density_entropy[frames_start_idx:frames_end_idx].contiguous()
            if self.use_semantic and patch_semantic is not None:
                patch_semantic = patch_semantic[frames_start_idx:frames_end_idx].contiguous()
            if self.use_3D_pos_embed and patch_xyz is not None:
                patch_xyz = patch_xyz[frames_start_idx:frames_end_idx].contiguous()


        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        if self.use_3D_pos_embed:
            if patch_xyz.shape[-2] != patch_h or patch_xyz.shape[-1] != patch_w:
                patch_xyz = F.interpolate(patch_xyz, size=(patch_h, patch_w), mode='nearest')
            patch_xyz = patch_xyz.permute(0, 2, 3, 1).reshape(S * patch_h * patch_w, 3)
            # embed_3D_pos = self.get_embed_3D_pose(patch_xyz)
            # embed_3D_pos = embed_3D_pos.reshape(S, patch_h, patch_w, -1).permute(0, 3, 1, 2)

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # # Visualize the feature distance
            # if layer_idx==23:
            #     # distance = self.compute_min_distances(x)
            #     distance = self.compute_avg_min_distances_per_frame(x)
            #     distance = distance.squeeze(-1)
            #     distance = distance.reshape((distance.shape[0], distance.shape[1], patch_h, patch_w))
            #     distance = F.interpolate(distance, size=(H, W), mode='bilinear')

            #     save_dir = 'ptb_visualization/'
            #     V = distance.shape[1]  # 帧数，例如 V = 9
                
            #     # 转换为 NumPy 数组（确保在 CPU 上）
            #     distance = distance.cpu().numpy()
                
            #     # 展平所有帧的距离数据以计算全局阈值
            #     all_distances = distance[0].reshape(-1)  # 形状 [V * patch_h * patch_w]
            #     global_threshold = np.percentile(all_distances, 50)  # 计算全局 50% 百分位数
                           
            #     # 遍历每一帧进行可视化
            #     for k in range(V):
            #         # 提取第 k 帧的距离矩阵，形状 [28, 37]
            #         image = all_images.clone()
            #         image = image[0,k]
            #         image = image.permute(1,2,0)*255
            #         image = image.cpu().numpy().astype(np.uint8)
            #         frame_distance = distance[0, k, :, :]
            #         inverse_distance = np.max(frame_distance) - frame_distance
            #         global_mask = np.zeros_like(frame_distance, dtype=np.uint8)
            #         global_mask = np.where(frame_distance <= global_threshold, 255, global_mask)

            #         # 创建新图形，包含两个子图
            #         fig, axes = plt.subplots(1, 3, figsize=(24, 6))
                    
            #         # 第一个子图：原始距离热图
            #         im1 = axes[0].imshow(inverse_distance, cmap='plasma', interpolation='nearest', vmax=None)
            #         axes[0].set_title(f'Frame {k} Distance Matrix')
            #         axes[0].set_xlabel('Patch Width')
            #         axes[0].set_ylabel('Patch Height')
            #         fig.colorbar(im1, ax=axes[0], label='Distance')
                    
            #         # 第二个子图：全局二进制掩码
            #         axes[1].imshow(global_mask, cmap='gray', interpolation='nearest')
            #         axes[1].set_title('Global Binary Mask (Top 50% Across All Frames)')
            #         axes[1].set_xlabel('Patch Width')
            #         axes[1].set_ylabel('Patch Height')

            #         axes[2].imshow(image)  # 显示原始图像
            #         axes[2].imshow(inverse_distance, cmap='viridis', alpha=0.5, interpolation='nearest')  # 重叠热图，透明度为0.5
            #         axes[2].set_title(f'Frame {k} Image with Distance Overlay')
            #         axes[2].set_xlabel('Width')
            #         axes[2].set_ylabel('Height')
                    
            #         # 保存图像
            #         save_path = os.path.join(save_dir, f'frame_{k}_distance_with_global_mask.png')
            #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
            #         plt.close()
                
            #     print(f"已将 {V} 个帧的距离矩阵和全局二进制掩码可视化保存到 {save_dir} 目录")

            #     # for k in range(V):
            #     #     # 提取第 k 帧的距离矩阵，形状 [H, W]
            #     #     image = all_images.clone()
            #     #     image = image[0, k]
            #     #     image = image.permute(1, 2, 0) * 255
            #     #     image = image.cpu().numpy().astype(np.uint8)
            #     #     frame_distance = distance[0, k, :, :]
            #     #     inverse_distance = np.max(frame_distance) - frame_distance

            #     #     # 计算当前帧的 50% 百分位数阈值
            #     #     frame_distances = frame_distance.reshape(-1)  # 展平为一维数组
            #     #     frame_threshold = np.percentile(frame_distances, 50)  # 当前帧的 50% 阈值
            #     #     frame_mask = np.zeros_like(frame_distance, dtype=np.uint8)
            #     #     frame_mask = np.where(frame_distance <= frame_threshold, 255, frame_mask)

            #     #     # 创建新图形，包含三个子图
            #     #     fig, axes = plt.subplots(1, 3, figsize=(24, 6))

            #     #     # 第一个子图：距离热图
            #     #     im1 = axes[0].imshow(inverse_distance, cmap='viridis', interpolation='nearest')
            #     #     axes[0].set_title(f'Frame {k} Distance Matrix')
            #     #     axes[0].set_xlabel('Patch Width')
            #     #     axes[0].set_ylabel('Patch Height')
            #     #     fig.colorbar(im1, ax=axes[0], label='Distance')

            #     #     # 第二个子图：当前帧的二进制掩码
            #     #     axes[1].imshow(frame_mask, cmap='gray', interpolation='nearest')
            #     #     axes[1].set_title(f'Frame {k} Binary Mask (Top 50% Patches)')
            #     #     axes[1].set_xlabel('Patch Width')
            #     #     axes[1].set_ylabel('Patch Height')

            #     #     # 第三个子图：原始图像与距离热图叠加
            #     #     axes[2].imshow(image)  # 显示原始图像
            #     #     axes[2].imshow(inverse_distance, cmap='viridis', alpha=0.5, interpolation='nearest')  # 重叠热图
            #     #     axes[2].set_title(f'Frame {k} Image with Distance Overlay')
            #     #     axes[2].set_xlabel('Width')
            #     #     axes[2].set_ylabel('Height')

            #     #     # 保存图像
            #     #     save_path = os.path.join(save_dir, f'frame_{k}_distance_with_frame_mask.png')
            #     #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
            #     #     plt.close()

            #     # print(f"已将 {V} 个帧的距离矩阵和每帧独立 50% 二进制掩码可视化保存到 {save_dir} 目录")


            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)

            if self.use_3D_pos_embed:
                patch_3D_pos = patch_xyz.clone()
                embed_3D_pos = self.get_embed_3D_pose[dpt_idx](patch_3D_pos)
                embed_3D_pos = embed_3D_pos.reshape(S, patch_h, patch_w, -1).permute(0, 3, 1, 2)
                x = x + embed_3D_pos
            elif self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        if self.use_density:
            density_entropy = density_entropy.view(B * S, 1, density_entropy.shape[-2], density_entropy.shape[-1])
            density_feature1 = self.density_to_layer1(density_entropy)
            density_feature2 = self.density_to_layer2(density_entropy)
            density_feature3 = self.density_to_layer3(density_entropy)
            density_feature4 = self.density_to_layer4(density_entropy)
            density_features = [density_feature1, density_feature2, density_feature3, density_feature4]
        else:
            density_features = None

        if self.use_semantic:
            patch_semantic = patch_semantic.view(B * S, 256, patch_semantic.shape[-2], patch_semantic.shape[-1])
            semantic_feature1 = self.semantic_to_layer1(patch_semantic)
            semantic_feature2 = self.semantic_to_layer2(patch_semantic)
            semantic_feature3 = self.semantic_to_layer3(patch_semantic)
            semantic_feature4 = self.semantic_to_layer4(patch_semantic)
            semantic_features = [semantic_feature1, semantic_feature2, semantic_feature3, semantic_feature4]
        else:
            semantic_features = None

        if self.use_img:
            images = images.view(B * S, 3, images.shape[-2], images.shape[-1])
            image_base_feature = self.image_base(images)
            image_feature1 = self.image_to_layer1(image_base_feature)
            image_feature2 = self.image_to_layer2(image_base_feature)
            image_feature3 = self.image_to_layer3(image_base_feature)
            image_feature4 = self.image_to_layer4(image_base_feature)
            image_features = [image_feature1, image_feature2, image_feature3, image_feature4]
        else:
            image_features = None

        out = self.scratch_forward(out, density_features, semantic_features, image_features)

        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        preds = self.scratch.output_conv2(out)
        preds = preds.view(B, S, *preds.shape[1:])
        return preds

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x 
        pos_embed

    def scratch_forward(
        self,
        features: List[torch.Tensor],
        density_features: List[torch.Tensor] = None,
        semantic_features: List[torch.Tensor] = None,
        image_features: List[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the fusion blocks with optional additional features.

        Args:
            features (List[Tensor]): List of feature maps from different layers.
            density_features (List[Tensor], optional): List of density feature maps for each layer.
            semantic_features (List[Tensor], optional): List of semantic feature maps for each layer.
            image_features (List[Tensor], optional): List of image feature maps for each layer.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        # Process each layer with self.scratch.layerX_rn
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # Fuse additional features based on flags
        if self.use_density and density_features is not None:
            layer_1_rn = layer_1_rn + density_features[0]
            layer_2_rn = layer_2_rn + density_features[1]
            layer_3_rn = layer_3_rn + density_features[2]
            layer_4_rn = layer_4_rn + density_features[3]
            del density_features

        if self.use_semantic and semantic_features is not None:
            layer_1_rn = layer_1_rn + semantic_features[0]
            layer_2_rn = layer_2_rn + semantic_features[1]
            layer_3_rn = layer_3_rn + semantic_features[2]
            layer_4_rn = layer_4_rn + semantic_features[3]
            del semantic_features

        if self.use_img and image_features is not None:
            layer_1_rn = layer_1_rn + image_features[0]
            layer_2_rn = layer_2_rn + image_features[1]
            layer_3_rn = layer_3_rn + image_features[2]
            layer_4_rn = layer_4_rn + image_features[3]
            del image_features

        # Proceed with the fusion through refinenet modules
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out
    
    def compute_min_distances(self, x):
        V = x.shape[1]  # 帧数
        patch_num = x.shape[2]  # 每帧的 patch 数量，例如 1036
        channel = x.shape[3]
        
        # 初始化结果张量，形状为 [1, V, patch_num, 1]
        min_distances = torch.empty(1, V, patch_num, 1)
        
        # 遍历每一帧
        for k in range(V):
            # 提取第 k 帧的所有 patch，形状 [1036, 2048]
            patches_k = x[0, k, :, :]
            
            # 提取其他帧的所有 patch
            other_frames_indices = [l for l in range(V) if l != k]
            other_patches = x[0, other_frames_indices, :, :]  # 形状 [V-1, 1036, 2048]
            other_patches = other_patches.view(-1, channel)  # 重塑为 [(V-1)*1036, 2048]
            
            # 计算第 k 帧每个 patch 与其他帧所有 patch 的欧几里得距离
            distances = torch.cdist(patches_k, other_patches)  # 形状 [1036, (V-1)*1036]
            
            # 找到每个 patch 的最小距离
            min_distances_k = torch.min(distances, dim=1).values  # 形状 [1036]
            
            # 存储到结果张量
            min_distances[0, k, :, 0] = min_distances_k
        
        return min_distances

    def compute_avg_min_distances_per_frame(self, x):
        V = x.shape[1]  # 帧数
        patch_num = x.shape[2]  # patch 数量
        channel = x.shape[3]
        
        avg_min_distances = torch.empty(1, V, patch_num, 1, device=x.device)
        
        for k in range(V):
            patches_k = x[0, k, :, :]
            
            min_distances_per_other_frame = []
            for l in range(V):
                if l != k:
                    other_patches = x[0, l, :, :]
                    distances = torch.cdist(patches_k, other_patches)
                    min_dist_l = torch.min(distances, dim=1).values
                    min_distances_per_other_frame.append(min_dist_l)
            
            # 栈叠并取平均
            min_distances_stack = torch.stack(min_distances_per_other_frame, dim=1)
            avg_min_dist = torch.mean(min_distances_stack, dim=1)
            avg_min_distances[0, k, :, 0] = avg_min_dist
        
        return avg_min_distances

################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
