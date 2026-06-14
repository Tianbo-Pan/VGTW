# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vgtw.models.aggregator_lora import Aggregator
from vgtw.heads.camera_head import CameraHead
from vgtw.heads.dpt_head import DPTHead
# from vgtw.heads.dpt_heads_voxel import DPTHead_voxel
from vgtw.heads.dpt_heads_voxel_old import DPTHead_voxel
from vgtw.heads.track_head import TrackHead
from vgtw.utils.pose_enc import pose_encoding_to_extri_intri
from ..layers.attention_lora import LoRAConfig

from fast3r.dust3r.utils.misc import (
    freeze_all_params,
    activate_grad,
)


class VGTW(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 img_size=518, 
                 patch_size=14, 
                 embed_dim=1024, 
                 freeze='none', 
                 use_semantic=False, 
                 use_img=False, 
                 use_density=False,
                 use_conf=False, 
                 use_3D_pos_embed=False,  # 新增参数
                 lora_r = 8,
                 lora_alpha = 16.0,
                 lora_dropout = 0.05, 
                 mask_channels=1):
        """
        初始化 VGTW 模型，包含多个头用于处理不同任务。

        参数：
            img_size (int): 输入图像大小（默认 518x518）。默认为 518。
            patch_size (int): 分块大小，用于图像分割。默认为 14。
            embed_dim (int): 嵌入维度，传递给 Aggregator 和各个 head。默认为 1024。
            freeze (str): 冻结模型部分的参数，可选 'none' 或其他。默认为 'none'。
            use_semantic (bool): 是否使用语义特征。默认为 True。
            use_color_similarity (bool): 是否使用颜色特征（用于 mask_head）。默认为 True。
            use_mask (bool): 是否使用掩膜特征（用于 mask_head）。默认为 True。
            use_img (bool): 是否使用图像特征（用于 mask_head）。默认为 True。
            use_conf (bool): 是否使用置信度特征（用于 mask_head）。默认为 True。
            use_depth_f (bool): 是否使用深度特征（用于 mask_head）。默认为 True。
            use_semantic_similarity (bool): 是否使用语义相似性特征（用于 mask_head）。默认为 True。
            feat_in_channels (int): 深度特征的输入通道数（用于 mask_head）。默认为 128。
            mask_channels (int): 掩膜输出的通道数（用于 mask_head）。默认为 1。
        """
        super().__init__()

        # 初始化 Aggregator 和各个 head
        self.patch_size=patch_size
        lora_config = LoRAConfig(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, lora_config=lora_config)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        
        # 初始化 mask_head，传递所有相关参数
        self.use_semantic = use_semantic
        self.mask_head = DPTHead_voxel(dim_in=2 * embed_dim, output_dim=1, 
                                       use_density=use_density, use_semantic=use_semantic, use_3D_pos_embed=use_3D_pos_embed, use_img=use_img)
        # sam2_checkpoint = "/local_home2/pantianbo/projects/wild_reconstruction/vgtw_ptb/sam2_all_code/checkpoints/sam2.1_hiera_large.pt"
        # model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        # self.sam_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

        self.set_freeze(freeze)
        # self.voxel_occlusion_detector = VoxelOcclusionDetector(voxel_size=0.05)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "all_except_mask": [self.aggregator, self.camera_head, self.point_head, self.depth_head, self.track_head],  # 冻结所有模块
        }
        # 冻结指定模块的参数
        freeze_all_params(to_be_frozen[freeze])
        # 如果是 "all_except_mask" 模式，单独解冻 mask_head 的参数
        if freeze == "all_except_mask":
            activate_grad([self.mask_head])
            for name, param in self.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"  {name}: requires_grad={param.requires_grad}")
        # print()

    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        mode='training',
    ):
        """
        Forward pass of the VGTW model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        
        img_list = [item['img'] for item in views]
        images = torch.cat(img_list, dim=0)
        if len(images.shape) == 4:
            if images.shape[1] != 3:
                images = images.permute(0,3,1,2)
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        _, V, _, H, W = images.shape
        img_src_list = [item['img_original'][0] for item in views]

        predictions = {}

        # semantic_feature = torch.nn.functional.interpolate(semantic_feature,size=(H, W), mode="bilinear", align_corners=False)
        predictions["s_feats"] = None
        predictions["attn_feats"] = torch.stack(aggregated_tokens_list).permute(1,2,0,3,4)


        # with torch.cuda.amp.autocast(enabled=False):
        if self.mask_head is not None:
            depth_mask = self.mask_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, 
                density_entropy=None, patch_semantic=None, patch_xyz=None
            )
            depth_mask = depth_mask.squeeze(2)
            predictions["depth_mask"] = depth_mask
            predictions["depth_mask_binary"] = (torch.sigmoid(depth_mask) > 0.5).float()
            
        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

        if self.depth_head is not None:
            depth, depth_conf, _ = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            conf_prior = depth_conf.clone()
            conf_prior = conf_prior.permute(1, 0, 2, 3)
            # if depth_feature.shape[0] == 1:
            #     depth_feature = depth_feature.squeeze(0)

            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.point_head is not None:
            pts3d, pts3d_conf, _ = self.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )

            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf

        predictions["images"] = images

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                if predictions[key].shape[0] == 1:
                    predictions[key] = predictions[key].squeeze(0)  # remove batch dimension

        # depth_map = predictions["depth"]  # (S, H, W, 1)
        # world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        # world_points = torch.from_numpy(world_points).to(images.device)
        # predictions["world_points_from_depth"] = world_points


        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["refined_mask_binary"] = predictions["depth_mask_binary"]

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                if predictions[key].shape[0] == 1:
                    predictions[key] = predictions[key].squeeze(0)  # remove batch dimension

        S = len(views)
        output_views = []
        for i in range(S):
            view_dict = {}
            for key, value in predictions.items():
                if key in ('patch_size','s_feats'):
                    continue
                view_dict[key] = value[i:i+1]  # 切片保持 batch 维度为 1
            output_views.append(view_dict)

        del aggregated_tokens_list
        torch.cuda.empty_cache()
        return output_views


# Backward-compatible aliases
vgtw = VGTW
VGGT = VGTW
