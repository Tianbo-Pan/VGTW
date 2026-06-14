import logging
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

XFORMERS_AVAILABLE = False

def visualize_patch_attn(
    attn: torch.Tensor,
    head_idx: int = 0,
    batch_idx: int = 0,
    save_path: str = 'ptb_visualization/check.png',
    patch_height: int = 37,
    patch_width: int = 37,
    num_frames: int = 6,
    specified_patch: tuple = None,
    patch_size: int = 14,
    vmax: float = None,
) -> None:
    """
    Selects a patch from frame 0 (using specified (H, W) position or randomly chosen),
    visualizes its attention map in a 37x37 grid with a red dot at the selected patch,
    and visualizes the attention map of this patch to patches in other frames (1-5).
    Attention maps are upscaled by patch_size to match the original image resolution.

    Parameters:
    - attn: torch.Tensor [B, H, N, N], softmax-normalized global attention weights
    - head_idx: int, attention head index (default 0)
    - batch_idx: int, batch index (default 0)
    - save_path: str, optional, path to save the image (e.g., 'attn_maps.png')
    - patch_height: int, patch grid height (default 37)
    - patch_width: int, patch grid width (default 37)
    - num_frames: int, number of frames (default 6)
    - specified_patch: tuple, specified patch position (row, col), if None, randomly chosen (default None)
    - patch_size: int, size of each patch in pixels (default 14)

    Returns: None (displays or saves the image)
    """
    # Clone attn, convert to float32, move to CPU
    attn = attn.clone().to(dtype=torch.float32, device='cpu').numpy()

    # Define constants
    camera_tokens = 5
    tokens_per_frame = patch_height * patch_width + camera_tokens  # 37*37 + 5 = 1374
    patches_per_frame = patch_height * patch_width  # 37*37 = 1369
    total_tokens = num_frames * tokens_per_frame  # 6 * 1374 = 8244
    img_height = patch_height * patch_size  # e.g., 37 * 14 = 518
    img_width = patch_width * patch_size    # e.g., 37 * 14 = 518

    # Check specified patch position
    selected_patch_idx = None
    if specified_patch is not None:
        if len(specified_patch) != 2 or not isinstance(specified_patch[0], int) or not isinstance(specified_patch[1], int):
            raise ValueError("specified_patch must be a (row, col) tuple")
        patch_row, patch_col = specified_patch
        if patch_row < 0 or patch_row >= patch_height or patch_col < 0 or patch_col >= patch_width:
            raise ValueError(f"Specified patch position ({patch_row}, {patch_col}) is out of {patch_height}x{patch_width} grid range")
        selected_patch_idx = patch_row * patch_width + patch_col
        print(f"Using specified patch position: ({patch_row}, {patch_col})")
    else:
        # Randomly select a patch index
        selected_patch_idx = random.randint(0, patches_per_frame - 1)
        patch_row = selected_patch_idx // patch_width
        patch_col = selected_patch_idx % patch_width
        print(f"Randomly selected patch position: ({patch_row}, {patch_col})")

    # Global query index: frame0_start + camera_tokens + selected_patch_idx
    frame0_start = 0
    query_idx = frame0_start + camera_tokens + selected_patch_idx

    # Calculate the center of the selected patch in original resolution
    patch_center_y = (patch_row + 0.5) * patch_size  # Center in original image height
    patch_center_x = (patch_col + 0.5) * patch_size  # Center in original image width

    # Create subplots: 1 for frame 0 attention map + 5 for frames 1-5
    fig, axes = plt.subplots(1, num_frames, figsize=(5 * num_frames, 5))

    # Subplot 1: Visualize attention map for frame 0 patches
    frame0_patches_start = frame0_start + camera_tokens
    frame0_patches_end = frame0_start + tokens_per_frame
    attn_vector_frame0 = attn[batch_idx, head_idx, query_idx, frame0_patches_start:frame0_patches_end]  # [1369]
    attn_map_frame0 = attn_vector_frame0.reshape(patch_height, patch_width)  # [37, 37]

    # Upscale attention map to original image resolution
    attn_map_frame0_tensor = torch.from_numpy(attn_map_frame0).unsqueeze(0).unsqueeze(0)  # [1, 1, 37, 37]
    attn_map_frame0_upscaled = F.interpolate(
        attn_map_frame0_tensor,
        size=(img_height, img_width),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0).numpy()  # [518, 518]

    # Plot upscaled attention map for frame 0
    sns.heatmap(attn_map_frame0_upscaled, cmap='viridis', square=True, cbar=True, ax=axes[0],vmin=0, vmax=vmax)
    axes[0].set_title(f'Frame 0 Patches')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # # Mark the selected patch position with a red dot
    # axes[0].plot(patch_center_x, patch_center_y, 'ro', markersize=10, label='Selected Patch')
    # axes[0].legend()


    # Subplots 2-6: Visualize attention maps for query patch to frames 1-5
    for frame_j in range(1, num_frames):
        # Calculate patch range for frame_j
        frame_j_start = frame_j * tokens_per_frame
        key_patches_start = frame_j_start + camera_tokens
        key_patches_end = frame_j_start + tokens_per_frame

        # Extract attention vector: query to frame_j patches
        attn_vector = attn[batch_idx, head_idx, query_idx, key_patches_start:key_patches_end]  # [1369]
        attn_map = attn_vector.reshape(patch_height, patch_width)  # [37, 37]

        # Upscale attention map to original image resolution
        attn_map_tensor = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)  # [1, 1, 37, 37]
        attn_map_upscaled = F.interpolate(
            attn_map_tensor,
            size=(img_height, img_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()  # [518, 518]

        # Plot upscaled attention map
        sns.heatmap(attn_map_upscaled, cmap='viridis', square=True, cbar=False, ax=axes[frame_j], vmin=0, vmax=vmax)
        axes[frame_j].set_title(f'Frame {frame_j} Patches')
        axes[frame_j].set_xlabel('Column')
        axes[frame_j].set_ylabel('Row')
        axes[frame_j].set_xticks([])
        axes[frame_j].set_yticks([])

    # Set overall title
    fig.suptitle(f'Global Attention from Frame 0 Patch ({patch_row}, {patch_col}) to Frames 0-5 (Head {head_idx})', fontsize=16, y=1.05)

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Image saved to: {save_path}")
    else:
        plt.show()

    # Print selected patch information
    print(f"Selected patch: Frame 0, position ({patch_row}, {patch_col}), global index {query_idx}")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def _forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: Tensor, pos=None) -> Tensor:
        return checkpoint(self._forward, x, pos, use_reentrant=False)

class LoRAConfig:
    def __init__(self, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout

class LoraAttention(Attention):
    lora_config = None
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop, norm_layer, qk_norm, fused_attn, rope)
        if self.lora_config.r > 0:
            self.lora_A_v = nn.Parameter(torch.zeros((self.lora_config.r, dim)))
            self.lora_B_v = nn.Parameter(torch.zeros((dim, self.lora_config.r)))
            self.lora_A_q = nn.Parameter(torch.zeros((self.lora_config.r, dim)))
            self.lora_B_q = nn.Parameter(torch.zeros((dim, self.lora_config.r)))
            self.lora_scaling = self.lora_config.alpha / self.lora_config.r
            if self.lora_config.dropout > 0.:
                self.lora_dropout = nn.Dropout(p=self.lora_config.dropout)
            else:
                self.lora_dropout = lambda x: x
            self.lora_r = self.lora_config.r
            self.reset_parameters()
        else:
            self.lora_r = 0

    def reset_parameters(self):
        if hasattr(self, 'lora_A_q'):
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q)
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_v)

    def _forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.lora_r > 0:
            previous_dtype = q.dtype
            q = q.to(self.lora_A_q.data.dtype)
            q_after_A = F.linear(self.lora_dropout(x), self.lora_A_q)
            q_after_B = F.linear(q_after_A, self.lora_B_q)
            q_after_B = q_after_B.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = q + q_after_B * self.lora_scaling
            q = q.to(previous_dtype)

            v = v.to(self.lora_A_v.data.dtype)
            v_after_A = F.linear(self.lora_dropout(x), self.lora_A_v)
            v_after_B = F.linear(v_after_A, self.lora_B_v)
            v_after_B = v_after_B.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v + v_after_B * self.lora_scaling
            v = v.to(previous_dtype)


        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: Tensor, pos=None) -> Tensor:
        return checkpoint(self._forward, x, pos, use_reentrant=False)

class MemEffAttention(Attention):
    def _forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = qkv.unbind(2)
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        return checkpoint(self._forward, x, attn_bias, pos, use_reentrant=False)