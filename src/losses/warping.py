"""Warping operators and loss for stereo consistency."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from src.utils.warp_utils import build_warp_grid, valid_warp_mask
from src.losses.ssim import ssim


def warp_image_left_to_right(left: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
    """Warp left image to right view using disparity.

    Args:
        left: (B, C, H, W) tensor
        disparity: (B, 1, H, W) or (B, H, W)
    Returns:
        warped: (B, C, H, W) tensor
    """
    grid = build_warp_grid(disparity)
    warped = F.grid_sample(left, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped


def warp_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    disparity: torch.Tensor,
    lambda_l1: float = 1.0,
    lambda_ssim: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute warping loss between right image and warped left image.

    Returns:
        total_loss, l1_loss, ssim_loss
    """
    warped = warp_image_left_to_right(left, disparity)
    mask = valid_warp_mask(disparity)
    # Expand mask to channels for L1
    mask_c = mask.expand_as(right)

    # L1
    l1 = (torch.abs(right - warped) * mask_c).sum() / (mask_c.sum() + 1e-6)

    # SSIM expects range [-1,1] or [0,1]; we assume [-1,1] from pipeline and map to [0,1]
    right_01 = (right + 1.0) * 0.5
    warped_01 = (warped + 1.0) * 0.5
    ssim_map = ssim(right_01, warped_01, window_size=11, sigma=1.5)
    # ssim_map is (B,1,H,W); apply mask
    ssim_val = (ssim_map * mask).sum() / (mask.sum() + 1e-6)
    ssim_loss = 1.0 - ssim_val

    total = lambda_l1 * l1 + lambda_ssim * ssim_loss
    return total, l1, ssim_loss


__all__ = ["warp_image_left_to_right", "warp_loss"]
