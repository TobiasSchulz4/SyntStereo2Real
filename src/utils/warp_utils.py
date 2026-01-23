"""Warping utility helpers.

Provides grid construction and validity mask utilities used by warping loss.
"""
from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["build_warp_grid", "valid_warp_mask"]


def build_warp_grid(disparity: torch.Tensor) -> torch.Tensor:
    """Build a normalized sampling grid for grid_sample from disparity.

    Args:
        disparity: Tensor of shape (B, 1, H, W) or (B, H, W) containing
            left->right disparity in pixels. Positive disparity shifts sampling
            to the right (x + d).

    Returns:
        grid: Tensor of shape (B, H, W, 2) with normalized coordinates in [-1, 1].
    """
    if disparity.dim() == 3:
        disparity = disparity.unsqueeze(1)
    if disparity.dim() != 4:
        raise ValueError(f"disparity must be 3D or 4D tensor, got {disparity.shape}")

    b, _, h, w = disparity.shape
    device = disparity.device
    dtype = disparity.dtype

    # Create base grid
    xs = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
    ys = torch.linspace(0, h - 1, h, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)

    # Apply disparity (left->right)
    x_shifted = grid_x + disparity.squeeze(1)

    # Normalize to [-1,1]
    x_norm = 2.0 * x_shifted / (w - 1) - 1.0
    y_norm = 2.0 * grid_y / (h - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    return grid


def valid_warp_mask(disparity: torch.Tensor) -> torch.Tensor:
    """Compute a validity mask for warping based on disparity.

    Args:
        disparity: Tensor of shape (B, 1, H, W) or (B, H, W).

    Returns:
        mask: Float tensor of shape (B, 1, H, W) with 1 for valid sampling
            positions and 0 otherwise.
    """
    if disparity.dim() == 3:
        disparity = disparity.unsqueeze(1)
    b, _, h, w = disparity.shape
    device = disparity.device

    xs = torch.linspace(0, w - 1, w, device=device, dtype=disparity.dtype)
    ys = torch.linspace(0, h - 1, h, device=device, dtype=disparity.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)

    x_shifted = grid_x + disparity.squeeze(1)
    valid = (x_shifted >= 0.0) & (x_shifted <= (w - 1))
    return valid.unsqueeze(1).float()
