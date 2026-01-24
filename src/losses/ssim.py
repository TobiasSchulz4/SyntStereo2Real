"""Differentiable SSIM implementation.

Provides a basic SSIM computation with a Gaussian window. Returns a per-pixel
SSIM map in range [0, 1] for images assumed to be in [0, 1].
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ["ssim"]


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.view(1, 1, -1)
    window_2d = window_1d.transpose(1, 2) @ window_1d
    return window_2d


def _create_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    window_2d = _gaussian_window(window_size, sigma=sigma, device=device, dtype=dtype)
    window_2d = window_2d.unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute SSIM map for two images.

    Args:
        img1: (B, C, H, W) tensor in [0, 1]
        img2: (B, C, H, W) tensor in [0, 1]
        window_size: Gaussian window size
        C1, C2: SSIM stability constants

    Returns:
        SSIM map tensor of shape (B, 1, H, W), values in [0, 1].
    """
    if img1.shape != img2.shape:
        raise ValueError("SSIM expects inputs with the same shape")

    b, c, h, w = img1.shape
    window = _create_window(window_size, sigma, c, img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=c)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=c) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_map.clamp(0, 1)

    # Reduce to single channel map by averaging over channels
    ssim_map = ssim_map.mean(dim=1, keepdim=True)
    return ssim_map
