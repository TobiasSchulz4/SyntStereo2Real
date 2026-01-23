"""Adaptive Instance Normalization (AdaIN) module."""
from typing import Tuple

import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    Expects per-sample gamma/beta vectors that are broadcastable to feature maps.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply AdaIN normalization.

        Args:
            x: feature map (B, C, H, W)
            gamma: scale (B, C) or (B, C, 1, 1)
            beta: bias (B, C) or (B, C, 1, 1)
        """
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        if beta.dim() == 2:
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + self.eps
        x_norm = (x - mean) / std
        return x_norm * gamma + beta


__all__ = ["AdaIN"]
