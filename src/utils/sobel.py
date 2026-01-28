"""Sobel edge extraction utilities."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelOperator(nn.Module):
    """Compute Sobel edges using fixed-weight conv2d.

    Outputs either magnitude (1 channel) or raw gradients (2 channels).
    """

    def __init__(self, in_channels: int = 3, return_magnitude: bool = True, normalize: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.return_magnitude = return_magnitude
        self.normalize = normalize

        gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

        # Compute gradients per input channel (no channel mixing) and reduce afterwards.
        # Conv2d in PyTorch implements cross-correlation, which matches the typical Sobel
        # mask usage in image processing.
        weight = torch.zeros(2 * in_channels, 1, 3, 3)
        for c in range(in_channels):
            weight[2 * c + 0, 0] = gx
            weight[2 * c + 1, 0] = gy

        self.register_buffer("weight", weight)
        self.conv = nn.Conv2d(
            in_channels,
            2 * in_channels,
            kernel_size=3,
            padding=0,
            bias=False,
            groups=in_channels,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected in range [-1,1] or [0,1]
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        grads = self.conv(x)
        gx = grads[:, 0::2, :, :]
        gy = grads[:, 1::2, :, :]
        if self.return_magnitude:
            mag = torch.sqrt(gx**2 + gy**2 + 1e-12)
            mag = mag.mean(dim=1, keepdim=True)
            if self.normalize:
                mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            return mag

        gx_mean = gx.mean(dim=1, keepdim=True)
        gy_mean = gy.mean(dim=1, keepdim=True)
        grads_out = torch.cat([gx_mean, gy_mean], dim=1)
        if self.normalize:
            grads_out = grads_out / (grads_out.abs().amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return grads_out


def sobel_edges(x: torch.Tensor, return_magnitude: bool = True, normalize: bool = True) -> torch.Tensor:
    """Functional interface for Sobel edges."""
    op = SobelOperator(in_channels=x.shape[1], return_magnitude=return_magnitude, normalize=normalize)
    op = op.to(device=x.device, dtype=x.dtype)
    return op(x)


__all__ = ["SobelOperator", "sobel_edges"]
