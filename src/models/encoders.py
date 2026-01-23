"""Encoders for SyntStereo2Real.

Includes ContentEncoder and EdgeEncoder modules used by the EdgeAwareGenerator.
"""
from typing import Optional

import torch
import torch.nn as nn

from .resblocks import ResidualBlock


class ContentEncoder(nn.Module):
    """Content encoder E(x) for RGB images.

    Architecture: Conv7x7 -> Conv4x4 s2 -> Conv4x4 s2 -> 4 ResBlocks.
    Output channels: 256; spatial downsample by 4.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_res_blocks: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.ReLU(inplace=True),
        )
        res_blocks = [ResidualBlock(base_channels * 4, norm_layer=nn.InstanceNorm2d) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.res_blocks(x)
        return x


class EdgeEncoder(nn.Module):
    """Edge encoder E_e(edge) for Sobel edge maps.

    Uses lower channel widths and projects to content channel count.
    Output channels match content encoder (default 256) to allow addition.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        out_channels: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, out_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )
        res_blocks = [ResidualBlock(out_channels, norm_layer=nn.InstanceNorm2d) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.res_blocks(x)
        return x


__all__ = ["ContentEncoder", "EdgeEncoder"]
