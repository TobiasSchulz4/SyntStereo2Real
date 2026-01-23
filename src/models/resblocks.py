"""Residual blocks used across encoders/decoders."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard residual block with two conv layers and normalization.

    Args:
        channels: number of input/output channels
        norm_layer: normalization module class (default: InstanceNorm2d)
        activation: activation module (default: ReLU)
    """

    def __init__(
        self,
        channels: int,
        norm_layer: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm1 = norm_layer(channels, affine=False)
        self.act1 = activation
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm2 = norm_layer(channels, affine=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + identity


class AdaINResidualBlock(nn.Module):
    """Residual block with AdaIN normalization for decoder.

    Expects external AdaIN module to apply style modulation.
    """

    def __init__(self, channels: int, adain_layer: nn.Module) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.adain = adain_layer
        self.act = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.adain(out, gamma, beta)
        out = self.act(out)
        out = self.conv2(out)
        out = self.adain(out, gamma, beta)
        return out + identity


__all__ = ["ResidualBlock", "AdaINResidualBlock"]
