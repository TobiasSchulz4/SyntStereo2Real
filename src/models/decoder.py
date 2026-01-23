"""Decoder and EdgeAwareGenerator components for SyntStereo2Real.

Implements AdaIN-based decoder with style MLP producing per-layer gamma/beta.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaIN import AdaIN
from .resblocks import AdaINResidualBlock


class StyleMLP(nn.Module):
    """Style MLP that maps style vectors to AdaIN parameters."""

    def __init__(self, style_dim: int, adain_channels: List[int], hidden_dim: int = 256):
        super().__init__()
        self.style_dim = style_dim
        self.adain_channels = adain_channels
        out_dim = 0
        for ch in adain_channels:
            out_dim += 2 * ch  # gamma + beta per layer
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, style: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return list of (gamma, beta) for each AdaIN layer.

        Args:
            style: (B, style_dim)
        Returns:
            list of tuples with gamma/beta shaped (B, C)
        """
        params = self.mlp(style)
        splits: List[Tuple[torch.Tensor, torch.Tensor]] = []
        idx = 0
        for ch in self.adain_channels:
            gamma = params[:, idx : idx + ch]
            idx += ch
            beta = params[:, idx : idx + ch]
            idx += ch
            splits.append((gamma, beta))
        return splits


class Decoder(nn.Module):
    """AdaIN-based decoder for EdgeAwareGenerator."""

    def __init__(
        self,
        in_channels: int = 256,
        base_channels: int = 64,
        num_res_blocks: int = 4,
        style_dim: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.style_dim = style_dim

        self.adain = AdaIN()
        adain_channels = [in_channels] * num_res_blocks
        self.style_mlp = StyleMLP(style_dim=style_dim, adain_channels=adain_channels)

        self.res_blocks = nn.ModuleList(
            [AdaINResidualBlock(in_channels, self.adain) for _ in range(num_res_blocks)]
        )

        # Upsampling blocks
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, padding=2),
            nn.InstanceNorm2d(in_channels // 2, affine=False),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=5, padding=2),
            nn.InstanceNorm2d(in_channels // 4, affine=False),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(in_channels // 4, 3, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Decode feature map with given style vector."""
        adain_params = self.style_mlp(style)
        out = x
        for block, (gamma, beta) in zip(self.res_blocks, adain_params):
            out = block(out, gamma, beta)
        out = self.up1(out)
        out = self.up2(out)
        out = self.final(out)
        return self.tanh(out)


__all__ = ["Decoder", "StyleMLP"]
