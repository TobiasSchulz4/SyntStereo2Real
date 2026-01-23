"""PatchGAN discriminator for SyntStereo2Real.

Implements a 70x70 PatchGAN-style discriminator with InstanceNorm and
optional spectral normalization. Outputs a logits map suitable for
BCEWithLogitsLoss.
"""
from typing import Optional

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with 4 downsampling conv blocks."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        def conv_layer(in_c: int, out_c: int, k: int, s: int, p: int, norm: bool = True) -> nn.Sequential:
            conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=not norm)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers = [conv]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Architecture: C64-C128-C256-C512-C1
        self.model = nn.Sequential(
            conv_layer(in_channels, base_channels, k=4, s=2, p=1, norm=False),
            conv_layer(base_channels, base_channels * 2, k=4, s=2, p=1, norm=True),
            conv_layer(base_channels * 2, base_channels * 4, k=4, s=2, p=1, norm=True),
            conv_layer(base_channels * 4, base_channels * 8, k=4, s=1, p=1, norm=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = ["PatchGANDiscriminator"]
