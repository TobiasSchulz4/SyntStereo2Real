"""Edge-aware generator composed of content encoder, edge encoder, and AdaIN decoder."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .encoders import ContentEncoder, EdgeEncoder
from .decoder import Decoder


class EdgeAwareGenerator(nn.Module):
    """Edge-aware generator for unpaired translation.

    Encodes content and edges, sums the features, and decodes using AdaIN
    conditioned on a fixed style code per domain.
    """

    def __init__(
        self,
        content_in_channels: int = 3,
        edge_in_channels: int = 1,
        content_base_channels: int = 64,
        edge_base_channels: int = 32,
        content_res_blocks: int = 4,
        edge_res_blocks: int = 2,
        style_dim: int = 8,
        decoder_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.content_encoder = ContentEncoder(
            in_channels=content_in_channels,
            base_channels=content_base_channels,
            num_res_blocks=content_res_blocks,
        )
        self.edge_encoder = EdgeEncoder(
            in_channels=edge_in_channels,
            base_channels=edge_base_channels,
            out_channels=content_base_channels * 4,
            num_res_blocks=edge_res_blocks,
        )
        self.decoder = Decoder(
            in_channels=content_base_channels * 4,
            num_res_blocks=decoder_res_blocks,
            style_dim=style_dim,
        )
        self.style_dim = style_dim

        # Fixed style codes for domains A and B
        sa = torch.randn(1, style_dim)
        sb = torch.randn(1, style_dim)
        self.register_buffer("style_a", sa)
        self.register_buffer("style_b", sb)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.content_encoder(x)

    def encode_edge(self, edge: torch.Tensor) -> torch.Tensor:
        return self.edge_encoder(edge)

    def fuse(self, content: torch.Tensor, edge: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge is None:
            return content
        return content + edge

    def decode(self, fused: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.decoder(fused, style)

    def get_style(self, domain: str, batch_size: int, device: torch.device) -> torch.Tensor:
        if domain.lower() in ["a", "sa", "source", "synthetic"]:
            style = self.style_a
        else:
            style = self.style_b
        return style.to(device).repeat(batch_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge: Optional[torch.Tensor] = None,
        style: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        content = self.encode(x)
        edge_feat = self.encode_edge(edge) if edge is not None else None
        fused = self.fuse(content, edge_feat)
        if style is None:
            style = self.style_a.to(x.device).repeat(x.size(0), 1)
        return self.decode(fused, style)

    @torch.no_grad()
    def translate(
        self,
        x: torch.Tensor,
        edge: Optional[torch.Tensor],
        target_domain: str = "b",
    ) -> torch.Tensor:
        style = self.get_style(target_domain, x.size(0), x.device)
        return self.forward(x, edge=edge, style=style)

    def reconstruct(self, x: torch.Tensor, edge: Optional[torch.Tensor], domain: str) -> torch.Tensor:
        style = self.get_style(domain, x.size(0), x.device)
        return self.forward(x, edge=edge, style=style)

    def cycle(
        self,
        x: torch.Tensor,
        edge: Optional[torch.Tensor],
        source_domain: str,
        target_domain: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Translate to target domain then cycle back to source domain.

        Returns: (x_translated, x_cycled)
        """
        x_ab = self.translate(x, edge=edge, target_domain=target_domain)
        # Edge for cycled image is expected to be computed by caller if needed
        x_ba = self.translate(x_ab, edge=None, target_domain=source_domain)
        return x_ab, x_ba


__all__ = ["EdgeAwareGenerator"]
