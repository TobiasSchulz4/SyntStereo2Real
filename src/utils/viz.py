"""Visualization helpers for SyntStereo2Real.

Utilities to save image grids, edge overlays, and warping debug outputs.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np

__all__ = ["save_image_grid", "tensor_to_pil", "overlay_edges"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image_grid(images: Dict[str, torch.Tensor], out_dir: str, step: int, max_items: int = 4) -> None:
    """Save a grid of images to disk.

    Args:
        images: dict of name -> tensor (B,C,H,W) in [-1,1] or [0,1].
        out_dir: output directory.
        step: step index used in filename.
        max_items: max batch items per grid.
    """
    _ensure_dir(out_dir)
    for name, tensor in images.items():
        if tensor is None:
            continue
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor[:max_items]
        # normalize to [0,1] for saving
        grid = vutils.make_grid(tensor, nrow=max_items, normalize=True, value_range=(-1, 1))
        out_path = os.path.join(out_dir, f"{name}_{step:07d}.png")
        vutils.save_image(grid, out_path)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW tensor to a PIL Image (expects values in [-1,1] or [0,1])."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1.0) * 0.5
    tensor = tensor.clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def overlay_edges(image: torch.Tensor, edges: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Overlay edge map on image tensor for visualization.

    Args:
        image: (C,H,W) or (B,C,H,W) tensor in [-1,1] or [0,1].
        edges: (1,H,W) or (B,1,H,W) tensor in [0,1].
        alpha: blending factor for edges.

    Returns:
        Tensor with same shape as image (B,C,H,W).
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if edges.dim() == 3:
        edges = edges.unsqueeze(0)
    # normalize image to [0,1]
    img = image.clone()
    if img.min() < 0:
        img = (img + 1.0) * 0.5
    img = img.clamp(0, 1)
    # edges to 3-channel heat
    edge_rgb = edges.repeat(1, 3, 1, 1).clamp(0, 1)
    blended = (1 - alpha) * img + alpha * edge_rgb
    return blended
