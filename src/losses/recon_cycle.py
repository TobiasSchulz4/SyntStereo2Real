"""Reconstruction and cycle-consistency losses."""
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ["reconstruction_loss", "cycle_loss"]


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L1 reconstruction loss.

    Args:
        pred: reconstructed image tensor (B,C,H,W)
        target: original image tensor (B,C,H,W)

    Returns:
        L1 loss scalar tensor.
    """
    return F.l1_loss(pred, target)


def cycle_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L1 cycle-consistency loss.

    Args:
        reconstructed: cycled image tensor (B,C,H,W)
        target: original image tensor (B,C,H,W)

    Returns:
        L1 loss scalar tensor.
    """
    return F.l1_loss(reconstructed, target)
