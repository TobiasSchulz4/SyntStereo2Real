"""Adversarial loss helpers for PatchGAN discriminators."""
from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["gan_discriminator_loss", "gan_generator_loss"]


def gan_discriminator_loss(
    pred_real: torch.Tensor,
    pred_fake: torch.Tensor,
    loss_fn: nn.Module = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute discriminator loss for real and fake predictions.

    Args:
        pred_real: discriminator logits for real images (N, 1, H, W)
        pred_fake: discriminator logits for fake images (N, 1, H, W)
        loss_fn: optional loss function; defaults to BCEWithLogitsLoss

    Returns:
        total_loss, loss_real, loss_fake
    """
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    real_targets = torch.ones_like(pred_real)
    fake_targets = torch.zeros_like(pred_fake)

    loss_real = loss_fn(pred_real, real_targets)
    loss_fake = loss_fn(pred_fake, fake_targets)
    total = 0.5 * (loss_real + loss_fake)
    return total, loss_real, loss_fake


def gan_generator_loss(
    pred_fake: torch.Tensor,
    loss_fn: nn.Module = None,
) -> torch.Tensor:
    """Compute generator adversarial loss (non-saturating BCE).

    Args:
        pred_fake: discriminator logits for generated images
        loss_fn: optional loss function; defaults to BCEWithLogitsLoss

    Returns:
        generator loss tensor
    """
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    target_real = torch.ones_like(pred_fake)
    return loss_fn(pred_fake, target_real)
