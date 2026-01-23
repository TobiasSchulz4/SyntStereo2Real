"""Metrics for disparity evaluation.

Implements Median Absolute Deviation (MAD) and 1px/3px accuracy.
All metrics operate on valid pixels only (mask excludes invalid/undefined).
"""

from typing import Dict, Optional, Tuple

import torch

__all__ = ["mad", "accuracy_threshold", "compute_metrics"]


def _valid_mask(gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return validity mask (float tensor) based on gt and optional mask.

    If mask is provided, combines it with finite/positive checks from gt.
    """
    if gt.dim() == 4 and gt.size(1) == 1:
        gt_vals = gt[:, 0]
    else:
        gt_vals = gt
    valid = torch.isfinite(gt_vals)
    # Treat non-positive disparity as invalid by default (common in datasets)
    valid = valid & (gt_vals > 0)
    if mask is not None:
        if mask.dim() == 4 and mask.size(1) == 1:
            m = mask[:, 0] > 0.5
        else:
            m = mask > 0.5
        valid = valid & m
    return valid


def mad(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Median Absolute Deviation between predicted and ground-truth disparity.

    Args:
        pred: predicted disparity, shape (B,1,H,W) or (B,H,W)
        gt: ground-truth disparity, same shape
        mask: optional validity mask (B,1,H,W) or (B,H,W)
    Returns:
        scalar tensor (median absolute error over valid pixels)
    """
    if pred.dim() == 4 and pred.size(1) == 1:
        pred_vals = pred[:, 0]
    else:
        pred_vals = pred
    if gt.dim() == 4 and gt.size(1) == 1:
        gt_vals = gt[:, 0]
    else:
        gt_vals = gt

    valid = _valid_mask(gt, mask)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    abs_err = (pred_vals - gt_vals).abs()
    return abs_err[valid].median()


def accuracy_threshold(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute accuracy under a pixel error threshold.

    Returns the fraction of valid pixels with |pred-gt| <= threshold.
    """
    if pred.dim() == 4 and pred.size(1) == 1:
        pred_vals = pred[:, 0]
    else:
        pred_vals = pred
    if gt.dim() == 4 and gt.size(1) == 1:
        gt_vals = gt[:, 0]
    else:
        gt_vals = gt

    valid = _valid_mask(gt, mask)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    abs_err = (pred_vals - gt_vals).abs()
    acc = (abs_err <= threshold).float()
    return acc[valid].mean()


def compute_metrics(
    pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Compute MAD, 1px accuracy, and 3px accuracy.

    Returns:
        dict with keys: "mad", "acc_1px", "acc_3px"
    """
    return {
        "mad": mad(pred, gt, mask),
        "acc_1px": accuracy_threshold(pred, gt, 1.0, mask),
        "acc_3px": accuracy_threshold(pred, gt, 3.0, mask),
    }
