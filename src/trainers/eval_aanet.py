"""Evaluate AANet predictions on a real test set.

This script computes MAD, 1px, and 3px accuracy on a dataset of
predicted disparities vs ground-truth disparities and optionally saves
visualizations.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.losses.metrics import compute_metrics


def _load_disp(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    disp = np.array(Image.open(path)).astype(np.float32)
    return disp


class DispEvalDataset(Dataset):
    """Dataset for disparity evaluation.

    Expects directory structure:
      root/
        pred/  (predicted disparity maps .npy or .png)
        gt/    (ground-truth disparity maps .npy or .png)
        mask/  (optional validity masks .png/.npy)
    """

    def __init__(self, root: str, pred_dir: str = "pred", gt_dir: str = "gt", mask_dir: str = "mask") -> None:
        self.root = root
        self.pred_dir = os.path.join(root, pred_dir)
        self.gt_dir = os.path.join(root, gt_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.has_mask = os.path.isdir(self.mask_dir)

        self.ids = []
        for name in sorted(os.listdir(self.gt_dir)):
            stem, _ = os.path.splitext(name)
            self.ids.append(stem)

    def __len__(self) -> int:
        return len(self.ids)

    def _resolve(self, base_dir: str, stem: str) -> str:
        for ext in [".npy", ".png", ".jpg", ".jpeg"]:
            path = os.path.join(base_dir, stem + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No file for {stem} in {base_dir}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.ids[idx]
        pred_path = self._resolve(self.pred_dir, stem)
        gt_path = self._resolve(self.gt_dir, stem)

        pred = _load_disp(pred_path)
        gt = _load_disp(gt_path)

        pred_t = torch.from_numpy(pred).unsqueeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0)

        sample = {"pred": pred_t, "gt": gt_t}

        if self.has_mask:
            mask_path = self._resolve(self.mask_dir, stem)
            mask = _load_disp(mask_path)
            sample["mask"] = torch.from_numpy(mask).unsqueeze(0)
        return sample


def _aggregate_metrics(metrics_list: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    if not metrics_list:
        return {"mad": 0.0, "acc_1px": 0.0, "acc_3px": 0.0}
    keys = metrics_list[0].keys()
    agg = {k: torch.stack([m[k] for m in metrics_list]).mean().item() for k in keys}
    return agg


def evaluate(root: str, batch_size: int = 1, num_workers: int = 0, device: str = "cpu") -> Dict[str, float]:
    dataset = DispEvalDataset(root)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    torch_device = torch.device(device)

    metrics_list = []
    for batch in loader:
        pred = batch["pred"].to(torch_device)
        gt = batch["gt"].to(torch_device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(torch_device)
        metrics = compute_metrics(pred, gt, mask)
        metrics_list.append(metrics)

    return _aggregate_metrics(metrics_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AANet predictions on a dataset")
    parser.add_argument("--root", type=str, required=True, help="Evaluation root with pred/gt directories")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    if device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    metrics = evaluate(args.root, args.batch_size, args.num_workers, device)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
