"""Synthetic stereo dataset loader.

Yields left/right images, disparity and edge maps for each sample.
Expected directory layout (configurable via root and lists):
  root/
    left/xxx.png
    right/xxx.png
    disp/xxx.npy or xxx.png
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import build_image_transform, build_disparity_transform
from src.utils.sobel import sobel_edges


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_disparity(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        disp = np.load(path)
    else:
        disp = np.array(Image.open(path))
    return disp.astype(np.float32)


class SyntheticStereoDataset(Dataset):
    """Dataset for synthetic stereo pairs and disparity.

    Returns dict with keys: xl, xr, xd, xl_edge, xr_edge.
    """

    def __init__(
        self,
        root: str,
        left_dir: str = "left",
        right_dir: str = "right",
        disp_dir: str = "disp",
        file_list: Optional[str] = None,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        compute_edges: bool = True,
        return_edges: bool = True,
    ) -> None:
        self.root = root
        self.left_dir = os.path.join(root, left_dir)
        self.right_dir = os.path.join(root, right_dir)
        self.disp_dir = os.path.join(root, disp_dir)
        self.resize = resize
        self.normalize = normalize
        self.compute_edges = compute_edges
        self.return_edges = return_edges

        if file_list is not None:
            with open(file_list, "r", encoding="utf-8") as f:
                self.ids = [line.strip() for line in f if line.strip()]
        else:
            self.ids = sorted([os.path.splitext(f)[0] for f in os.listdir(self.left_dir)])

        self.img_tf = build_image_transform(resize=resize, normalize=normalize)
        self.disp_tf = build_disparity_transform(resize=resize)

    def __len__(self) -> int:
        return len(self.ids)

    def _paths(self, idx: int) -> Tuple[str, str, str]:
        fid = self.ids[idx]
        left = os.path.join(self.left_dir, f"{fid}.png")
        right = os.path.join(self.right_dir, f"{fid}.png")
        disp_npy = os.path.join(self.disp_dir, f"{fid}.npy")
        disp_png = os.path.join(self.disp_dir, f"{fid}.png")
        disp = disp_npy if os.path.exists(disp_npy) else disp_png
        return left, right, disp

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        left_path, right_path, disp_path = self._paths(idx)
        left_img = _load_image(left_path)
        right_img = _load_image(right_path)
        disp = _load_disparity(disp_path)

        xl = self.img_tf(left_img)
        xr = self.img_tf(right_img)
        xd = self.disp_tf(disp)

        sample = {"xl": xl, "xr": xr, "xd": xd}

        if self.compute_edges and self.return_edges:
            xl_edge = sobel_edges(xl.unsqueeze(0), return_magnitude=True).squeeze(0)
            xr_edge = sobel_edges(xr.unsqueeze(0), return_magnitude=True).squeeze(0)
            sample.update({"xl_edge": xl_edge, "xr_edge": xr_edge})

        return sample


__all__ = ["SyntheticStereoDataset"]
