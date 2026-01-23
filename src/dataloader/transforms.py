"""Image and disparity transforms for datasets.

Provides helpers to build torchvision-style transforms for RGB images and
single-channel disparity maps. Handles resizing, normalization, and optional
horizontal flip (with disparity sign adjustment).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    tensor = torch.from_numpy(arr).permute(2, 0, 1) / 255.0
    return tensor


def _resize_image_tensor(img: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    # img: C,H,W
    img = img.unsqueeze(0)
    resized = F.interpolate(img, size=size, mode=mode, align_corners=False if mode in ["bilinear", "bicubic"] else None)
    return resized.squeeze(0)


def _resize_disparity_tensor(disp: torch.Tensor, size: Tuple[int, int], orig_size: Tuple[int, int]) -> torch.Tensor:
    # disp: 1,H,W
    disp = disp.unsqueeze(0)
    resized = F.interpolate(disp, size=size, mode="nearest")
    # scale disparity by width ratio
    scale = float(size[1]) / float(orig_size[1])
    resized = resized * scale
    return resized.squeeze(0)


def build_image_transform(resize: Optional[Tuple[int, int]] = None, normalize: bool = True):
    """Return a transform function for RGB images.

    Args:
        resize: (H, W) target size; if None no resize.
        normalize: if True, map [0,1] to [-1,1].
    """

    def _transform(img: Image.Image) -> torch.Tensor:
        tensor = _to_tensor(img)
        if resize is not None:
            tensor = _resize_image_tensor(tensor, resize, mode="bicubic")
        if normalize:
            tensor = tensor * 2.0 - 1.0
        return tensor

    return _transform


def build_disparity_transform(resize: Optional[Tuple[int, int]] = None):
    """Return a transform function for disparity maps.

    Args:
        resize: (H, W) target size; if None no resize.
    """

    def _transform(disp: np.ndarray) -> torch.Tensor:
        if disp.ndim == 3:
            disp_arr = disp[:, :, 0]
        else:
            disp_arr = disp
        disp_arr = disp_arr.astype(np.float32)
        h, w = disp_arr.shape[:2]
        tensor = torch.from_numpy(disp_arr).unsqueeze(0)
        if resize is not None:
            tensor = _resize_disparity_tensor(tensor, resize, (h, w))
        return tensor

    return _transform


def random_hflip(sample: dict, p: float = 0.5) -> dict:
    """Random horizontal flip for stereo sample.

    Expects keys: xl, xr, xd (and optional edges). Flips images and disparity.
    Disparity sign is flipped (left->right convention).
    """

    if torch.rand(1).item() > p:
        return sample

    def _flip(t: torch.Tensor) -> torch.Tensor:
        return torch.flip(t, dims=[-1])

    for key in ["xl", "xr", "xl_edge", "xr_edge"]:
        if key in sample and sample[key] is not None:
            sample[key] = _flip(sample[key])

    if "xd" in sample and sample["xd"] is not None:
        sample["xd"] = -_flip(sample["xd"])

    return sample


__all__ = ["build_image_transform", "build_disparity_transform", "random_hflip"]
