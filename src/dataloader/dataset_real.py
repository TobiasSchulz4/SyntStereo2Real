"""Dataset for real (target) domain images used in translation training."""

import os
from typing import Optional, Tuple, List, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import build_image_transform
from src.utils.sobel import sobel_edges

__all__ = ["RealImageDataset"]


def _load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


class RealImageDataset(Dataset):
    """Dataset yielding real domain images with optional Sobel edge maps."""

    def __init__(
        self,
        root: str,
        image_dir: str = "images",
        file_list: Optional[str] = None,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        compute_edges: bool = True,
        return_edges: bool = True,
    ) -> None:
        self.root = root
        self.image_dir = image_dir
        self.resize = resize
        self.normalize = normalize
        self.compute_edges = compute_edges
        self.return_edges = return_edges

        if file_list is not None:
            with open(file_list, "r") as f:
                self.ids = [ln.strip() for ln in f.readlines() if ln.strip()]
        else:
            img_root = os.path.join(root, image_dir)
            self.ids = [os.path.splitext(fn)[0] for fn in os.listdir(img_root) if fn.lower().endswith((".png", ".jpg", ".jpeg"))]
            self.ids.sort()
        self.ids = self.ids[:10]

        self.img_tf = build_image_transform(resize=resize, normalize=normalize)

    def __len__(self) -> int:
        return len(self.ids)

    def _path(self, idx: int) -> str:
        name = self.ids[idx]
        img_root = os.path.join(self.root, self.image_dir)
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(img_root, name + ext)
            if os.path.exists(p):
                return p
        return os.path.join(img_root, name + ".png")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self._path(idx)
        img = _load_image(path)
        xb = self.img_tf(img)

        sample: Dict[str, torch.Tensor] = {"xb": xb}

        if self.compute_edges and self.return_edges:
            edge = sobel_edges(xb.unsqueeze(0), return_magnitude=True, normalize=True).squeeze(0)
            sample["xb_edge"] = edge

        return sample
