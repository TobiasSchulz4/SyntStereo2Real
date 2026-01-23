"""Translate full synthetic stereo dataset using a trained generator.

This script loads a generator checkpoint, iterates over the synthetic dataset,
translates left/right images into target domain, and saves translated images
along with original disparities (and optional masks) for downstream stereo training.
"""

import argparse
import os
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml

from src.dataloader.dataset_synthetic import SyntheticStereoDataset
from src.dataloader.transforms import random_hflip
from src.models.generator import EdgeAwareGenerator
from src.utils.sobel import sobel_edges
from src.utils.checkpoints import load_checkpoint


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate synthetic dataset using trained generator")
    parser.add_argument("--config", type=str, required=True, help="Path to translation config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--output_root", type=str, required=True, help="Output directory for translated dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for translation")
    parser.add_argument("--max_items", type=int, default=-1, help="Optional limit of samples to translate")
    parser.add_argument("--use_edge", action="store_true", help="Use Sobel edges in translation")
    parser.add_argument("--flip", action="store_true", help="Apply random horizontal flips during translation")
    return parser.parse_args()


def _save_batch(images: torch.Tensor, out_dir: str, start_idx: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i in range(images.size(0)):
        save_path = os.path.join(out_dir, f"{start_idx + i:06d}.png")
        save_image(images[i], save_path, normalize=True, value_range=(-1, 1))


def _save_disparity(disp: torch.Tensor, out_dir: str, start_idx: int) -> None:
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    disp_np = disp.squeeze(1).cpu().numpy().astype("float32")
    for i in range(disp_np.shape[0]):
        save_path = os.path.join(out_dir, f"{start_idx + i:06d}.npy")
        np.save(save_path, disp_np[i])


def translate_dataset(
    config_path: str,
    checkpoint_path: str,
    output_root: str,
    device: str,
    num_workers: int,
    batch_size: int,
    max_items: int,
    use_edge: bool,
    use_flip: bool,
) -> None:
    cfg = _load_config(config_path)
    synthetic_cfg = cfg.get("synthetic", {})
    model_cfg = cfg.get("model", {})

    dataset = SyntheticStereoDataset(
        root=synthetic_cfg.get("root", "datasets/syntcities"),
        left_dir=synthetic_cfg.get("left_dir", "left"),
        right_dir=synthetic_cfg.get("right_dir", "right"),
        disp_dir=synthetic_cfg.get("disp_dir", "disp"),
        file_list=synthetic_cfg.get("file_list", None),
        resize=tuple(synthetic_cfg.get("resize", [256, 512])),
        normalize=synthetic_cfg.get("normalize", True),
        compute_edges=False,
        return_edges=False,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    generator = EdgeAwareGenerator(**model_cfg)
    ckpt = load_checkpoint(checkpoint_path, device=device)
    if "gen" in ckpt:
        generator.load_state_dict(ckpt["gen"], strict=False)
    elif "generator" in ckpt:
        generator.load_state_dict(ckpt["generator"], strict=False)
    else:
        generator.load_state_dict(ckpt, strict=False)

    generator.to(device)
    generator.eval()

    out_left = os.path.join(output_root, "left")
    out_right = os.path.join(output_root, "right")
    out_disp = os.path.join(output_root, "disp")

    os.makedirs(output_root, exist_ok=True)

    global_idx = 0
    with torch.no_grad():
        for batch in loader:
            if max_items > 0 and global_idx >= max_items:
                break
            if use_flip:
                batch = random_hflip(batch, p=0.5)
            xl = batch["xl"].to(device)
            xr = batch["xr"].to(device)
            xd = batch["xd"].to(device)

            if use_edge:
                xl_edge = sobel_edges(xl, return_magnitude=True, normalize=True)
                xr_edge = sobel_edges(xr, return_magnitude=True, normalize=True)
            else:
                xl_edge = None
                xr_edge = None

            style_b = generator.get_style("b", xl.size(0), xl.device)
            c_l = generator.encode(xl)
            c_r = generator.encode(xr)
            if xl_edge is not None:
                e_l = generator.encode_edge(xl_edge)
                e_r = generator.encode_edge(xr_edge)
                c_l = generator.fuse(c_l, e_l)
                c_r = generator.fuse(c_r, e_r)
            x_l_b = generator.decode(c_l, style_b)
            x_r_b = generator.decode(c_r, style_b)

            _save_batch(x_l_b, out_left, global_idx)
            _save_batch(x_r_b, out_right, global_idx)
            _save_disparity(xd, out_disp, global_idx)

            global_idx += xl.size(0)


def main() -> None:
    args = _parse_args()
    translate_dataset(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_items=args.max_items,
        use_edge=args.use_edge,
        use_flip=args.flip,
    )


if __name__ == "__main__":
    main()
