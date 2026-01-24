"""Quick inference script for the EdgeAwareGenerator.

Loads a trained generator checkpoint and translates a single image (or a pair of
stereo images) using optional Sobel edges. Saves output images to disk.
"""

import argparse
import os
from typing import Optional

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.models.generator import EdgeAwareGenerator
from src.utils.sobel import sobel_edges
from src.utils.checkpoints import load_checkpoint


def _load_image(path: str, resize: Optional[tuple] = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize[1], resize[0]), resample=Image.BICUBIC)
    tensor = TF.to_tensor(img)  # [0,1]
    tensor = tensor * 2.0 - 1.0
    return tensor


def _save_image(tensor: torch.Tensor, path: str) -> None:
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = (tensor + 1.0) * 0.5
    tensor = torch.clamp(tensor, 0.0, 1.0)
    img = TF.to_pil_image(tensor)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick inference for SyntStereo2Real generator")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint")
    parser.add_argument("--input", required=True, help="Path to input image (left view)")
    parser.add_argument("--input_right", default=None, help="Optional path to right image")
    parser.add_argument("--output", required=True, help="Output path for translated left image")
    parser.add_argument("--output_right", default=None, help="Optional output path for translated right image")
    parser.add_argument("--device", default="cuda", help="Device string")
    parser.add_argument("--resize", nargs=2, type=int, default=None, help="Resize (H W)")
    parser.add_argument("--use_edge", action="store_true", help="Use Sobel edges for edge encoder")
    parser.add_argument("--target_domain", default="b", choices=["a", "b"], help="Target domain style")
    return parser.parse_args()


def _load_generator(checkpoint_path: str, device: str) -> EdgeAwareGenerator:
    ckpt = load_checkpoint(checkpoint_path, device=device)
    if "config" in ckpt:
        model_cfg = ckpt["config"].get("model", {})
    else:
        model_cfg = {}
    gen = EdgeAwareGenerator(**model_cfg)
    state = ckpt.get("gen") or ckpt.get("generator") or ckpt
    gen.load_state_dict(state, strict=False)
    gen.to(device)
    gen.eval()
    return gen


def _translate_image(
    gen: EdgeAwareGenerator,
    image: torch.Tensor,
    device: str,
    use_edge: bool,
    target_domain: str,
) -> torch.Tensor:
    image = image.unsqueeze(0).to(device)
    if use_edge:
        edge = sobel_edges(image, return_magnitude=True, normalize=True)
    else:
        edge = None
    style = gen.get_style(target_domain, image.size(0), device)
    content = gen.encode(image)
    edge_code = gen.encode_edge(edge) if edge is not None else None
    fused = gen.fuse(content, edge_code)
    out = gen.decode(fused, style)
    return out


def main() -> None:
    args = _parse_args()
    device = args.device
    
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    if device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    gen = _load_generator(args.checkpoint, device)

    resize = tuple(args.resize) if args.resize is not None else None
    left = _load_image(args.input, resize=resize)
    out_left = _translate_image(gen, left, device, args.use_edge, args.target_domain)
    _save_image(out_left, args.output)

    if args.input_right is not None:
        if args.output_right is None:
            raise ValueError("--output_right must be provided when --input_right is set")
        right = _load_image(args.input_right, resize=resize)
        out_right = _translate_image(gen, right, device, args.use_edge, args.target_domain)
        _save_image(out_right, args.output_right)


if __name__ == "__main__":
    main()
