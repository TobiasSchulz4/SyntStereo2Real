"""Prepare US3D dataset into standard layout.

This script organizes real-domain images into a single folder structure:
  out_root/images
and writes a files.txt manifest listing all image stems.

It supports symlinking or copying files and can scan a source root with
nested folders. Intended to align with RealImageDataset expectations.
"""

import argparse
import os
import shutil
from typing import List, Tuple


def _find_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    files: List[str] = []
    for base, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(exts):
                files.append(os.path.join(base, name))
    return sorted(files)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _link_or_copy(src: str, dst: str, use_symlink: bool) -> None:
    _ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return
    if use_symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def prepare(src_root: str, out_root: str, img_exts: Tuple[str, ...], use_symlink: bool) -> None:
    img_dir_candidates = [
        os.path.join(src_root, "images"),
        os.path.join(src_root, "rgb"),
        src_root,
    ]
    image_files: List[str] = []
    for cand in img_dir_candidates:
        if os.path.isdir(cand):
            image_files = _find_files(cand, img_exts)
            if image_files:
                break

    if not image_files:
        raise FileNotFoundError(f"No images found under {src_root} with exts {img_exts}")

    out_images = os.path.join(out_root, "images")
    _ensure_dir(out_images)

    stems: List[str] = []
    for path in image_files:
        stem = _stem(path)
        stems.append(stem)
        dst = os.path.join(out_images, f"{stem}.png")
        _link_or_copy(path, dst, use_symlink)

    manifest = os.path.join(out_root, "files.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for stem in sorted(set(stems)):
            f.write(stem + "\n")

    print(f"Prepared {len(stems)} images -> {out_root}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare US3D dataset layout")
    parser.add_argument("--src_root", type=str, required=True, help="Path to raw US3D dataset")
    parser.add_argument("--out_root", type=str, required=True, help="Output dataset root")
    parser.add_argument(
        "--exts",
        type=str,
        default=".png,.jpg,.jpeg",
        help="Comma-separated list of image extensions to include",
    )
    parser.add_argument("--use_symlink", action="store_true", help="Use symlinks instead of copying")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    img_exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    prepare(args.src_root, args.out_root, img_exts, args.use_symlink)


if __name__ == "__main__":
    main()
