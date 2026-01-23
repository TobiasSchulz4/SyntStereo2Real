"""Prepare KITTI dataset layout.

Creates a simple folder layout compatible with RealImageDataset or stereo datasets:
- out_root/left
- out_root/right
- out_root/disp (optional if disp files provided)
- out_root/files.txt (stems)

This script scans the source root for left/right image files (and optional disparity files)
and either copies or symlinks them into the output structure.
"""

import argparse
import os
import shutil
from typing import List, Tuple


def _find_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    files: List[str] = []
    for base, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if any(lower.endswith(ext) for ext in exts):
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


def prepare(
    src_root: str,
    out_root: str,
    left_exts: Tuple[str, ...],
    right_exts: Tuple[str, ...],
    disp_exts: Tuple[str, ...],
    use_symlink: bool,
) -> None:
    """Prepare KITTI dataset.

    Args:
        src_root: path to raw KITTI dataset root.
        out_root: output directory to create left/right/disp folders.
        left_exts: file extensions for left images.
        right_exts: file extensions for right images.
        disp_exts: file extensions for disparity (optional).
        use_symlink: if True, create symlinks instead of copying files.
    """
    left_root = os.path.join(src_root, "left")
    right_root = os.path.join(src_root, "right")
    disp_root = os.path.join(src_root, "disp")

    left_files = _find_files(left_root, left_exts) if os.path.isdir(left_root) else []
    right_files = _find_files(right_root, right_exts) if os.path.isdir(right_root) else []
    disp_files = _find_files(disp_root, disp_exts) if os.path.isdir(disp_root) else []

    if not left_files:
        left_files = _find_files(src_root, left_exts)
    if not right_files:
        right_files = _find_files(src_root, right_exts)
    if not disp_files:
        disp_files = _find_files(src_root, disp_exts)

    out_left = os.path.join(out_root, "left")
    out_right = os.path.join(out_root, "right")
    out_disp = os.path.join(out_root, "disp")
    _ensure_dir(out_left)
    _ensure_dir(out_right)
    if disp_files:
        _ensure_dir(out_disp)

    stems = set()
    for path in left_files:
        stem = _stem(path)
        stems.add(stem)
        _link_or_copy(path, os.path.join(out_left, f"{stem}.png"), use_symlink)
    for path in right_files:
        stem = _stem(path)
        stems.add(stem)
        _link_or_copy(path, os.path.join(out_right, f"{stem}.png"), use_symlink)
    for path in disp_files:
        stem = _stem(path)
        stems.add(stem)
        ext = os.path.splitext(path)[1].lower()
        _link_or_copy(path, os.path.join(out_disp, f"{stem}{ext}"), use_symlink)

    _ensure_dir(out_root)
    with open(os.path.join(out_root, "files.txt"), "w", encoding="utf-8") as f:
        for stem in sorted(stems):
            f.write(f"{stem}\n")

    print(
        f"Prepared KITTI dataset at {out_root} with {len(stems)} samples. "
        f"Left: {len(left_files)}, Right: {len(right_files)}, Disp: {len(disp_files)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KITTI dataset layout")
    parser.add_argument("--src_root", type=str, required=True, help="Path to KITTI root")
    parser.add_argument("--out_root", type=str, required=True, help="Output directory")
    parser.add_argument("--use_symlink", action="store_true", help="Use symlinks instead of copying")
    parser.add_argument("--left_exts", type=str, default=".png,.jpg,.jpeg")
    parser.add_argument("--right_exts", type=str, default=".png,.jpg,.jpeg")
    parser.add_argument("--disp_exts", type=str, default=".png,.npy")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    left_exts = tuple(ext.strip().lower() for ext in args.left_exts.split(",") if ext.strip())
    right_exts = tuple(ext.strip().lower() for ext in args.right_exts.split(",") if ext.strip())
    disp_exts = tuple(ext.strip().lower() for ext in args.disp_exts.split(",") if ext.strip())
    prepare(
        src_root=args.src_root,
        out_root=args.out_root,
        left_exts=left_exts,
        right_exts=right_exts,
        disp_exts=disp_exts,
        use_symlink=args.use_symlink,
    )


if __name__ == "__main__":
    main()
