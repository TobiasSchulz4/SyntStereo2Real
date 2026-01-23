"""Prepare SceneFlow/Driving (synthetic) dataset into standard layout.

This script scans a source directory for left/right images and disparity maps and
organizes them into a simple structure:
  out_root/
    left/
    right/
    disp/
    files.txt

It can either copy files or create symlinks.
"""

import argparse
import os
import shutil
from typing import List, Tuple


def _find_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    if not os.path.isdir(root):
        return []
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lname = name.lower()
            if any(lname.endswith(ext) for ext in exts):
                matches.append(os.path.join(dirpath, name))
    return sorted(matches)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _stem(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def _link_or_copy(src: str, dst: str, use_symlink: bool) -> None:
    if os.path.exists(dst):
        return
    _ensure_dir(os.path.dirname(dst))
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
    left_root = os.path.join(src_root, "left")
    right_root = os.path.join(src_root, "right")
    disp_root = os.path.join(src_root, "disp")

    left_files = _find_files(left_root, left_exts) or _find_files(src_root, left_exts)
    right_files = _find_files(right_root, right_exts) or _find_files(src_root, right_exts)
    disp_files = _find_files(disp_root, disp_exts) or _find_files(src_root, disp_exts)

    left_map = {_stem(p): p for p in left_files}
    right_map = {_stem(p): p for p in right_files}
    disp_map = {_stem(p): p for p in disp_files}

    stems = sorted(set(left_map.keys()) | set(right_map.keys()) | set(disp_map.keys()))

    out_left = os.path.join(out_root, "left")
    out_right = os.path.join(out_root, "right")
    out_disp = os.path.join(out_root, "disp")
    _ensure_dir(out_left)
    _ensure_dir(out_right)
    _ensure_dir(out_disp)

    for stem in stems:
        if stem in left_map:
            _link_or_copy(left_map[stem], os.path.join(out_left, f"{stem}.png"), use_symlink)
        if stem in right_map:
            _link_or_copy(right_map[stem], os.path.join(out_right, f"{stem}.png"), use_symlink)
        if stem in disp_map:
            disp_src = disp_map[stem]
            disp_ext = os.path.splitext(disp_src)[1].lower()
            _link_or_copy(disp_src, os.path.join(out_disp, f"{stem}{disp_ext}"), use_symlink)

    manifest = os.path.join(out_root, "files.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for stem in stems:
            f.write(f"{stem}\n")

    print(
        f"Prepared SceneFlow dataset with {len(stems)} samples -> {out_root} (symlink={use_symlink})"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SceneFlow/Driving dataset layout")
    parser.add_argument("--src_root", type=str, required=True, help="Path to raw SceneFlow data")
    parser.add_argument("--out_root", type=str, required=True, help="Output dataset root")
    parser.add_argument("--use_symlink", action="store_true", help="Use symlinks instead of copying")
    parser.add_argument(
        "--left_exts",
        type=str,
        default=".png,.jpg,.jpeg",
        help="Comma-separated extensions for left images",
    )
    parser.add_argument(
        "--right_exts",
        type=str,
        default=".png,.jpg,.jpeg",
        help="Comma-separated extensions for right images",
    )
    parser.add_argument(
        "--disp_exts",
        type=str,
        default=".pfm,.npy,.png",
        help="Comma-separated extensions for disparity",
    )
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
