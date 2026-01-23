"""Dataset preparation helper for SyntCities.

This script does *not* download data. It organizes an existing SyntCities
extraction into the expected folder layout and optionally writes a file list.
Expected layout after running:

  <output_root>/
    left/   (RGB left images)
    right/  (RGB right images)
    disp/   (disparity maps .npy or .png)
    files.txt (optional list of sample stems)

The script supports either copying or symlinking files from the source tree.
"""

import argparse
import os
import shutil
from typing import List, Tuple


def _find_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(os.path.join(dirpath, name))
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
    left_glob: Tuple[str, ...],
    right_glob: Tuple[str, ...],
    disp_glob: Tuple[str, ...],
    use_symlink: bool,
) -> None:
    left_files = _find_files(os.path.join(src_root, "left"), left_glob)
    right_files = _find_files(os.path.join(src_root, "right"), right_glob)
    disp_files = _find_files(os.path.join(src_root, "disp"), disp_glob)

    if not left_files:
        left_files = _find_files(src_root, left_glob)
    if not right_files:
        right_files = _find_files(src_root, right_glob)
    if not disp_files:
        disp_files = _find_files(src_root, disp_glob)

    left_dir = os.path.join(out_root, "left")
    right_dir = os.path.join(out_root, "right")
    disp_dir = os.path.join(out_root, "disp")
    _ensure_dir(left_dir)
    _ensure_dir(right_dir)
    _ensure_dir(disp_dir)

    stems = set()
    for lf in left_files:
        stem = _stem(lf)
        stems.add(stem)
        _link_or_copy(lf, os.path.join(left_dir, f"{stem}.png"), use_symlink)

    for rf in right_files:
        stem = _stem(rf)
        stems.add(stem)
        _link_or_copy(rf, os.path.join(right_dir, f"{stem}.png"), use_symlink)

    for df in disp_files:
        stem = _stem(df)
        stems.add(stem)
        ext = os.path.splitext(df)[1].lower()
        dst = os.path.join(disp_dir, f"{stem}{ext}")
        _link_or_copy(df, dst, use_symlink)

    file_list = os.path.join(out_root, "files.txt")
    with open(file_list, "w", encoding="utf-8") as f:
        for s in sorted(stems):
            f.write(s + "\n")

    print(f"Prepared {len(stems)} samples at {out_root}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SyntCities dataset")
    parser.add_argument("--src_root", required=True, help="Path to extracted SyntCities root")
    parser.add_argument("--out_root", required=True, help="Output root for standardized layout")
    parser.add_argument("--use_symlink", action="store_true", help="Use symlinks instead of copying")
    parser.add_argument(
        "--left_exts",
        default=".png,.jpg,.jpeg",
        help="Comma-separated extensions for left images",
    )
    parser.add_argument(
        "--right_exts",
        default=".png,.jpg,.jpeg",
        help="Comma-separated extensions for right images",
    )
    parser.add_argument(
        "--disp_exts",
        default=".npy,.png",
        help="Comma-separated extensions for disparity maps",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare(
        src_root=args.src_root,
        out_root=args.out_root,
        left_glob=tuple(e.strip() for e in args.left_exts.split(",")),
        right_glob=tuple(e.strip() for e in args.right_exts.split(",")),
        disp_glob=tuple(e.strip() for e in args.disp_exts.split(",")),
        use_symlink=args.use_symlink,
    )


if __name__ == "__main__":
    main()
