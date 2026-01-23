"""Train AANet on translated data.

This is a lightweight wrapper that prepares command-line arguments and invokes the
third-party AANet training script. It also provides a simple pythonic interface
for programmatic use.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Any, Dict

import yaml


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _build_command(cfg: Dict[str, Any]) -> list:
    """Build the training command for AANet.

    Expected config keys (defaults in config_aanet_training.yaml):
      - aanet_root: path to third_party/aanet
      - train_script: relative or absolute path to training script
      - data_root: path to translated dataset
      - batch_size, epochs, max_disp, lr
      - save_dir: output directory for AANet checkpoints
      - extra_args: dict of additional CLI args to pass through
    """
    aanet_root = cfg.get("aanet_root", "third_party/aanet")
    train_script = cfg.get("train_script", os.path.join(aanet_root, "train.py"))
    if not os.path.isabs(train_script):
        train_script = os.path.join(aanet_root, train_script)

    cmd = [
        "python",
        train_script,
        "--data_root",
        cfg.get("data_root", "datasets/translated"),
        "--batch_size",
        str(cfg.get("batch_size", 20)),
        "--epochs",
        str(cfg.get("epochs", 400)),
        "--max_disp",
        str(cfg.get("max_disp", 192)),
        "--lr",
        str(cfg.get("lr", 1e-3)),
        "--save_dir",
        cfg.get("save_dir", "experiments/checkpoints/aanet"),
    ]

    extra_args = cfg.get("extra_args", {}) or {}
    for key, value in extra_args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def train(config_path: str) -> None:
    cfg = _load_config(config_path)
    save_dir = cfg.get("save_dir", "experiments/checkpoints/aanet")
    _ensure_dir(save_dir)
    cmd = _build_command(cfg)
    print("[train_aanet] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AANet on translated data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_aanet_training.yaml",
        help="Path to AANet training config YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
