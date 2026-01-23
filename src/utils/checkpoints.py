"""Checkpoint utilities for saving/loading training state."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(state: Dict[str, Any], out_path: str, create_dirs: bool = True) -> None:
    """Save a checkpoint dictionary to disk.

    Args:
        state: Dictionary containing model/optimizer states and metadata.
        out_path: Full output file path (e.g., /path/to/checkpoint.pt).
        create_dirs: If True, create parent directories if missing.
    """
    if create_dirs:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(state, out_path)


def load_checkpoint(path: str, device: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint dictionary from disk.

    Args:
        path: Path to the checkpoint file.
        device: Optional torch device string to map tensors.

    Returns:
        Loaded checkpoint dictionary.
    """
    map_location = device if device is not None else "cpu"
    return torch.load(path, map_location=map_location)
