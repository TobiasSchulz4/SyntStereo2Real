"""Logging utilities for training and evaluation."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

__all__ = ["TensorboardLogger"]


class TensorboardLogger:
    """Thin wrapper around SummaryWriter with safe directory creation."""

    def __init__(self, log_dir: str, flush_secs: int = 30) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

    def add_scalars(self, tag: str, scalars: Dict[str, Any], step: int) -> None:
        for k, v in scalars.items():
            self.writer.add_scalar(f"{tag}/{k}", v, step)

    def add_images(self, tag: str, images, step: int, max_images: int = 4) -> None:
        # images: tensor BCHW or list/tuple of tensors
        self.writer.add_images(tag, images[:max_images], global_step=step)

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def add_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        self.writer.add_text(tag, text, global_step=step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
