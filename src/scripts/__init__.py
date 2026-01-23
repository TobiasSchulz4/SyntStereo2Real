"""Utility scripts package for SyntStereo2Real.

This module exposes small CLI-style helpers (e.g., quick_inference) for
interactive testing and lightweight inference runs.
"""

from .quick_inference import main as quick_inference

__all__ = ["quick_inference"]
