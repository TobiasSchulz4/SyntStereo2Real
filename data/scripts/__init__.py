"""Dataset preparation scripts package.

This module exposes helper functions for preparing datasets into the
project's expected layout. Each script can be used as a CLI or imported
programmatically.
"""

from .prepare_syntcities import prepare as prepare_syntcities
from .prepare_us3d import prepare as prepare_us3d
from .prepare_scene_flow import prepare as prepare_scene_flow
from .prepare_kitti import prepare as prepare_kitti

__all__ = [
    "prepare_syntcities",
    "prepare_us3d",
    "prepare_scene_flow",
    "prepare_kitti",
]
