"""PoseConv3D SlowOnly-R50 model factory. PRD §9.5.

Phase E1 implements ``build_poseconv3d``. Phase A defines the signature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_poseconv3d(
    num_classes: int = 4,
    pretrained_path: Path | None = None,
) -> Any:  # nn.Module — typed Any to keep mypy happy without torch import
    """Build a PoseConv3D SlowOnly-R50 with ``num_classes`` outputs.

    If ``pretrained_path`` is given, loads the Kinetics-400 (or later transfer
    stage) checkpoint and swaps the head to ``num_classes``.
    """
    raise NotImplementedError("Phase E1")


__all__ = ["build_poseconv3d"]
