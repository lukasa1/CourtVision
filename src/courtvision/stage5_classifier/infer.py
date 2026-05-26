"""Sliding-window inference for Stage 5. PRD §9.5.3, §E5.

32-frame windows @ 60 fps, stride 4, ball-ascent gate, late fusion across
cameras at probability level, NMS on peak P(shoot)+P(layup). Returns
``ShotEventSequence``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from courtvision.schemas import ShotEventSequence


def run_sliding_window_inference(
    pose_dirs: list[Path],
    ball_dirs: list[Path],
    ckpt_path: Path,
    config: dict[str, Any],
) -> ShotEventSequence:
    raise NotImplementedError("Phase E5")


__all__ = ["run_sliding_window_inference"]
