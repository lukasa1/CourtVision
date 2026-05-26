"""Stage 2 — RTMPose-l Halpe-26 top-down inference. PRD §9.2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from courtvision.schemas import Pose2DSequence, TrackSequence


def run_stage2(
    tracks: TrackSequence, video_path: Path, config: dict[str, Any]
) -> Pose2DSequence:
    """Top-down pose inference on Stage 1 crops. Returns ``Pose2DSequence``.

    Locked config: RTMPose-l, Halpe-26 keypoints, MMPose
    ``rtmpose-l_8xb256-420e_coco-256x192`` adapted for Halpe-26.
    """
    raise NotImplementedError("Phase C")


__all__ = ["run_stage2"]
