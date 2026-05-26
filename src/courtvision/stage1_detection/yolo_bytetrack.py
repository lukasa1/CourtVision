"""Stage 1 — YOLOv8m + ByteTrack inference. PRD §9.1.

Phase B implements this. Phase A defines the public entrypoint signature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from courtvision.schemas import TrackSequence


def run_stage1(video_path: Path, config: dict[str, Any]) -> TrackSequence:
    """Per-camera player tracking. Returns a ``TrackSequence``.

    Locked config (PRD §9.1): YOLOv8m, ``classes=[0]`` (person), ``conf=0.4``,
    ``iou=0.5``. Tracker: ByteTrack via ultralytics' built-in
    ``tracker='bytetrack.yaml'``.
    """
    raise NotImplementedError("Phase B")


__all__ = ["run_stage1"]
