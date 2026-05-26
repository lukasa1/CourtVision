"""Stage 1 eval — MOTA / IDF1 on TrackID3x3. PRD §9.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_trackid3x3(tracks_dir: Path, gt_dir: Path) -> dict[str, Any]:
    """Compute MOTA and IDF1 against TrackID3x3 ground truth."""
    raise NotImplementedError("Phase B")


__all__ = ["evaluate_trackid3x3"]
