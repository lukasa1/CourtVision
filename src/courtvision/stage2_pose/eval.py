"""Stage 2 eval — wrist PCK@0.05 on TrackID3x3, COCO-17 comparison on MultiSubjects.

PRD §9.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_trackid3x3(poses_dir: Path, gt_dir: Path) -> dict[str, Any]:
    """Wrist PCK@0.05 on TrackID3x3 (subject to wrist availability — Day 2 check)."""
    raise NotImplementedError("Phase C")


def evaluate_multisubjects(poses_dir: Path, gt_dir: Path) -> dict[str, Any]:
    """COCO-17 subset PCK against MultiSubjects BlazePose-33 → COCO-17 conversion."""
    raise NotImplementedError("Phase C")


__all__ = ["evaluate_multisubjects", "evaluate_trackid3x3"]
