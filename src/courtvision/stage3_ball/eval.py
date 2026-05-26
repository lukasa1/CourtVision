"""Stage 3 eval — mAP on U Arizona; release-window recall on WASB basketball.

PRD §9.3.
"""

from __future__ import annotations

from pathlib import Path


def evaluate_u_arizona(preds_dir: Path, gt_dir: Path) -> dict:
    """mAP@0.5 and mAP@0.5:0.95 on the U Arizona Roboflow slice."""
    raise NotImplementedError("Phase D")


def evaluate_release_window_recall(preds_dir: Path, gt_dir: Path) -> dict:
    """Recall in ±15-frame windows around hand-picked release moments."""
    raise NotImplementedError("Phase D")


__all__ = ["evaluate_u_arizona", "evaluate_release_window_recall"]
