"""Stage 4 calibration helpers — chessboard intrinsics + court-landmark extrinsics.

STUB — implementation in weeks 3-4.
"""

from __future__ import annotations

from pathlib import Path


def compute_intrinsics(chessboard_clip: Path) -> dict:
    """One-time per-phone chessboard intrinsics."""
    raise NotImplementedError("Stage 4 — weeks 3-4")


def compute_extrinsics(court_landmarks: Path, intrinsics: dict) -> dict:
    """Per-session court-landmark extrinsics."""
    raise NotImplementedError("Stage 4 — weeks 3-4")


__all__ = ["compute_intrinsics", "compute_extrinsics"]
