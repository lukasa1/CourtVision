"""Stage 4 — multi-view triangulation. PRD §9.4.

STUB — implementation in weeks 3-4 (requires own footage + calibration).
"""

from __future__ import annotations

from courtvision.schemas import Ball3DSequence, Pose3DSequence


def triangulate_session(session_id: str) -> Pose3DSequence:
    """Triangulate per-camera 2D pose into 3D world coords via Pose2Sim DLT."""
    raise NotImplementedError("Stage 4 — weeks 3-4")


def triangulate_ball_session(session_id: str) -> Ball3DSequence:
    """Triangulate per-camera 2D ball into 3D via ``cv2.triangulatePoints``."""
    raise NotImplementedError("Stage 4 — weeks 3-4")


__all__ = ["triangulate_ball_session", "triangulate_session"]
