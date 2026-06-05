"""Stage 6 — pose geometry helpers for 2D single-camera release detection.

Pure NumPy on COCO-17 keypoints (``[T, 17, 2]`` pixel coords, image y points
DOWN, so a higher hand = smaller y). No ball, no 3D — these back the
``pose_apex`` / ``pose_forward`` release methods (PRD §9.6 deviation, flags F23).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# COCO-17 indices.
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
_SIDE = {"L": (L_SHOULDER, L_ELBOW, L_WRIST), "R": (R_SHOULDER, R_ELBOW, R_WRIST)}


def _angle(a: Any, b: Any, c: Any) -> Any:
    """Per-frame angle (degrees) at vertex ``b`` for point triples a-b-c."""
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6
    cos = np.clip((v1 * v2).sum(-1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def elbow_angle_series(kp: Any, side: str) -> Any:
    """Elbow angle per frame (≈180° = fully extended) for ``side`` in {'L','R'}."""
    sh, el, wr = _SIDE[side]
    return _angle(kp[:, sh], kp[:, el], kp[:, wr])


def dominant_wrist_side(kp: Any, ks: Any, lo: int, hi: int, score_thresh: float = 0.2) -> str:
    """The shooting-hand side: the wrist that reaches the highest point (min y)
    over ``[lo, hi)`` among confident frames. Defaults to 'R' if neither is seen."""
    best_y, best_side = None, "R"
    for side in ("L", "R"):
        wr = _SIDE[side][2]
        ys, ss = kp[lo:hi, wr, 1], ks[lo:hi, wr]
        conf = ss >= score_thresh
        if conf.any():
            top = float(ys[conf].min())
            if best_y is None or top < best_y:
                best_y, best_side = top, side
    return best_side


def vertical_velocity(y: Any) -> Any:
    """Upward velocity per frame (central diff; positive = wrist rising, since
    image y points down)."""
    vy = np.zeros_like(y, dtype=np.float64)
    if len(y) >= 3:
        vy[1:-1] = -(y[2:] - y[:-2]) / 2.0
    return vy


__all__ = [
    "L_ELBOW",
    "L_SHOULDER",
    "L_WRIST",
    "R_ELBOW",
    "R_SHOULDER",
    "R_WRIST",
    "dominant_wrist_side",
    "elbow_angle_series",
    "vertical_velocity",
]
