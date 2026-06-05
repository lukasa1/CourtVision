"""Stage 6 — release-detection driver (2D single-camera, no ball).

Runs both pose rules on one shooter's COCO-17 track and returns both
``ReleaseFrame`` results plus the wrist/elbow traces, so the overlay can show
``pose_apex`` vs ``pose_forward`` side-by-side for qualitative comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from courtvision.schemas import ReleaseFrame

from .geometry import _SIDE, dominant_wrist_side, elbow_angle_series, vertical_velocity
from .rule_backward import detect_release_apex
from .rule_forward import detect_release_forward, first_sustained_interval, forward_active_mask


def is_verified_shot(release: ReleaseFrame) -> bool:
    """Cross-method gate (flags.md F24): a Stage-5-detected shot event is
    *verified* if Stage 6's forward rule found a sustained overhead-extension
    interval — i.e. ``release.confidence_flag == "high"``. When the geometric
    rule cannot confirm the extension, Stage 6 falls back to the wrist apex and
    flags low confidence; that pose was probably shot-LIKE without the actual
    release geometry, so the Stage 5 fire was likely a false positive.

    The signal works because Stage 5 (CNN on pose heatmaps) and Stage 6
    (explicit shot geometry on the same pose) are **same-input, different-
    method**; disagreement is meaningful. Empirical small-sample test (n=9):
    Stage 5 @ 0.6 + this gate → 4 TP / 1 FP / 0 FN (vs. Stage 5 @ 0.75 alone
    → 3 TP / 1 FP / 1 FN). Re-validate on iPhone data when collected.
    """
    return release.confidence_flag == "high"


def analyze_release(
    keypoint: Any,  # [T, 17, 2]
    keypoint_score: Any,  # [T, 17]
    event_id: str,
    *,
    window: tuple[int, int] | None = None,
    score_thresh: float = 0.2,
    elbow_extend_deg: float = 160.0,
    release_offset: int = -1,
    min_run: int = 2,
) -> dict[str, Any]:
    """Run forward (primary) + apex (secondary) and return both releases, the
    forward-active interval, and the dominant-wrist traces (for visualisation)."""
    kp = np.asarray(keypoint, dtype=np.float64)
    ks = np.asarray(keypoint_score, dtype=np.float64)
    total = kp.shape[0]
    lo, hi = (0, total) if window is None else (max(0, window[0]), min(total, window[1]))
    side = dominant_wrist_side(kp, ks, lo, hi)
    wr = _SIDE[side][2]
    mask = forward_active_mask(kp, ks, side, score_thresh=score_thresh, elbow_extend_deg=elbow_extend_deg)
    return {
        "forward": detect_release_forward(
            kp, ks, event_id, window=window, score_thresh=score_thresh,
            elbow_extend_deg=elbow_extend_deg, release_offset=release_offset, min_run=min_run,
        ),
        "apex": detect_release_apex(kp, ks, event_id, window=window, score_thresh=score_thresh),
        "side": side,
        "window": (lo, hi),
        "forward_interval": first_sustained_interval(mask, lo, hi, min_run),
        "wrist_y": kp[:, wr, 1].tolist(),
        "wrist_score": ks[:, wr].tolist(),
        "wrist_vy": vertical_velocity(kp[:, wr, 1]).tolist(),
        "elbow_angle": elbow_angle_series(kp, side).tolist(),
    }


__all__ = ["analyze_release"]
