"""Stage 6 — ``pose_apex`` release rule (PRIMARY, 2D single-camera, no ball).

The 2D-pose analogue of the PRD's backward-trace-from-apex: the release of a
jump shot ≈ the frame where the shooting wrist reaches its peak height (the
apex of the wrist's vertical trajectory), with the arm near full extension.
The ball-based ``backward_trace`` remains for the future 3D+ball variant.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from courtvision.schemas import ReleaseFrame

from .geometry import _SIDE, dominant_wrist_side, elbow_angle_series


def detect_release_apex(
    keypoint: Any,  # [T, 17, 2] pixel coords
    keypoint_score: Any,  # [T, 17]
    event_id: str,
    *,
    window: tuple[int, int] | None = None,
    score_thresh: float = 0.2,
    elbow_extend_deg: float = 150.0,
) -> ReleaseFrame:
    """Release = the dominant wrist's peak-height frame within ``window``.

    ``window`` is ``[lo, hi)`` in frame indices (default: whole sequence).
    Confidence is ``low`` when the wrist is poorly tracked at the apex or the
    elbow is not near-extended there (a raised-but-bent arm is not a release).
    """
    kp = np.asarray(keypoint, dtype=np.float64)
    ks = np.asarray(keypoint_score, dtype=np.float64)
    total = kp.shape[0]
    lo, hi = (0, total) if window is None else (max(0, window[0]), min(total, window[1]))
    side = dominant_wrist_side(kp, ks, lo, hi, score_thresh)
    sh, el, wr = _SIDE[side]
    elbow = elbow_angle_series(kp, side)

    flag: Literal["high", "low"]
    conf_idx = [i for i in range(lo, hi) if ks[i, wr] >= score_thresh]
    if conf_idx:
        rel = min(conf_idx, key=lambda i: kp[i, wr, 1])  # highest wrist = min y
        flag = "high"
        if ks[rel, wr] < 0.3 or elbow[rel] < elbow_extend_deg:
            flag = "low"
    else:  # no confident wrist in window → centre of window, low confidence
        rel = (lo + hi) // 2
        flag = "low"

    return ReleaseFrame(
        event_id=event_id,
        release_frame_idx=int(rel),
        method="pose_apex",
        confidence_flag=flag,
        per_keypoint_confidence={
            "wrist": float(ks[rel, wr]),
            "elbow": float(ks[rel, el]),
            "shoulder": float(ks[rel, sh]),
            "elbow_angle_deg": float(elbow[rel]),
        },
        n_cameras_at_release=1,
    )


__all__ = ["detect_release_apex"]
