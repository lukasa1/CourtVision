"""Stage 6 — ``pose_forward`` release rule (PRIMARY, 2D single-camera, no ball).

The 2D-pose subset of the PRD's forward rule (drops the two ball conditions):
the shooting arm being extended overhead — wrist above the shoulder and elbow
near-straight — is true over a contiguous *interval* (onset of extension →
through release → follow-through hold). The release sits near the **onset** of
that interval (qualitatively ~0-2 frames in), so we return the onset of the
first sustained interval + a tunable ``release_offset``. This beats the wrist
apex, which is systematically late (the wrist keeps rising on follow-through).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from courtvision.schemas import ReleaseFrame

from .geometry import _SIDE, dominant_wrist_side, elbow_angle_series


def forward_active_mask(
    kp: Any, ks: Any, side: str, *, score_thresh: float, elbow_extend_deg: float
) -> Any:
    """Per-frame bool: shooting arm extended overhead (wrist above shoulder +
    elbow ≥ threshold) with a confident wrist."""
    sh, _el, wr = _SIDE[side]
    above = kp[:, wr, 1] < kp[:, sh, 1]  # smaller y = higher
    extended = elbow_angle_series(kp, side) >= elbow_extend_deg
    conf = ks[:, wr] >= score_thresh
    return above & extended & conf


def first_sustained_interval(
    mask: Any, lo: int, hi: int, min_run: int, gap_tol: int = 2
) -> tuple[int, int] | None:
    """First forward-active interval in ``mask[lo:hi]``, returned as
    ``[onset, end)``. Bridges gaps of ≤ ``gap_tol`` frames (so a 1-frame
    tracking flicker doesn't split a real extension and push the onset late),
    and requires ≥ ``min_run`` active frames total (so isolated blips are
    rejected). The onset is the first True frame of that interval."""
    start: int | None = None
    last_true: int | None = None
    active = 0
    for i in range(lo, hi):
        if mask[i]:
            if start is None:
                start, active = i, 0
            last_true, active = i, active + 1
        elif start is not None and last_true is not None and i - last_true > gap_tol:
            if active >= min_run:
                return (start, last_true + 1)
            start, last_true, active = None, None, 0
    if start is not None and last_true is not None and active >= min_run:
        return (start, last_true + 1)
    return None


def detect_release_forward(
    keypoint: Any,  # [T, 17, 2] pixel coords
    keypoint_score: Any,  # [T, 17]
    event_id: str,
    *,
    window: tuple[int, int] | None = None,
    score_thresh: float = 0.2,
    elbow_extend_deg: float = 160.0,
    release_offset: int = -1,
    min_run: int = 2,
) -> ReleaseFrame:
    """Release = onset of the first sustained forward-active interval, plus
    ``release_offset`` (clamped to the **search window** so the offset can be
    negative — the ball typically leaves the hand the frame before our
    "sustained extension" condition first registers; user-calibrated default
    is -1). Falls back to the wrist apex (low confidence) if no sustained
    interval exists."""
    kp = np.asarray(keypoint, dtype=np.float64)
    ks = np.asarray(keypoint_score, dtype=np.float64)
    total = kp.shape[0]
    lo, hi = (0, total) if window is None else (max(0, window[0]), min(total, window[1]))
    side = dominant_wrist_side(kp, ks, lo, hi, score_thresh)
    sh, el, wr = _SIDE[side]
    elbow = elbow_angle_series(kp, side)
    mask = forward_active_mask(kp, ks, side, score_thresh=score_thresh, elbow_extend_deg=elbow_extend_deg)
    interval = first_sustained_interval(mask, lo, hi, min_run)

    flag: Literal["high", "low"]
    if interval is not None:
        rel = max(lo, min(interval[0] + release_offset, hi - 1))
        flag = "high" if ks[rel, wr] >= 0.3 else "low"
    else:
        conf_idx = [i for i in range(lo, hi) if ks[i, wr] >= score_thresh]
        rel = min(conf_idx, key=lambda i: kp[i, wr, 1]) if conf_idx else (lo + hi) // 2
        flag = "low"

    return ReleaseFrame(
        event_id=event_id,
        release_frame_idx=int(rel),
        method="pose_forward",
        confidence_flag=flag,
        per_keypoint_confidence={
            "wrist": float(ks[rel, wr]),
            "elbow": float(ks[rel, el]),
            "shoulder": float(ks[rel, sh]),
            "elbow_angle_deg": float(elbow[rel]),
        },
        n_cameras_at_release=1,
    )


__all__ = ["detect_release_forward", "first_sustained_interval", "forward_active_mask"]
