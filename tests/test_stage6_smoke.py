"""Stage 6 smoke tests — 2D pose-only release detection (CPU, no torch/ball)."""

from __future__ import annotations

import numpy as np

from courtvision.schemas import ReleaseFrame, ReleaseFrameSequence
from courtvision.stage6_release import analyze_release, detect_release_apex, is_verified_shot


def _synthetic_shot(total: int = 40, apex: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """COCO-17 track: the RIGHT wrist rises to a peak height at frame ``apex``
    (min y), arm kept extended (shoulder-elbow-wrist colinear); LEFT wrist stays
    low so RIGHT is dominant."""
    kp = np.zeros((total, 17, 2), dtype=np.float64)
    ks = np.ones((total, 17), dtype=np.float64)
    rise = np.maximum(0.0, 40.0 - np.abs(np.arange(total) - apex) * 2.0)  # peak at apex
    rwy = 100.0 - rise  # smaller y = higher
    kp[:, 6] = np.column_stack([np.full(total, 100.0), np.full(total, 120.0)])  # R shoulder
    kp[:, 10] = np.column_stack([np.full(total, 100.0), rwy])  # R wrist
    kp[:, 8] = np.column_stack([np.full(total, 100.0), (120.0 + rwy) / 2.0])  # R elbow (colinear)
    kp[:, 5] = np.column_stack([np.full(total, 80.0), np.full(total, 120.0)])  # L shoulder
    kp[:, 9] = np.column_stack([np.full(total, 80.0), np.full(total, 130.0)])  # L wrist (low)
    kp[:, 7] = np.column_stack([np.full(total, 80.0), np.full(total, 125.0)])  # L elbow
    return kp, ks


def test_apex_finds_peak_height() -> None:
    kp, ks = _synthetic_shot(apex=20)
    rf = detect_release_apex(kp, ks, "evt0")
    assert rf.release_frame_idx == 20
    assert rf.method == "pose_apex"
    assert rf.n_cameras_at_release == 1
    assert rf.confidence_flag == "high"


def test_analyze_release_returns_both_and_traces() -> None:
    kp, ks = _synthetic_shot(apex=25)
    out = analyze_release(kp, ks, "evt1")
    assert out["side"] == "R"
    assert out["apex"].release_frame_idx == 25
    assert out["apex"].method == "pose_apex"
    assert out["forward"].method == "pose_forward"
    assert len(out["wrist_y"]) == kp.shape[0]


def test_release_frame_roundtrips() -> None:
    kp, ks = _synthetic_shot()
    rf = detect_release_apex(kp, ks, "evt2")
    seq = ReleaseFrameSequence(session_id="s", releases=[rf])
    again = ReleaseFrameSequence.model_validate(seq.model_dump())
    assert again.releases[0].release_frame_idx == rf.release_frame_idx
    assert isinstance(again.releases[0], ReleaseFrame)


def test_window_restricts_search() -> None:
    kp, ks = _synthetic_shot(total=40, apex=20)
    # restrict to [0,10): true global apex (20) is excluded, so the highest
    # confident wrist within the window is at its end (frame 9).
    rf = detect_release_apex(kp, ks, "evt3", window=(0, 10))
    assert rf.release_frame_idx == 9


def test_is_verified_shot_gates_on_confidence() -> None:
    base = dict(
        event_id="evt",
        release_frame_idx=0,
        method="pose_forward",
        per_keypoint_confidence={"wrist": 0.9},
        n_cameras_at_release=1,
    )
    assert is_verified_shot(ReleaseFrame(**base, confidence_flag="high")) is True
    assert is_verified_shot(ReleaseFrame(**base, confidence_flag="low")) is False
