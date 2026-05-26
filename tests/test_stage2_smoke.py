"""Stage 2 smoke — Phase A: stub raises + Halpe-26 → COCO-17 conversion works."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtvision.schemas import Keypoint2D, Pose2D, TrackSequence
from courtvision.stage2_pose.halpe_to_coco17 import HALPE26_TO_COCO17, halpe26_to_coco17
from courtvision.stage2_pose.rtmpose_inference import run_stage2


def test_stage2_stub_raises() -> None:
    empty_tracks = TrackSequence(session_id="s", camera_id="c", fps=60, n_frames=0, tracks=[])
    with pytest.raises(NotImplementedError, match="Phase C"):
        run_stage2(empty_tracks, Path("/dev/null"), {})


def test_halpe26_to_coco17() -> None:
    kps = [Keypoint2D(x=float(i), y=float(i), score=0.9) for i in range(26)]
    pose = Pose2D(frame_idx=0, track_id=0, keypoints=kps, keypoint_format="halpe26")
    converted = halpe26_to_coco17(pose)
    assert converted.keypoint_format == "coco17"
    assert len(converted.keypoints) == 17
    # Identity mapping on the first 17 indices.
    for new_i, src_i in enumerate(HALPE26_TO_COCO17):
        assert converted.keypoints[new_i] == kps[src_i]


def test_halpe26_to_coco17_rejects_wrong_format() -> None:
    kps = [Keypoint2D(x=0.0, y=0.0, score=0.0) for _ in range(17)]
    bad_pose = Pose2D(frame_idx=0, track_id=0, keypoints=kps, keypoint_format="coco17")
    with pytest.raises(ValueError, match="halpe26"):
        halpe26_to_coco17(bad_pose)
