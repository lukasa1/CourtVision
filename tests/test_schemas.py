"""Round-trip every Pydantic schema. PRD §7 contract integrity check."""

from __future__ import annotations

from pathlib import Path

from courtvision.io_utils import read_json_as, write_json
from courtvision.schemas import (
    Ball2DSequence,
    Ball3DSequence,
    BallDetection2D,
    BallDetection3D,
    ContestMeasurement,
    ContestMeasurementSequence,
    Keypoint2D,
    Keypoint3D,
    Pose2D,
    Pose2DSequence,
    Pose3D,
    Pose3DSequence,
    ReleaseFrame,
    ReleaseFrameSequence,
    ShotEvent,
    ShotEventSequence,
    Track,
    TrackSequence,
)


def _roundtrip(model_cls: type, instance: object, tmp_path: Path) -> None:
    out = tmp_path / f"{model_cls.__name__}.json"
    write_json(instance, out)  # type: ignore[arg-type]
    loaded = read_json_as(model_cls, out)
    assert loaded == instance


def test_track_sequence(tmp_path: Path) -> None:
    seq = TrackSequence(
        session_id="s1",
        camera_id="cam0",
        fps=60.0,
        n_frames=10,
        tracks=[Track(track_id=0, bbox=[0.0, 0.0, 10.0, 20.0], confidence=0.9, frame_idx=0)],
    )
    _roundtrip(TrackSequence, seq, tmp_path)


def test_pose2d_sequence(tmp_path: Path) -> None:
    kps = [Keypoint2D(x=float(i), y=float(i), score=0.9) for i in range(26)]
    seq = Pose2DSequence(
        session_id="s1",
        camera_id="cam0",
        fps=60.0,
        poses=[Pose2D(frame_idx=0, track_id=0, keypoints=kps, keypoint_format="halpe26")],
    )
    _roundtrip(Pose2DSequence, seq, tmp_path)


def test_ball2d_sequence(tmp_path: Path) -> None:
    seq = Ball2DSequence(
        session_id="s1",
        camera_id="cam0",
        fps=60.0,
        detections=[BallDetection2D(frame_idx=0, x=100.0, y=200.0, score=0.8)],
    )
    _roundtrip(Ball2DSequence, seq, tmp_path)


def test_pose3d_sequence(tmp_path: Path) -> None:
    kps = [Keypoint3D(x=0.1, y=0.2, z=0.3, confidence=0.9, n_cameras=3) for _ in range(26)]
    seq = Pose3DSequence(
        session_id="s1",
        fps=60.0,
        poses=[Pose3D(frame_idx=0, track_id=0, keypoints=kps, reprojection_error_px=2.5)],
        calibration_ref="/artifacts/calib/s1.json",
    )
    _roundtrip(Pose3DSequence, seq, tmp_path)


def test_ball3d_sequence(tmp_path: Path) -> None:
    seq = Ball3DSequence(
        session_id="s1",
        fps=60.0,
        detections=[
            BallDetection3D(
                frame_idx=0, x=1.0, y=2.0, z=3.0, n_cameras=3, reprojection_error_px=1.5
            )
        ],
        calibration_ref="/artifacts/calib/s1.json",
    )
    _roundtrip(Ball3DSequence, seq, tmp_path)


def test_shot_event_sequence(tmp_path: Path) -> None:
    seq = ShotEventSequence(
        session_id="s1",
        events=[
            ShotEvent(
                event_id="e1",
                shooter_track_id=0,
                window_start_frame=0,
                window_end_frame=31,
                peak_frame=15,
                p_shoot=0.7,
                p_layup=0.1,
                p_dribble=0.1,
                p_background=0.1,
                per_camera_probs={
                    "cam0": {"shoot": 0.7, "layup": 0.1, "dribble": 0.1, "background": 0.1}
                },
            )
        ],
    )
    _roundtrip(ShotEventSequence, seq, tmp_path)


def test_release_frame_sequence(tmp_path: Path) -> None:
    seq = ReleaseFrameSequence(
        session_id="s1",
        releases=[
            ReleaseFrame(
                event_id="e1",
                release_frame_idx=18,
                method="forward_rule",
                confidence_flag="high",
                per_keypoint_confidence={"right_wrist": 0.95, "right_elbow": 0.90},
                n_cameras_at_release=3,
            )
        ],
    )
    _roundtrip(ReleaseFrameSequence, seq, tmp_path)


def test_contest_measurement_sequence(tmp_path: Path) -> None:
    seq = ContestMeasurementSequence(
        session_id="s1",
        measurements=[
            ContestMeasurement(
                event_id="e1",
                wrist_to_wrist_distance_m=0.42,
                wrist_height_advantage_m=-0.05,
                shooter_fade_mps=0.32,
                floor_distance_m=1.1,
                composite_score=0.78,
                closest_defender_track_id=1,
                flags=[],
            )
        ],
    )
    _roundtrip(ContestMeasurementSequence, seq, tmp_path)


def test_extra_fields_forbidden() -> None:
    """Schema drift guard: unknown keys must raise on validation."""
    import pytest
    from pydantic import ValidationError

    bad = {
        "session_id": "s1",
        "camera_id": "cam0",
        "fps": 60.0,
        "n_frames": 10,
        "tracks": [],
        "unexpected_field": "should fail",
    }
    with pytest.raises(ValidationError):
        TrackSequence.model_validate(bad)
