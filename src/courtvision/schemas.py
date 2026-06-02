"""Cross-stage I/O contracts. Definitive source — see PRD §7.

Every stage that emits a JSON artifact validates against one of these models on
write (via ``courtvision.io_utils.write_json``) and on read (via
``read_json_as``). Schemas use ``extra='forbid'`` so unexpected fields fail
loudly rather than drifting silently.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Shared model config: forbid unknown keys so schema drift is caught at parse
# time, not at downstream-consumer time.
_StrictModel = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Stage 1 — per-camera player tracks
# ---------------------------------------------------------------------------


class Track(BaseModel):
    """A single tracked-player bounding box at one frame."""

    model_config = _StrictModel

    track_id: int
    bbox: list[float] = Field(..., min_length=4, max_length=4)  # [x, y, w, h] px
    confidence: float
    frame_idx: int


class TrackSequence(BaseModel):
    """All tracks for one (session, camera) pair."""

    model_config = _StrictModel

    session_id: str
    camera_id: str
    fps: float
    n_frames: int
    tracks: list[Track]


# ---------------------------------------------------------------------------
# Stage 2 — per-camera 2D pose
# ---------------------------------------------------------------------------


class Keypoint2D(BaseModel):
    model_config = _StrictModel

    x: float
    y: float
    score: float


class Pose2D(BaseModel):
    """One pose at one frame for one tracked player."""

    model_config = _StrictModel

    frame_idx: int
    track_id: int
    keypoints: list[Keypoint2D]  # length 26 for Halpe-26, 17 for COCO-17 subset
    keypoint_format: Literal["halpe26", "coco17"] = "halpe26"


class Pose2DSequence(BaseModel):
    model_config = _StrictModel

    session_id: str
    camera_id: str
    fps: float
    poses: list[Pose2D]


# ---------------------------------------------------------------------------
# Stage 3 — per-camera ball detections
# ---------------------------------------------------------------------------


class BallDetection2D(BaseModel):
    model_config = _StrictModel

    frame_idx: int
    x: float
    y: float
    score: float


class Ball2DSequence(BaseModel):
    model_config = _StrictModel

    session_id: str
    camera_id: str
    fps: float
    detections: list[BallDetection2D]


# ---------------------------------------------------------------------------
# Stage 4 — triangulated 3D pose + ball (CONTRACT; impl in weeks 3-4)
# ---------------------------------------------------------------------------


class Keypoint3D(BaseModel):
    model_config = _StrictModel

    x: float
    y: float
    z: float
    confidence: float  # mean of contributing 2D scores
    n_cameras: int  # how many cameras contributed (0/2/3)


class Pose3D(BaseModel):
    model_config = _StrictModel

    frame_idx: int
    track_id: int
    keypoints: list[Keypoint3D]  # Halpe-26 in world coords (metres)
    reprojection_error_px: float


class Pose3DSequence(BaseModel):
    model_config = _StrictModel

    session_id: str
    fps: float
    poses: list[Pose3D]
    calibration_ref: str  # path to extrinsics file used for this session


class BallDetection3D(BaseModel):
    model_config = _StrictModel

    frame_idx: int
    x: float
    y: float
    z: float
    n_cameras: int
    reprojection_error_px: float


class Ball3DSequence(BaseModel):
    model_config = _StrictModel

    session_id: str
    fps: float
    detections: list[BallDetection3D]
    calibration_ref: str


# ---------------------------------------------------------------------------
# Stage 5 — shot events
# ---------------------------------------------------------------------------


class ShotEvent(BaseModel):
    model_config = _StrictModel

    event_id: str
    shooter_track_id: int
    window_start_frame: int
    window_end_frame: int
    peak_frame: int  # frame of max P(shoot)+P(layup)
    p_shoot: float
    p_layup: float
    p_dribble: float
    p_background: float
    per_camera_probs: dict[str, dict[str, float]]  # camera_id -> class -> prob


class ShotEventSequence(BaseModel):
    model_config = _StrictModel

    session_id: str
    events: list[ShotEvent]


# ---------------------------------------------------------------------------
# Stage 6 — release frames
# ---------------------------------------------------------------------------


class ReleaseFrame(BaseModel):
    model_config = _StrictModel

    event_id: str
    release_frame_idx: int
    # pose_apex / pose_forward: single-camera 2D pose-only methods (no ball, no
    # 3D) — the implemented release detectors. The ball-based forward_rule /
    # backward_trace / fallback_apex remain for the eventual 3D+ball variant
    # (n_cameras_at_release==1 marks the 2D single-camera methods). See flags F23.
    method: Literal["forward_rule", "backward_trace", "fallback_apex", "pose_apex", "pose_forward"]
    confidence_flag: Literal["high", "low"]
    per_keypoint_confidence: dict[str, float]
    n_cameras_at_release: int


class ReleaseFrameSequence(BaseModel):
    model_config = _StrictModel

    session_id: str
    releases: list[ReleaseFrame]


# ---------------------------------------------------------------------------
# Stage 7 — contest measurements
# ---------------------------------------------------------------------------


class ContestMeasurement(BaseModel):
    model_config = _StrictModel

    event_id: str
    # Headline pose-only proxy (Phase D-drop redesign — decisions.md Decision 9,
    # flags.md F25). 3D distance from closest defender's nearer wrist to the
    # shooter's shooting wrist at release. Renamed from wrist_to_trajectory_distance_m.
    wrist_to_wrist_distance_m: float | None
    # defender wrist z minus shooter shooting-wrist z (was z vs ball position).
    wrist_height_advantage_m: float | None
    # sqrt(max(0, -v_forward)^2 + v_lateral^2) in the court frame; forward = toward
    # rim. 0 = no fade (pure forward / stationary). Replaces 3 shooter_velocity_*
    # fields; vertical dropped (captured by wrist_height_advantage).
    shooter_fade_mps: float
    floor_distance_m: float | None
    composite_score: float | None
    closest_defender_track_id: int | None
    flags: list[str]  # e.g. "uncontested", "id_swap", "weak_triangulation"


class ContestMeasurementSequence(BaseModel):
    model_config = _StrictModel

    session_id: str
    measurements: list[ContestMeasurement]


__all__ = [
    "Ball2DSequence",
    "Ball3DSequence",
    "BallDetection2D",
    "BallDetection3D",
    "ContestMeasurement",
    "ContestMeasurementSequence",
    "Keypoint2D",
    "Keypoint3D",
    "Pose2D",
    "Pose2DSequence",
    "Pose3D",
    "Pose3DSequence",
    "ReleaseFrame",
    "ReleaseFrameSequence",
    "ShotEvent",
    "ShotEventSequence",
    "Track",
    "TrackSequence",
]
