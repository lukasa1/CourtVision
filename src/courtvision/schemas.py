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
    method: Literal["forward_rule", "backward_trace", "fallback_apex"]
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
    wrist_to_trajectory_distance_m: float | None  # headline novel metric
    wrist_height_advantage_m: float | None
    shooter_velocity_forward_mps: float
    shooter_velocity_lateral_mps: float
    shooter_velocity_vertical_mps: float
    floor_distance_m: float | None
    composite_score: float | None
    closest_defender_track_id: int | None
    flags: list[str]  # e.g. "uncontested", "id_swap", "weak_triangulation"


class ContestMeasurementSequence(BaseModel):
    model_config = _StrictModel

    session_id: str
    measurements: list[ContestMeasurement]


__all__ = [
    "Track",
    "TrackSequence",
    "Keypoint2D",
    "Pose2D",
    "Pose2DSequence",
    "BallDetection2D",
    "Ball2DSequence",
    "Keypoint3D",
    "Pose3D",
    "Pose3DSequence",
    "BallDetection3D",
    "Ball3DSequence",
    "ShotEvent",
    "ShotEventSequence",
    "ReleaseFrame",
    "ReleaseFrameSequence",
    "ContestMeasurement",
    "ContestMeasurementSequence",
]
