"""Stage 7 — contest measurements at release. PRD §9.7, Decision 9.

5 measurements per shot event; ``wrist_to_trajectory_distance`` is the
headline novel metric. STUB — weeks 3-4.
"""

from __future__ import annotations

from courtvision.schemas import (
    Ball3DSequence,
    ContestMeasurement,
    Pose3DSequence,
    ReleaseFrame,
    ShotEvent,
)


def wrist_to_trajectory_distance(
    event: ShotEvent,
    release: ReleaseFrame,
    shooter_pose3d: Pose3DSequence,
    defender_pose3d: Pose3DSequence,
    ball3d: Ball3DSequence,
) -> float:
    """3D defender-wrist distance to the actual ball trajectory at release.

    Headline novel metric — a refinement of NBA SportVU's floor-distance
    convention using 3D hand tracking.
    """
    raise NotImplementedError("weeks 3-4")


def wrist_height_advantage(
    event: ShotEvent,
    release: ReleaseFrame,
    shooter_pose3d: Pose3DSequence,
    defender_pose3d: Pose3DSequence,
) -> float:
    raise NotImplementedError("weeks 3-4")


def shooter_velocity_decomposed(
    event: ShotEvent, release: ReleaseFrame, shooter_pose3d: Pose3DSequence
) -> tuple[float, float, float]:
    """Returns (forward, lateral, vertical) m/s."""
    raise NotImplementedError("weeks 3-4")


def floor_distance(
    event: ShotEvent,
    release: ReleaseFrame,
    shooter_pose3d: Pose3DSequence,
    defender_pose3d: Pose3DSequence,
) -> float:
    raise NotImplementedError("weeks 3-4")


def composite(event_measurements: dict) -> float:
    raise NotImplementedError("weeks 3-4")


def measure_event(
    event: ShotEvent,
    release: ReleaseFrame,
    shooter_pose3d: Pose3DSequence,
    defender_pose3d: Pose3DSequence,
    ball3d: Ball3DSequence,
) -> ContestMeasurement:
    raise NotImplementedError("weeks 3-4")


__all__ = [
    "wrist_to_trajectory_distance",
    "wrist_height_advantage",
    "shooter_velocity_decomposed",
    "floor_distance",
    "composite",
    "measure_event",
]
