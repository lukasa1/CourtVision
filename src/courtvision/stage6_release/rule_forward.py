"""Stage 6 — forward 4-condition release-frame rule.

STUB — weeks 3-4. See README for spec.
"""

from __future__ import annotations

from courtvision.schemas import Pose3DSequence, ReleaseFrame, ShotEvent


def find_release_forward(
    event: ShotEvent, pose3d: Pose3DSequence, ball3d_z: list[float]
) -> ReleaseFrame:
    raise NotImplementedError("weeks 3-4")


__all__ = ["find_release_forward"]
