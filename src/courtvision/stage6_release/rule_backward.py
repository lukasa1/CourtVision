"""Stage 6 — backward-trace-from-apex release-frame rule (alternate impl).

STUB — weeks 3-4. Head-to-head against rule_forward per PRD §9.6.
"""

from __future__ import annotations

from courtvision.schemas import Pose3DSequence, ReleaseFrame, ShotEvent


def find_release_backward(
    event: ShotEvent, pose3d: Pose3DSequence, ball3d_z: list[float]
) -> ReleaseFrame:
    raise NotImplementedError("weeks 3-4")


__all__ = ["find_release_backward"]
