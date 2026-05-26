"""Halpe-26 → COCO-17 keypoint subsetting for Stage 5 classifier consumption.

The geometric pipeline retains the full Halpe-26 (extra wrist/foot keypoints
are useful for the contest metric). The classifier consumes COCO-17 because
PoseConv3D's Kinetics checkpoint was trained on COCO-17.
"""

from __future__ import annotations

from courtvision.schemas import Pose2D

# Halpe-26 index → COCO-17 index. Halpe-26 is a superset of COCO-17 with
# extra head/foot keypoints. Map: the first 17 indices of Halpe-26 align
# with COCO-17 by convention (nose, eyes, ears, shoulders, elbows, wrists,
# hips, knees, ankles).
# Reference: https://github.com/Fang-Haoshu/Halpe-FullBody/blob/master/docs/keypoints.md
HALPE26_TO_COCO17 = list(range(17))


def halpe26_to_coco17(pose: Pose2D) -> Pose2D:
    """Return a new ``Pose2D`` containing only the 17 COCO keypoints."""
    if pose.keypoint_format != "halpe26":
        raise ValueError(f"expected keypoint_format='halpe26', got {pose.keypoint_format!r}")
    if len(pose.keypoints) != 26:
        raise ValueError(f"expected 26 keypoints, got {len(pose.keypoints)}")
    return Pose2D(
        frame_idx=pose.frame_idx,
        track_id=pose.track_id,
        keypoints=[pose.keypoints[i] for i in HALPE26_TO_COCO17],
        keypoint_format="coco17",
    )


__all__ = ["HALPE26_TO_COCO17", "halpe26_to_coco17"]
