"""Two-stream PoseConv3D + Ball MLP, head-level fusion. PRD §9.5.1."""

from __future__ import annotations

from typing import Any


def build_two_stream(
    num_classes: int = 4,
    pose_pretrained_path: Any = None,
    ball_enabled: bool = False,
) -> Any:
    """Combined two-stream classifier.

    When ``ball_enabled`` is False (Phase E1-E4, pre-week-3-4), the ball
    stream returns zeros and the head consumes only pose features.
    """
    raise NotImplementedError("Phase E1")


__all__ = ["build_two_stream"]
