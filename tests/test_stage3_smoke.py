"""Stage 3 smoke — Phase D.

The full WASB path (``run_stage3``) needs CUDA + the WASB checkout at
``/opt/wasb``, so it only runs on Modal — covered by the user-test step, not
here. Locally we cover what's GPU-free: clean module import (no torch/WASB at
import time), the pre-flight guards, and the ball-overlay renderer end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from courtvision.io_utils import read_json_as, write_json
from courtvision.schemas import Ball2DSequence, BallDetection2D


def test_wasb_inference_imports_without_gpu() -> None:
    """Importing the wrapper must not pull torch/WASB — those are deferred.

    Guards the contract that lets this module import on a laptop with no GPU
    and no WASB checkout (all heavy imports live inside ``run_stage3``).
    """
    from courtvision.stage3_ball import wasb_inference

    assert callable(wasb_inference.run_stage3)
    # torch is imported inside run_stage3, never at module scope, so it must
    # not be bound as a module attribute.
    assert not hasattr(wasb_inference, "torch")


def test_run_stage3_missing_video_raises() -> None:
    """Nonexistent video → FileNotFoundError, before any WASB import."""
    from courtvision.stage3_ball.wasb_inference import run_stage3

    with pytest.raises(FileNotFoundError, match="video not found"):
        run_stage3(
            Path("/nonexistent/clip.mp4"),
            {"session_id": "s", "camera_id": "c"},
        )


def test_make_ball_overlay_video_smoke(synthetic_clip: Path, tmp_path: Path) -> None:
    """Synthetic clip + a few detections → valid MP4 with the same frame count."""
    import cv2

    from courtvision.stage3_ball.overlay import make_ball_overlay_video

    seq = Ball2DSequence(
        session_id="overlay",
        camera_id="cam0",
        fps=30.0,
        detections=[
            BallDetection2D(frame_idx=1, x=40.0, y=40.0, score=0.9),
            BallDetection2D(frame_idx=3, x=55.0, y=42.0, score=0.8),
        ],
    )
    out = tmp_path / "ball_overlay.mp4"
    make_ball_overlay_video(synthetic_clip, seq, out)
    assert out.exists() and out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    try:
        n_out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    assert n_out_frames == 5


def test_make_ball_overlay_video_caps_max_seconds(synthetic_clip: Path, tmp_path: Path) -> None:
    """max_seconds=0.1 on a 30fps 5-frame clip should produce ~3 frames."""
    import cv2

    from courtvision.stage3_ball.overlay import make_ball_overlay_video

    seq = Ball2DSequence(session_id="overlay", camera_id="cam0", fps=30.0, detections=[])
    out = tmp_path / "ball_overlay_capped.mp4"
    make_ball_overlay_video(synthetic_clip, seq, out, max_seconds=0.1)

    cap = cv2.VideoCapture(str(out))
    try:
        n_out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    # 0.1s @ 30fps = 3 frames; allow ±1 for container quirks.
    assert 2 <= n_out_frames <= 4


def test_ball2d_sequence_round_trip_schema(tmp_path: Path) -> None:
    """Ball2DSequence round-trips through write/read."""
    seq = Ball2DSequence(
        session_id="rt",
        camera_id="cam0",
        fps=30.0,
        detections=[BallDetection2D(frame_idx=0, x=1.0, y=2.0, score=0.5)],
    )
    p = tmp_path / "ball2d_rt_cam0.json"
    write_json(seq, p)
    loaded = read_json_as(Ball2DSequence, p)
    assert loaded == seq
