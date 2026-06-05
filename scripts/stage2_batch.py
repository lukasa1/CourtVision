"""Batch Stage 2 over several TrackID3x3 Indoor clips for qualitative review.

For each clip: Stage 1 (calibrated YOLO+ByteTrack+postprocess) -> Stage 2
(RTMPose-l Halpe-26 via rtmlib) -> Halpe-26 skeleton overlay with highlighted
wrists. Clips run in parallel across Modal containers (one `.spawn` wave per
stage). Pose JSONs land in `stage2_batch/` so a single `stage2_eval` run gives
an aggregate wrist PCK; overlays land in `stage2_batch_overlays/`.

Usage::

    uv run modal run scripts/stage2_batch.py::run --clips \\
      basket_S4T6_post,basket_S4T7_post,basket_S5T4_post,...
"""

from __future__ import annotations

import json

from courtvision.logging import get_logger
from courtvision.modal_app import app, make_stage2_overlay, run_rtmpose, run_yolo_bytetrack

_log = get_logger(__name__)

# Calibrated Stage 1 config (mirrors scripts/stage1_visualize.py).
_S1_MODEL = {
    "name": "yolov8m",
    "weights": "/artifacts/checkpoints/yolov8m.pt",
    "classes": [0],
    "conf": 0.3,
    "iou": 0.5,
    "imgsz": 1280,
}
_S1_TRACKER = {
    "name": "bytetrack",
    "cfg": "bytetrack.yaml",
    "overrides": {"new_track_thresh": 0.8, "fuse_score": False},
}
_S1_POST = {
    "stitch": {"max_gap_frames": 15, "base_radius_px": 60},
    "filter_non_players": {"min_players": 6, "speed_floor_frac": 0.00068},
}
# Stage 2 rtmlib config (mirrors scripts/stage2_visualize.py).
_POSE_MODEL = {
    "name": "rtmpose-l",
    "onnx_model": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip"
    ),
    "keypoint_format": "halpe26",
    "num_keypoints": 26,
    "input_size": [192, 256],
    "backend": "onnxruntime",
    "device": "cpu",
}


def _vrel(clip: str) -> str:
    return f"trackid3x3/videos/Indoor/raw/{clip}.mp4"


def _session_camera(clip: str) -> tuple[str, str]:
    session, camera = clip.rsplit("_", 1)
    return session, camera


@app.local_entrypoint()
def run(clips: str, max_seconds: float = 30.0) -> None:
    """Process a comma-separated list of clip stems (e.g. ``basket_S4T6_post``)."""
    clip_list = [c.strip() for c in clips.split(",") if c.strip()]
    _log.info("batch over %d clips: %s", len(clip_list), clip_list)

    # Wave 1 — Stage 1 tracks (parallel).
    s1 = {}
    for clip in clip_list:
        session, camera = _session_camera(clip)
        cfg = {
            "session_id": session,
            "camera_id": camera,
            "model": _S1_MODEL,
            "tracker": _S1_TRACKER,
            "postprocess": _S1_POST,
            "output_relpath": f"stage2_batch/tracks_{clip}.json",
        }
        s1[clip] = run_yolo_bytetrack.spawn(_vrel(clip), cfg)
    tracks = {clip: s1[clip].get() for clip in clip_list}

    # Wave 2 — Stage 2 poses (parallel).
    s2 = {}
    for clip in clip_list:
        pose_cfg = {"model": _POSE_MODEL, "output_relpath": f"stage2_batch/pose2d_{clip}.json"}
        s2[clip] = run_rtmpose.spawn(_vrel(clip), tracks[clip], pose_cfg)
    poses = {clip: s2[clip].get() for clip in clip_list}

    # Wave 3 — overlays (parallel).
    ov = {}
    for clip in clip_list:
        ov[clip] = make_stage2_overlay.spawn(
            _vrel(clip), poses[clip], max_seconds, f"stage2_batch_overlays/{clip}.mp4"
        )
    overlays = {clip: ov[clip].get() for clip in clip_list}

    print(
        json.dumps(
            {
                "n_clips": len(clip_list),
                "poses_dir": "stage2_batch",
                "overlays_dir": "stage2_batch_overlays",
                "overlays": overlays,
                "pull_overlays": (
                    "uv run modal volume get courtvision-artifacts "
                    "stage2_batch_overlays outputs/stage2_gate/"
                ),
            }
        )
    )
