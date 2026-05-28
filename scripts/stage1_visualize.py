"""Stage 1 visualization pipeline.

Runs YOLOv8m + ByteTrack on a clip, then renders an overlay video with
bboxes + track ids. Use this for the Phase B qualitative user-test
step (PRD §10).

Usage::

    uv run modal run scripts/stage1_visualize.py::run \\
        --video-relpath trackid3x3/videos/Indoor/raw/basket_S1T1_pre.mp4 \\
        --session-id basket_S1T1 --camera-id pre --max-seconds 30
"""

from __future__ import annotations

import json

from courtvision.logging import get_logger
from courtvision.modal_app import app, make_stage1_overlay, run_yolo_bytetrack

_log = get_logger(__name__)

_DEFAULT_MODEL = {
    "name": "yolov8m",
    "weights": "/artifacts/checkpoints/yolov8m.pt",
    "classes": [0],
    "conf": 0.3,
    "iou": 0.5,
    "imgsz": 1280,
}
# Calibrated config from the Phase B sweep + persistence work.
_DEFAULT_TRACKER = {
    "name": "bytetrack",
    "cfg": "bytetrack.yaml",
    "overrides": {"new_track_thresh": 0.8, "fuse_score": False},
}
_DEFAULT_POSTPROCESS = {
    "stitch": {"max_gap_frames": 15, "base_radius_px": 60},
    "filter_non_players": {"min_players": 6, "speed_floor_frac": 0.00068},
}


@app.local_entrypoint()
def run(
    video_relpath: str,
    session_id: str,
    camera_id: str,
    max_seconds: float = 30.0,
    conf: float = _DEFAULT_MODEL["conf"],  # type: ignore[assignment]
    iou: float = _DEFAULT_MODEL["iou"],  # type: ignore[assignment]
    imgsz: int = _DEFAULT_MODEL["imgsz"],  # type: ignore[assignment]
    postprocess: bool = True,
) -> None:
    """Inference → overlay pipeline. Prints the artifacts-relpath of the overlay."""
    config: dict = {
        "session_id": session_id,
        "camera_id": camera_id,
        "model": {**_DEFAULT_MODEL, "conf": conf, "iou": iou, "imgsz": imgsz},
        "tracker": _DEFAULT_TRACKER,
    }
    if postprocess:
        config["postprocess"] = _DEFAULT_POSTPROCESS
    tracks_relpath = run_yolo_bytetrack.remote(video_relpath, config)
    _log.info("tracks written to /artifacts/%s", tracks_relpath)

    overlay_relpath = make_stage1_overlay.remote(
        video_relpath, tracks_relpath, max_seconds
    )
    _log.info("overlay written to /artifacts/%s", overlay_relpath)

    print(
        json.dumps(
            {
                "tracks_relpath": tracks_relpath,
                "overlay_relpath": overlay_relpath,
                "pull_command": f"uv run modal volume get courtvision-artifacts {overlay_relpath} outputs/",
            }
        )
    )
