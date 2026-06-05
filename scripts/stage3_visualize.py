"""Stage 3 visualization pipeline.

Runs WASB basketball detection on a clip, then renders an overlay video with
a per-frame ball marker + trajectory trail. Use this for the Phase D
qualitative user-test step (PRD §10).

Usage::

    uv run modal run scripts/stage3_visualize.py::run \\
        --video-relpath trackid3x3/videos/Indoor/raw/basket_S1T1_pre.mp4 \\
        --session-id basket_S1T1 --camera-id pre --max-seconds 30
"""

from __future__ import annotations

import json

from courtvision.logging import get_logger
from courtvision.modal_app import (
    app,
    diagnose_wasb,
    make_stage3_overlay,
    run_wasb,
    run_yolo_ball,
)

_log = get_logger(__name__)

_DEFAULT_MODEL = {
    "name": "wasb-basketball",
    "weights": "/artifacts/checkpoints/wasb_basketball.pth",
    "conf_threshold": 0.5,
}


@app.local_entrypoint()
def run(
    video_relpath: str,
    session_id: str,
    camera_id: str,
    max_seconds: float = 30.0,
    conf_threshold: float = _DEFAULT_MODEL["conf_threshold"],  # type: ignore[assignment]
    max_disp: int = 300,
) -> None:
    """Inference → overlay pipeline. Prints the artifacts-relpath of the overlay."""
    config: dict = {
        "session_id": session_id,
        "camera_id": camera_id,
        "model": {**_DEFAULT_MODEL, "conf_threshold": conf_threshold},
        "score_threshold": conf_threshold,
        "max_disp": max_disp,
        "max_seconds": max_seconds,
    }
    ball_relpath = run_wasb.remote(video_relpath, config)
    _log.info("ball detections written to /artifacts/%s", ball_relpath)

    overlay_relpath = make_stage3_overlay.remote(video_relpath, ball_relpath, max_seconds)
    _log.info("overlay written to /artifacts/%s", overlay_relpath)

    print(
        json.dumps(
            {
                "ball_relpath": ball_relpath,
                "overlay_relpath": overlay_relpath,
                "pull_command": f"uv run modal volume get courtvision-artifacts {overlay_relpath} outputs/",
            }
        )
    )


@app.local_entrypoint()
def diagnose(
    video_relpath: str,
    session_id: str,
    camera_id: str,
    max_seconds: float = 30.0,
) -> None:
    """Heatmap diagnostic: per-frame raw heatmap-max distribution + overlay video.

    Prints the threshold-sweep stats (how many frames have a heatmap peak above
    each threshold) and the relpath of the heatmap-overlay video to pull.
    """
    config: dict = {
        "session_id": session_id,
        "camera_id": camera_id,
        "max_seconds": max_seconds,
    }
    stats = diagnose_wasb.remote(video_relpath, config)
    print(json.dumps(stats, indent=2))
    print(
        f"pull heatmap video: uv run modal volume get courtvision-artifacts "
        f"{stats['heatmap_relpath']} outputs/stage3_diag/"
    )


@app.local_entrypoint()
def run_yolo(
    video_relpath: str,
    session_id: str,
    camera_id: str,
    weights: str = "/artifacts/checkpoints/yolov8m.pt",
    classes: str = "32",  # COCO "sports ball"; use "0" for the custom single-class weight
    conf: float = 0.25,
    imgsz: int = 1280,
    max_disp: float = 0.0,  # 0 disables the motion gate
    tiles: str = "",  # e.g. "2x2" or "3x2" (cols x rows) enables tiled inference
    tile_overlap: float = 0.2,
    max_seconds: float = 30.0,
) -> None:
    """YOLOv8 ball-detection → overlay pipeline (flags.md F19).

    Prints the ball-JSON + overlay relpaths and the pull command. Output names
    use ``camera_id`` so different weights/configs don't overwrite each other.
    """
    config: dict = {
        "session_id": session_id,
        "camera_id": camera_id,
        "model": {
            "weights": weights,
            "classes": [int(c) for c in classes.split(",")],
            "conf": conf,
            "imgsz": imgsz,
        },
        "max_seconds": max_seconds,
    }
    if max_disp > 0:
        config["max_disp"] = max_disp
    if tiles:
        cols, rows = (int(x) for x in tiles.lower().split("x"))
        config["tile"] = {"rows": rows, "cols": cols, "overlap": tile_overlap}
    ball_relpath = run_yolo_ball.remote(video_relpath, config)
    _log.info("ball detections written to /artifacts/%s", ball_relpath)

    overlay_relpath = make_stage3_overlay.remote(video_relpath, ball_relpath, max_seconds)
    _log.info("overlay written to /artifacts/%s", overlay_relpath)

    print(
        json.dumps(
            {
                "ball_relpath": ball_relpath,
                "overlay_relpath": overlay_relpath,
                "pull_command": f"uv run modal volume get courtvision-artifacts {overlay_relpath} outputs/stage3_overlays/",
            }
        )
    )
