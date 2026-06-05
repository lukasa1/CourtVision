"""Stage 2 visualization pipeline — RTMPose inference + skeleton overlay.

Runs RTMPose-l on an existing Stage 1 TrackSequence, then renders a Halpe-26
skeleton overlay (wrists highlighted) for the Phase C qualitative user-test
step (PRD §10).

Usage::

    uv run modal run scripts/stage2_visualize.py::run \\
        --video-relpath trackid3x3/videos/Indoor/raw/basket_S4T4_post.mp4 \\
        --tracks-relpath stage1/basket_S4T4/tracks_basket_S4T4_post.json \\
        --max-seconds 30
"""

from __future__ import annotations

import json

from courtvision.logging import get_logger
from courtvision.modal_app import app, make_stage2_overlay, run_rtmpose

_log = get_logger(__name__)

_DEFAULT_MODEL = {
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


@app.local_entrypoint()
def run(
    video_relpath: str,
    tracks_relpath: str,
    max_seconds: float = 30.0,
) -> None:
    """Inference → skeleton overlay. Prints the artifacts-relpaths + pull command."""
    config = {"model": dict(_DEFAULT_MODEL)}
    poses_relpath = run_rtmpose.remote(video_relpath, tracks_relpath, config)
    _log.info("poses written to /artifacts/%s", poses_relpath)

    overlay_relpath = make_stage2_overlay.remote(video_relpath, poses_relpath, max_seconds)
    _log.info("overlay written to /artifacts/%s", overlay_relpath)

    print(
        json.dumps(
            {
                "poses_relpath": poses_relpath,
                "overlay_relpath": overlay_relpath,
                "pull_command": (
                    f"uv run modal volume get courtvision-artifacts {overlay_relpath} outputs/"
                ),
            }
        )
    )
