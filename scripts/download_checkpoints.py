"""Pretrained-checkpoint downloaders. PRD §8.3.

Pulls YOLOv8m / RTMPose-l Halpe-26 / WASB basketball / PoseConv3D Kinetics
weights onto the ``courtvision-artifacts`` Modal Volume.

Usage::

    uv run modal run scripts/download_checkpoints.py::download_yolov8m
    uv run modal run scripts/download_checkpoints.py::download_all
"""

from __future__ import annotations

import os
from pathlib import Path

from courtvision.modal_app import app, artifacts_volume, image

_CKPT_DIR = "/artifacts/checkpoints"


@app.function(image=image, volumes={"/artifacts": artifacts_volume}, timeout=600, cpu=2)
def download_yolov8m() -> str:
    """Pull yolov8m.pt onto /artifacts/checkpoints/. Idempotent."""
    target = Path(_CKPT_DIR) / "yolov8m.pt"
    if target.exists():
        print(f"yolov8m.pt already at {target} ({target.stat().st_size:,} bytes); skipping")
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    # YOLO auto-downloads "yolov8m.pt" into cwd; chdir so it lands at the
    # final location with no extra copy.
    from ultralytics import YOLO  # type: ignore[attr-defined]

    prev_cwd = os.getcwd()
    os.chdir(target.parent)
    try:
        YOLO("yolov8m.pt")  # side effect: downloads to cwd
    finally:
        os.chdir(prev_cwd)

    assert target.exists(), f"download failed: {target} missing"
    artifacts_volume.commit()
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")
    return str(target)


@app.function(image=image, volumes={"/artifacts": artifacts_volume}, timeout=1200, cpu=2)
def download_rtmpose_l_halpe26() -> str:
    # MMPose model zoo: rtmpose-l_8xb256-420e_coco-256x192 adapted for Halpe-26.
    raise NotImplementedError("Phase A user-test step")


@app.function(image=image, volumes={"/artifacts": artifacts_volume}, timeout=1200, cpu=2)
def download_wasb_basketball_ckpt() -> str:
    # WASB basketball-pretrained: github.com/nttcom/WASB-SBDT release.
    raise NotImplementedError("Phase A user-test step")


@app.function(image=image, volumes={"/artifacts": artifacts_volume}, timeout=1200, cpu=2)
def download_poseconv3d_k400() -> str:
    # MMAction2 model zoo: slowonly_r50_8xb16-u48-240e_k400-keypoint.
    raise NotImplementedError("Phase A user-test step")


@app.local_entrypoint()
def download_all() -> None:
    """Convenience: download all four checkpoints in sequence."""
    download_yolov8m.remote()
    download_rtmpose_l_halpe26.remote()
    download_wasb_basketball_ckpt.remote()
    download_poseconv3d_k400.remote()
    print(f"all checkpoints written under {_CKPT_DIR}")
