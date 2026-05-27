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


def _https_pull(url: str, target: Path) -> str:
    """Download `url` to `target` if not already present. Returns the path."""
    import urllib.request

    if target.exists():
        print(f"{target.name} already at {target} ({target.stat().st_size:,} bytes); skipping")
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, target)
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")
    return str(target)


@app.function(image=image, volumes={"/artifacts": artifacts_volume}, timeout=1200, cpu=2)
def download_rtmpose_l_halpe26() -> str:
    """Pull RTMPose-l Halpe-26 (body7 pretrain, 256x192) onto /artifacts/checkpoints/."""
    url = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth"
    )
    target = Path(_CKPT_DIR) / "rtmpose_l_halpe26.pth"
    path = _https_pull(url, target)
    artifacts_volume.commit()
    return path


@app.function(image=image, volumes={"/artifacts": artifacts_volume}, timeout=1200, cpu=2)
def download_wasb_basketball_ckpt() -> str:
    """Pull WASB basketball checkpoint from upstream Drive onto /artifacts/checkpoints/.

    Source: ``nttcom/WASB-SBDT/MODEL_ZOO.md`` — Google Drive file id
    ``1nfECuSyJvPUmz3njZCdFERSQQbERt8FU``. Single-file Drive downloads are
    more reliable than the folder case that bit us on TrackID3x3 (see
    project feedback memory `feedback-gdown-drive-brittleness`).
    """
    import gdown

    target = Path(_CKPT_DIR) / "wasb_basketball.pth"
    if target.exists():
        print(f"wasb_basketball.pth already at {target} ({target.stat().st_size:,} bytes); skipping")
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id="1nfECuSyJvPUmz3njZCdFERSQQbERt8FU", output=str(target), quiet=False)
    if not target.exists():
        raise RuntimeError(f"WASB checkpoint download failed: {target} missing")
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")
    artifacts_volume.commit()
    return str(target)


@app.function(image=image, volumes={"/artifacts": artifacts_volume}, timeout=1200, cpu=2)
def download_poseconv3d_k400() -> str:
    """Pull PoseConv3D SlowOnly-R50 K400-keypoint onto /artifacts/checkpoints/."""
    url = (
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/"
        "slowonly_r50_8xb32-u48-240e_k400-keypoint/"
        "slowonly_r50_8xb32-u48-240e_k400-keypoint_20230731-7f498b55.pth"
    )
    target = Path(_CKPT_DIR) / "poseconv3d_k400.pth"
    path = _https_pull(url, target)
    artifacts_volume.commit()
    return path


@app.local_entrypoint()
def download_all() -> None:
    """Convenience: download all four checkpoints in sequence."""
    download_yolov8m.remote()
    download_rtmpose_l_halpe26.remote()
    download_wasb_basketball_ckpt.remote()
    download_poseconv3d_k400.remote()
    print(f"all checkpoints written under {_CKPT_DIR}")
