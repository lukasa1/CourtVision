"""Single Modal app for CourtVision. See PRD §8.

The shared image is built from ``pyproject.toml`` (core deps) plus the
OpenMMLab stack (``mmengine``, ``mmcv``, ``mmpose``, ``mmaction2``). The
stack lives in the Modal image — not in ``pyproject.toml`` — because mmcv
2.x doesn't build cleanly on macOS arm64 (PRD §13 risk #1 mitigation).

WASB and Pose2Sim are NOT in the shared image: WASB is git-only research
code, and including them would block Phase A downloads on a build failure.
Phase D / Stage 4 declare per-function images for those.

Function bodies are ``NotImplementedError`` stubs in Phase A. Phases B/C/D/E
fill them in, calling the corresponding ``courtvision.stageN_*`` modules.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Any

import modal

app = modal.App("courtvision")

# Image: Linux + Python 3.11 + apt deps + project deps + OpenMMLab + git deps.
# `add_local_python_source` mounts the package at runtime so changes in the
# repo are picked up without rebuilding the image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install(
        # OpenMMLab stack — Modal Linux x86_64 has prebuilt wheels.
        "mmengine",
        "mmcv>=2.0",
        "mmpose>=1.3",
        "mmaction2>=1.2",
    )
    .run_commands("mkdir -p /root/.cache/mmaction2")
    .add_local_python_source("courtvision")
)
# NOTE: WASB (`nttcom/WASB-SBDT`) and Pose2Sim are NOT in the shared image.
# WASB has no setup.py/pyproject.toml — it's research code meant to be
# cloned + used in-tree. Phase D (Stage 3) and Stage 4 will declare
# dedicated images that git-clone WASB into /opt/wasb and add it to
# PYTHONPATH, and install Pose2Sim from PyPI (`pose2sim` package).
# Logged in flags.md as F9.

# Volumes: data (datasets) + artifacts (checkpoints, eval JSONs, inference
# outputs). Created on first use; persist across runs.
data_volume = modal.Volume.from_name("courtvision-data", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("courtvision-artifacts", create_if_missing=True)

VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/data": data_volume,
    "/artifacts": artifacts_volume,
}

# Local-env-passthrough secrets. Both W&B and HF credentials live in the
# user's local shell (`.env` → `set -a; source .env; set +a`) and ride
# along to Modal at submit time. No persistent `modal secret create` step.
# If the env var isn't set, the secret list is empty — downstream code
# handles missing creds gracefully (W&B disables itself; HF allows public
# datasets only).
_WANDB_SECRETS: list[modal.Secret] = (
    [modal.Secret.from_local_environ(["WANDB_API_KEY"])] if os.environ.get("WANDB_API_KEY") else []
)
_HF_SECRETS: list[modal.Secret] = (
    [modal.Secret.from_local_environ(["HF_TOKEN"])] if os.environ.get("HF_TOKEN") else []
)


# ---------------------------------------------------------------------------
# Stages 1, 2, 3 — off-the-shelf inference
# ---------------------------------------------------------------------------


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_yolo_bytetrack(video_relpath: str, config: dict[str, Any]) -> str:
    """Stage 1: YOLOv8m + ByteTrack player detection + tracking.

    Returns the relpath (under ``/artifacts``) of the written
    ``TrackSequence`` JSON.
    """
    raise NotImplementedError("Phase B")


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_rtmpose(video_relpath: str, tracks_relpath: str, config: dict[str, Any]) -> str:
    """Stage 2: RTMPose-l Halpe-26 top-down on YOLO crops.

    Returns the relpath of the written ``Pose2DSequence`` JSON.
    """
    raise NotImplementedError("Phase C")


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_wasb(video_relpath: str, config: dict[str, Any]) -> str:
    """Stage 3: WASB basketball detection.

    Returns the relpath of the written ``Ball2DSequence`` JSON.
    """
    raise NotImplementedError("Phase D")


# ---------------------------------------------------------------------------
# Stage 5 — classifier training (transfer pipeline)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A10G",
    volumes=VOLUMES,
    secrets=_WANDB_SECRETS,
    timeout=3600 * 36,
)
def train_stage2_multisubjects(config: dict[str, Any]) -> str:
    """Transfer stage 2: full fine-tune PoseConv3D on MultiSubjects (≈24h, A10G).

    Returns the relpath of the saved checkpoint.
    """
    raise NotImplementedError("Phase E3")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes=VOLUMES,
    secrets=_WANDB_SECRETS,
    timeout=3600 * 48,
)
def train_stage3_finesports(config: dict[str, Any]) -> str:
    """Transfer stage 3: full fine-tune PoseConv3D on FineSports (≈24h, A100-40GB).

    Returns the relpath of the saved checkpoint.
    """
    raise NotImplementedError("Phase E4")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600 * 4)
def eval_external_benchmark(ckpt_relpath: str, dataset: str) -> dict[str, Any]:
    """Run a checkpoint against an external benchmark (MultiSubjects /
    FineSports / TrackID3x3 / U Arizona / WASB-basketball).
    """
    raise NotImplementedError("Phase E3 / E4")


__all__ = [
    "VOLUMES",
    "app",
    "artifacts_volume",
    "data_volume",
    "eval_external_benchmark",
    "image",
    "run_rtmpose",
    "run_wasb",
    "run_yolo_bytetrack",
    "train_stage2_multisubjects",
    "train_stage3_finesports",
]
