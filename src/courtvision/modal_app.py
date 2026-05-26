"""Single Modal app for CourtVision. See PRD §8.

The image is built from ``pyproject.toml`` (core deps) plus the OpenMMLab
stack (``mmengine``, ``mmcv``, ``mmpose``, ``mmaction2``) plus WASB and
Pose2Sim from git. The stack is installed *inside* the Modal image — not in
``pyproject.toml`` — because mmcv 2.x doesn't build cleanly on macOS arm64
(PRD §13 risk #1 mitigation).

Function bodies are ``NotImplementedError`` stubs in Phase A. Phases B/C/D/E
fill them in, calling the corresponding ``courtvision.stageN_*`` modules.
"""

from __future__ import annotations

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
    .pip_install(
        # WASB + Pose2Sim are git-only.
        "git+https://github.com/nttcom/WASB-SBDT.git",
        "git+https://github.com/perfanalytics/pose2sim.git",
    )
    .run_commands("mkdir -p /root/.cache/mmaction2")
    .add_local_python_source("courtvision")
)

# Volumes: data (datasets) + artifacts (checkpoints, eval JSONs, inference
# outputs). Created on first use; persist across runs.
data_volume = modal.Volume.from_name("courtvision-data", create_if_missing=True)
artifacts_volume = modal.Volume.from_name(
    "courtvision-artifacts", create_if_missing=True
)

VOLUMES = {
    "/data": data_volume,
    "/artifacts": artifacts_volume,
}

# W&B credential mounted as a Modal Secret. Create with:
#   modal secret create wandb-secret WANDB_API_KEY=...
WANDB_SECRET = modal.Secret.from_name("wandb-secret", required_keys=[])


# ---------------------------------------------------------------------------
# Stages 1, 2, 3 — off-the-shelf inference
# ---------------------------------------------------------------------------


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_yolo_bytetrack(video_relpath: str, config: dict) -> str:
    """Stage 1: YOLOv8m + ByteTrack player detection + tracking.

    Returns the relpath (under ``/artifacts``) of the written
    ``TrackSequence`` JSON.
    """
    raise NotImplementedError("Phase B")


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_rtmpose(video_relpath: str, tracks_relpath: str, config: dict) -> str:
    """Stage 2: RTMPose-l Halpe-26 top-down on YOLO crops.

    Returns the relpath of the written ``Pose2DSequence`` JSON.
    """
    raise NotImplementedError("Phase C")


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_wasb(video_relpath: str, config: dict) -> str:
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
    secrets=[WANDB_SECRET],
    timeout=3600 * 36,
)
def train_stage2_multisubjects(config: dict) -> str:
    """Transfer stage 2: full fine-tune PoseConv3D on MultiSubjects (≈24h, A10G).

    Returns the relpath of the saved checkpoint.
    """
    raise NotImplementedError("Phase E3")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes=VOLUMES,
    secrets=[WANDB_SECRET],
    timeout=3600 * 48,
)
def train_stage3_finesports(config: dict) -> str:
    """Transfer stage 3: full fine-tune PoseConv3D on FineSports (≈24h, A100-40GB).

    Returns the relpath of the saved checkpoint.
    """
    raise NotImplementedError("Phase E4")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600 * 4)
def eval_external_benchmark(ckpt_relpath: str, dataset: str) -> dict:
    """Run a checkpoint against an external benchmark (MultiSubjects /
    FineSports / TrackID3x3 / U Arizona / WASB-basketball).
    """
    raise NotImplementedError("Phase E3 / E4")


__all__ = [
    "app",
    "image",
    "data_volume",
    "artifacts_volume",
    "VOLUMES",
    "WANDB_SECRET",
    "run_yolo_bytetrack",
    "run_rtmpose",
    "run_wasb",
    "train_stage2_multisubjects",
    "train_stage3_finesports",
    "eval_external_benchmark",
]
