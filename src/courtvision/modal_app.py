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
    # `unar` (The Unarchiver) is the RAR backend for the `rarfile` Python
    # library on Debian — `unrar` itself lives in non-free and isn't in the
    # default debian:bookworm-slim apt sources.
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git", "unar")
    .pip_install_from_pyproject("pyproject.toml")
    # The OpenMMLab stack (mmengine/mmcv/mmpose/mmaction2) was REMOVED from the
    # shared image: it can't build from scratch on this image's torch 2.12+cu130
    # (mmcv source build fails — F20), and nothing on the shared image still
    # imports it (Stage 2 pose moved to rtmlib/ONNX on `rtmpose_image` below;
    # Stages 1/3 never used it). Phase E (PoseConv3D / mmaction2) will declare a
    # dedicated pinned-torch image — the same isolation pattern as `wasb_image`.
    .pip_install(
        # Dataset-download clients. Small; live in the shared image because
        # Modal forbids any build step (incl. `.pip_install`) after
        # `add_local_python_source`, so per-function image extensions don't
        # work for these. gdown is pinned to ≥5.2 — older versions lack the
        # `remaining_ok=True` kwarg used by `download_trackid3x3` to keep
        # going past Google-Drive per-file permission hiccups.
        "gdown>=5.2.0",
        "kaggle",
        "roboflow",
        "rarfile",  # uses the `unar` apt binary to read MultiSubjects RAR archives
    )
    .run_commands("mkdir -p /root/.cache/mmaction2")
    .add_local_python_source("courtvision")
)

# Stage 2 pose image — RTMPose-l Halpe-26 via rtmlib (ONNX), with NO torch /
# mmcv / mmpose / mmdet. The OpenMMLab stack can't run on the shared image's
# torch 2.12+cu130 (F20), so Stage 2 inference uses rtmlib + onnxruntime
# instead (same model + weights; PRD §5.2 deviation, decisions.md Decision 2).
# Lightweight — builds in ~1 min, runs CPU-only.
rtmpose_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "rtmlib",
        "onnxruntime",
        "opencv-python-headless",
        "numpy",
        "pydantic>=2",
        "rich",
    )
    .add_local_python_source("courtvision")
)
# NOTE: WASB (`nttcom/WASB-SBDT`) and Pose2Sim are NOT in the shared image.
# WASB has no setup.py/pyproject.toml — it's research code meant to be
# cloned + used in-tree. Phase D (Stage 3) and Stage 4 will declare
# dedicated images that git-clone WASB into /opt/wasb and add it to
# PYTHONPATH, and install Pose2Sim from PyPI (`pose2sim` package).
# Logged in flags.md as F9.

# Dedicated WASB (Stage 3) image — resolves F9. WASB is cloned into /opt/wasb
# and imported off sys.path by `courtvision.stage3_ball.wasb_inference`.
# Deps mirror WASB's Dockerfile, with two deliberate changes: (1) `numpy<2`
# because WASB's online tracker uses `np.Inf`, removed in NumPy 2.0; (2) torch
# 2.1.2 instead of their 1.11 (no CUDA-matching old wheels on modern Modal;
# HRNet is plain nn modules + torch<2.6 keeps `torch.load` weights_only=False,
# so the .pth.tar loads cleanly). No sklearn/skimage/timm/einops — grep of the
# WASB source confirms none are imported on the inference path.
wasb_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        "opencv-python-headless",
        "omegaconf",
        "hydra-core",
        "tqdm",
        "pandas",
        "pillow",
        "matplotlib",
        "pyyaml",
        # courtvision runtime deps used on the run_wasb path (schemas/io_utils
        # need pydantic; logging configures a rich handler at import time).
        "pydantic>=2",
        "rich",
    )
    .run_commands("git clone --depth 1 https://github.com/nttcom/WASB-SBDT.git /opt/wasb")
    .add_local_python_source("courtvision")
)

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
_KAGGLE_SECRETS: list[modal.Secret] = (
    [modal.Secret.from_local_environ(["KAGGLE_USERNAME", "KAGGLE_KEY"])]
    # Both env vars must be present at app-submit time — `from_local_environ`
    # raises if any named var is missing, which blocks every `modal run` on
    # this app (same failure mode as F8). Guarding on both makes the secret
    # list empty when either is unset; downstream Kaggle calls then fail
    # cleanly with a clear auth error instead of a cryptic app-submit blow-up.
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    else []
)
_ROBOFLOW_SECRETS: list[modal.Secret] = (
    [modal.Secret.from_local_environ(["ROBOFLOW_API_KEY"])]
    if os.environ.get("ROBOFLOW_API_KEY")
    else []
)


# ---------------------------------------------------------------------------
# Stages 1, 2, 3 — off-the-shelf inference
# ---------------------------------------------------------------------------


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_yolo_bytetrack(video_relpath: str, config: dict[str, Any]) -> str:
    """Stage 1: YOLOv8m + ByteTrack player detection + tracking.

    ``video_relpath`` is relative to the data Volume mount (``/data``).
    Output path defaults to
    ``stage1/<session>/tracks_<session>_<camera>.json``; pass
    ``config["output_relpath"]`` to override (used by the sweep harness
    to keep per-config outputs separate). Returns the relpath
    (under ``/artifacts``) of the written ``TrackSequence`` JSON.
    """
    from pathlib import Path as _Path

    from courtvision.io_utils import write_json
    from courtvision.stage1_detection.yolo_bytetrack import run_stage1

    video_path = _Path("/data") / video_relpath
    result = run_stage1(video_path, config)

    out_relpath: str = config.get(
        "output_relpath",
        f"stage1/{result.session_id}/tracks_{result.session_id}_{result.camera_id}.json",
    )
    write_json(result, _Path("/artifacts") / out_relpath)
    artifacts_volume.commit()
    return out_relpath


@app.function(image=image, volumes=VOLUMES, timeout=1800, cpu=2)
def make_stage1_overlay(
    video_relpath: str,
    tracks_relpath: str,
    max_seconds: float | None = None,
    output_relpath: str | None = None,
) -> str:
    """Render Stage 1 bbox + track-id overlay on top of the source video.

    Reads ``tracks_relpath`` (a ``TrackSequence`` JSON under ``/artifacts``)
    and ``video_relpath`` (under ``/data``); writes overlay MP4 to
    ``/artifacts/<output_relpath>`` (defaults to
    ``stage1_overlays/<session>_<camera>.mp4``) and returns that
    relpath. CPU-only. Reloads volumes first to see tracks just
    committed by a sibling Modal function.
    """
    from pathlib import Path as _Path

    from courtvision.io_utils import read_json_as
    from courtvision.schemas import TrackSequence
    from courtvision.stage1_detection.overlay import make_overlay_video

    artifacts_volume.reload()
    data_volume.reload()
    video_path = _Path("/data") / video_relpath
    tracks_path = _Path("/artifacts") / tracks_relpath
    sequence = read_json_as(TrackSequence, tracks_path)

    out_relpath = output_relpath or (
        f"stage1_overlays/{sequence.session_id}_{sequence.camera_id}.mp4"
    )
    out_path = _Path("/artifacts") / out_relpath
    make_overlay_video(video_path, sequence, out_path, max_seconds=max_seconds)
    artifacts_volume.commit()
    return out_relpath


@app.function(image=rtmpose_image, volumes=VOLUMES, timeout=3600, cpu=4)
def run_rtmpose(video_relpath: str, tracks_relpath: str, config: dict[str, Any]) -> str:
    """Stage 2: RTMPose-l Halpe-26 top-down on YOLO crops (rtmlib/ONNX, CPU).

    ``video_relpath`` is relative to ``/data``; ``tracks_relpath`` to
    ``/artifacts`` (the Stage 1 TrackSequence). Output defaults to
    ``stage2/<session>/pose2d_<session>_<camera>.json``; pass
    ``config["output_relpath"]`` to override. Returns the relpath (under
    ``/artifacts``) of the written ``Pose2DSequence`` JSON.
    """
    from pathlib import Path as _Path

    from courtvision.io_utils import read_json_as, write_json
    from courtvision.schemas import TrackSequence
    from courtvision.stage2_pose.rtmpose_inference import run_stage2

    artifacts_volume.reload()
    tracks = read_json_as(TrackSequence, _Path("/artifacts") / tracks_relpath)
    video_path = _Path("/data") / video_relpath
    result = run_stage2(tracks, video_path, config)

    out_relpath: str = config.get(
        "output_relpath",
        f"stage2/{result.session_id}/pose2d_{result.session_id}_{result.camera_id}.json",
    )
    write_json(result, _Path("/artifacts") / out_relpath)
    artifacts_volume.commit()
    return out_relpath


@app.function(image=image, volumes=VOLUMES, timeout=1800, cpu=2)
def make_stage2_overlay(
    video_relpath: str,
    poses_relpath: str,
    max_seconds: float | None = None,
    output_relpath: str | None = None,
) -> str:
    """Render the Stage 2 Halpe-26 skeleton overlay on top of the source video.

    Reads ``poses_relpath`` (a ``Pose2DSequence`` JSON under ``/artifacts``)
    and ``video_relpath`` (under ``/data``); writes the overlay MP4 to
    ``/artifacts/<output_relpath>`` (defaults to
    ``stage2_overlays/<session>_<camera>.mp4``) and returns that relpath.
    CPU-only; reloads volumes to see poses just committed by ``run_rtmpose``.
    """
    from pathlib import Path as _Path

    from courtvision.io_utils import read_json_as
    from courtvision.schemas import Pose2DSequence
    from courtvision.stage2_pose.overlay import make_pose_overlay_video

    artifacts_volume.reload()
    data_volume.reload()
    video_path = _Path("/data") / video_relpath
    poses_path = _Path("/artifacts") / poses_relpath
    sequence = read_json_as(Pose2DSequence, poses_path)

    out_relpath = output_relpath or (
        f"stage2_overlays/{sequence.session_id}_{sequence.camera_id}.mp4"
    )
    out_path = _Path("/artifacts") / out_relpath
    make_pose_overlay_video(video_path, sequence, out_path, max_seconds=max_seconds)
    artifacts_volume.commit()
    return out_relpath


@app.function(image=wasb_image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_wasb(video_relpath: str, config: dict[str, Any]) -> str:
    """Stage 3: WASB basketball detection.

    ``video_relpath`` is relative to the data Volume mount (``/data``).
    Output defaults to ``stage3/<session>/ball2d_<session>_<camera>.json``;
    pass ``config["output_relpath"]`` to override. Returns the relpath
    (under ``/artifacts``) of the written ``Ball2DSequence`` JSON.
    """
    from pathlib import Path as _Path

    from courtvision.io_utils import write_json
    from courtvision.stage3_ball.wasb_inference import run_stage3

    video_path = _Path("/data") / video_relpath
    result = run_stage3(video_path, config)

    out_relpath: str = config.get(
        "output_relpath",
        f"stage3/{result.session_id}/ball2d_{result.session_id}_{result.camera_id}.json",
    )
    write_json(result, _Path("/artifacts") / out_relpath)
    artifacts_volume.commit()
    return out_relpath


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_yolo_ball(video_relpath: str, config: dict[str, Any]) -> str:
    """Stage 3 (alt): appearance-based YOLOv8 ball detection. flags.md F19.

    Same I/O contract as ``run_wasb`` — reads ``/data/<video_relpath>``,
    writes a ``Ball2DSequence`` JSON to ``/artifacts`` (default
    ``stage3/<session>/ball2d_<session>_<camera>.json``), returns the relpath.
    Uses the shared image (ultralytics already installed).
    """
    from pathlib import Path as _Path

    from courtvision.io_utils import write_json
    from courtvision.stage3_ball.yolo_ball import run_stage3_yolo

    video_path = _Path("/data") / video_relpath
    result = run_stage3_yolo(video_path, config)

    out_relpath: str = config.get(
        "output_relpath",
        f"stage3/{result.session_id}/ball2d_{result.session_id}_{result.camera_id}.json",
    )
    write_json(result, _Path("/artifacts") / out_relpath)
    artifacts_volume.commit()
    return out_relpath


@app.function(image=image, volumes=VOLUMES, timeout=1800, cpu=2)
def make_stage3_overlay(
    video_relpath: str,
    ball_relpath: str,
    max_seconds: float | None = None,
    output_relpath: str | None = None,
) -> str:
    """Render the Stage 3 ball-detection overlay on top of the source video.

    Reads ``ball_relpath`` (a ``Ball2DSequence`` JSON under ``/artifacts``)
    and ``video_relpath`` (under ``/data``); writes the overlay MP4 to
    ``/artifacts/<output_relpath>`` (defaults to
    ``stage3_overlays/<session>_<camera>.mp4``) and returns that relpath.
    CPU-only. Reloads volumes first to see detections just committed by
    ``run_wasb``.
    """
    from pathlib import Path as _Path

    from courtvision.io_utils import read_json_as
    from courtvision.schemas import Ball2DSequence
    from courtvision.stage3_ball.overlay import make_ball_overlay_video

    artifacts_volume.reload()
    data_volume.reload()
    video_path = _Path("/data") / video_relpath
    ball_path = _Path("/artifacts") / ball_relpath
    sequence = read_json_as(Ball2DSequence, ball_path)

    out_relpath = output_relpath or (
        f"stage3_overlays/{sequence.session_id}_{sequence.camera_id}.mp4"
    )
    out_path = _Path("/artifacts") / out_relpath
    make_ball_overlay_video(video_path, sequence, out_path, max_seconds=max_seconds)
    artifacts_volume.commit()
    return out_relpath


@app.function(image=wasb_image, gpu="T4", volumes=VOLUMES, timeout=3600)
def diagnose_wasb(video_relpath: str, config: dict[str, Any]) -> dict[str, Any]:
    """Stage 3 heatmap diagnostic — is WASB firing on the ball at all?

    Runs ``diagnose_stage3`` (raw per-frame heatmap max + threshold-sweep
    distribution) and writes a heatmap-overlay video to
    ``/artifacts/stage3_diag/<session>_<camera>_hm.mp4``. Returns the stats
    dict (with the video relpath). Used to decide threshold-tuning vs
    fine-tuning before touching knobs.
    """
    from pathlib import Path as _Path

    from courtvision.stage3_ball.wasb_inference import diagnose_stage3

    session_id = str(config.get("session_id", "clip"))
    camera_id = str(config.get("camera_id", "cam"))
    hm_relpath = config.get("heatmap_relpath", f"stage3_diag/{session_id}_{camera_id}_hm.mp4")
    diag_config = {**config, "heatmap_video_path": str(_Path("/artifacts") / hm_relpath)}

    stats = diagnose_stage3(_Path("/data") / video_relpath, diag_config)
    stats["heatmap_relpath"] = hm_relpath
    artifacts_volume.commit()
    return stats


@app.function(image=wasb_image, volumes=VOLUMES, timeout=600, cpu=2)
def verify_wasb_checkpoint(
    weights_relpath: str = "checkpoints/wasb_basketball.pth",
) -> dict[str, Any]:
    """Sanity-check the WASB basketball checkpoint (resolves flags.md F3).

    Builds the WASB HRNet from its own config, loads the checkpoint's
    ``model_state_dict`` (strict=False to surface any key/shape mismatch),
    runs one CPU forward pass on a random clip-window, and reports param
    count, missing/unexpected keys, output shape, and heatmap value range.
    Empty missing/unexpected + a [0,1] sigmoid range == a real, complete,
    architecture-matching model (not a truncated 6 MB download).
    """
    import sys
    from pathlib import Path as _Path

    import torch
    from omegaconf import OmegaConf

    wasb_src = "/opt/wasb/src"
    if wasb_src not in sys.path:
        sys.path.insert(0, wasb_src)
    from models import build_model  # type: ignore[import-not-found]

    cfg = OmegaConf.create({"model": OmegaConf.load(f"{wasb_src}/configs/model/wasb.yaml")})
    model = build_model(cfg)
    model.eval()

    weights = _Path("/artifacts") / weights_relpath
    ckpt = torch.load(str(weights), map_location="cpu")
    ckpt_keys = sorted(ckpt.keys()) if isinstance(ckpt, dict) else ["<not-a-dict>"]
    state = (
        ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    )
    incompat = model.load_state_dict(state, strict=False)

    frames_in = int(cfg["model"]["frames_in"])
    h, w = int(cfg["model"]["inp_height"]), int(cfg["model"]["inp_width"])
    with torch.no_grad():
        out = model(torch.rand(1, 3 * frames_in, h, w))
    hm = out[0].sigmoid()

    report = {
        "weights": str(weights),
        "size_bytes": weights.stat().st_size,
        "checkpoint_keys": ckpt_keys,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_missing_keys": len(incompat.missing_keys),
        "n_unexpected_keys": len(incompat.unexpected_keys),
        "missing_keys_sample": list(incompat.missing_keys[:5]),
        "unexpected_keys_sample": list(incompat.unexpected_keys[:5]),
        "output_shape": list(out[0].shape),
        "heatmap_min": float(hm.min()),
        "heatmap_max": float(hm.max()),
    }
    print(report)
    return report


# ---------------------------------------------------------------------------
# Stage 5 — classifier training (transfer pipeline)
# ---------------------------------------------------------------------------


# Modal caps single-function timeout at 86400 s (24 h). PRD §9.5 estimates
# MultiSubjects ≈ 12-24 h on A10G and FineSports ≈ 24-48 h on A100-40GB,
# so FineSports in particular may exceed one timeout window. Phase E4
# will implement checkpoint-resume so a multi-day run becomes a sequence
# of ≤24 h invocations. Logged in flags.md as F10.
_MODAL_MAX_TIMEOUT_S = 3600 * 24


@app.function(
    image=image,
    gpu="A10G",
    volumes=VOLUMES,
    secrets=_WANDB_SECRETS,
    timeout=_MODAL_MAX_TIMEOUT_S,
)
def train_stage2_multisubjects(config: dict[str, Any]) -> str:
    """Transfer stage 2: full fine-tune PoseConv3D on MultiSubjects (≈12-24h, A10G).

    Returns the relpath of the saved checkpoint.
    """
    raise NotImplementedError("Phase E3")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes=VOLUMES,
    secrets=_WANDB_SECRETS,
    timeout=_MODAL_MAX_TIMEOUT_S,
)
def train_stage3_finesports(config: dict[str, Any]) -> str:
    """Transfer stage 3: full fine-tune PoseConv3D on FineSports (A100-40GB).

    PRD §9.5 estimates 24-48 h; if a single 24h timeout is insufficient,
    Phase E4 will implement checkpoint-resume so the run becomes a
    sequence of ≤24h invocations. Returns the relpath of the saved
    checkpoint.
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
    "diagnose_wasb",
    "eval_external_benchmark",
    "image",
    "make_stage1_overlay",
    "make_stage2_overlay",
    "make_stage3_overlay",
    "run_rtmpose",
    "run_wasb",
    "run_yolo_ball",
    "run_yolo_bytetrack",
    "rtmpose_image",
    "train_stage2_multisubjects",
    "train_stage3_finesports",
    "verify_wasb_checkpoint",
    "wasb_image",
]
