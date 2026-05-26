# CourtVision — Product Requirements Document (PRD)

**Project:** CourtVision — Multi-view 3D skeletal tracking + shot-contest measurement for half-court 2-on-2 basketball
**Author:** Lukas Allard
**Course:** Stanford CS231N, Spring 2026
**Date:** May 26, 2026
**Document purpose:** Definitive specification for Claude Code to scaffold the project repository and begin implementation of Stages 1, 2, 3, and 5. Stages 4, 6, 7, 8 are defined as I/O contracts + stub modules so early stages can be built against stable interfaces.

---

## 1. Executive Summary

CourtVision processes synchronized footage from three iPhones recording half-court 2-on-2 basketball and produces, per detected shot, a multi-dimensional contest metric quantifying defender pressure at the release frame. The headline novel measurement is **3D defender-wrist distance to the actual ball trajectory** — a refinement of the NBA SportVU floor-distance convention that uses 3D hand tracking unavailable to existing tracking systems.

The pipeline has 8 stages. Stages 1-4 are off-the-shelf (player detection/tracking, 2D pose, ball detection, multi-view triangulation). Stage 5 is the only trained component — a two-stream PoseConv3D + ball-trajectory MLP shot classifier trained via a 4-stage transfer pipeline (Kinetics → MultiSubjects → FineSports → own footage). Stages 6-8 are geometric (release-frame extraction, contest metric) and a coaching-review demo.

**Scope of this PRD:** scaffold the full repository now; implement Stages 1, 2, 3, 5 in a 2-week sprint while the user's own footage and weeks-3-4 stages (4, 6, 7, 8) remain in interface-only stub form.

---

## 2. Background and Locked Decisions

The decision space is settled in `decisions.md` (10 locked technical decisions). This PRD does not re-litigate those; it operationalizes them. Key locked choices:

- **Player detect + track:** YOLOv8m + ByteTrack, person class only.
- **2D pose:** RTMPose-l, Halpe-26, top-down on YOLO crops.
- **Ball detection:** WASB (BMVC 2023), off-the-shelf.
- **Triangulation:** Pose2Sim DLT for pose; `cv2.triangulatePoints` for ball; chessboard intrinsics + court-landmark extrinsics.
- **Shot classifier:** PoseConv3D SlowOnly-R50 + ball-trajectory MLP (two-stream, head-level fusion). COCO-17 subset of Halpe-26 for classifier; full Halpe-26 retained for geometric pipeline.
- **Transfer chain:** Path C (4-stage) primary — Kinetics → MultiSubjects → FineSports → own footage with LP-FT in final stage. Path A (3-stage, no FineSports) is the fallback.
- **Release frame:** 4-condition geometric rule on 3D pose + ball, with a planned backward-trace-from-apex variant for head-to-head comparison.
- **Contest metric:** 5 measurements at release; 3D wrist-to-trajectory distance is the headline novel metric.
- **Calibration:** chessboard intrinsics one-time per phone; court-landmark extrinsics per session; AE/AF lock per session.

All off-the-shelf tool choices are **locked for Phase 1** — no swappable backends, no runtime selection. Phase 2 (★ stretch) substitutions are out of scope for this PRD.

---

## 3. Project Goals and Non-Goals

### 3.1 Goals (2-week sprint)

1. Scaffold a clean, reproducible Python repository ready for Stages 1, 2, 3, 5 implementation.
2. Implement **Stage 1** (YOLOv8m + ByteTrack) — off-the-shelf inference + quantitative MOTA/IDF1 eval on TrackID3x3.
3. Implement **Stage 2** (RTMPose-l Halpe-26) — off-the-shelf inference + wrist-accuracy eval on TrackID3x3 and BlazePose-33→COCO-17 comparison on MultiSubjects.
4. Implement **Stage 3** (WASB) — off-the-shelf inference + mAP eval on U Arizona slice + release-window recall on WASB basketball subset.
5. Implement **Stage 5 scaffolding + pretraining** — PoseConv3D Kinetics-400 checkpoint loaded, MultiSubjects dataloader + Stage 2 training run kicked off, FineSports dataloader + Stage 3 training run kicked off. Stage 4 (LP-FT on own footage) and Stage 5 ball-MLP head training are out of scope until own footage exists.
6. Define **I/O contracts** for Stages 4, 6, 7, 8 so downstream code can be written without refactoring later.

### 3.2 Non-goals (deferred to weeks 3-4)

- Own footage capture, calibration, and recording.
- Stage 4 implementation (Pose2Sim triangulation requires own footage + calibration).
- Stage 5 LP-FT on own footage and ball-MLP head training (requires own labeled releases).
- Stage 6 release-frame extraction (requires own labeled releases for threshold tuning).
- Stage 7 contest metric implementation (requires Stage 4 outputs).
- Stage 8 3D playback demo (requires the whole pipeline).
- Phase 2 ★ fine-tunes (RTMPose on TrackID3x3, WASB custom training) — only triggered if Phase 1 evaluation gates fail.
- Rater study for contest-metric validation.

---

## 4. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Input: 3 iPhone clips @ 1080p 60fps           │
│                    (synchronized via audio clap)                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Stage 1    │         │  Stage 3    │         │  (calibrate)│
│  YOLO+BT    │         │  WASB ball  │         │  cam params │
│  per-cam    │         │  per-cam    │         │             │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       │                       │                       │
       ▼                       │                       │
┌─────────────┐                │                       │
│  Stage 2    │                │                       │
│  RTMPose-l  │                │                       │
│  Halpe-26   │                │                       │
└──────┬──────┘                │                       │
       │                       │                       │
       └──────────┬────────────┴───────────────────────┘
                  ▼
            ┌──────────────────────┐
            │   Stage 4            │
            │   Triangulation      │ (stub for now)
            │   Pose2Sim + cv2     │
            └──────────┬───────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Stage 5     │ │  Stage 6     │ │  Stage 7     │
│  Shot        │ │  Release     │ │  Contest     │
│  classifier  │ │  frame       │ │  metric      │
│  (trained)   │ │  (stub)      │ │  (stub)      │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
                ┌──────────────┐
                │  Stage 8     │
                │  3D demo     │ (stub)
                └──────────────┘
```

---

## 5. Tech Stack and Tooling

### 5.1 Decisions made

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | Maximum compatibility with current MMAction2/MMPose; 3.12 has too many ecosystem rough edges. |
| Package layout | `src/courtvision/` | Importable as `courtvision.stage1`, etc. |
| Dep mgmt | `uv` + `pyproject.toml` + `uv.lock` | Fast, reproducible. Single source of truth for local + Modal. |
| Repo name | `CourtVision` | GitHub private. |
| Package name | `courtvision` | Lowercase, importable. |
| Experiment tracking | Weights & Biases | Free for students; ablation comparison is the killer feature. |
| Config mgmt | Hydra (`hydra-core`) | YAML composition + CLI overrides. |
| Compute | Modal (single app, functions per stage) | All GPU work remote. Local dev is CPU-only (macOS). |
| Storage | Modal Volume (`courtvision-data`, `courtvision-artifacts`) + small local mirror | ~50-clip subset locally for dataloader iteration. |
| Testing | pytest, smoke tests per stage | Tiny synthetic inputs; checks output shape/schema. |
| Secrets | `.env` + `python-dotenv`; `.env.example` committed | Modal secrets separately via `modal secret create`. |
| Linting / formatting | `ruff` (lint + format) | Single tool replaces black + isort + flake8. |
| Type checking | `mypy` strict on `courtvision/`, lenient on scripts | Catches schema drift between stages. |
| Pre-commit | `pre-commit` running ruff + mypy | Keeps the repo clean. |
| Git LFS | For checkpoints stored in-repo (small ones only) | Big checkpoints live on Modal Volume. |
| GPU budget | ~$200 over 2 weeks | A10G / L4 default; A100 for FineSports training if needed. |
| OS target | macOS local + Linux on Modal | All scripts must run on both. Avoid `/tmp` for outputs — use repo-relative paths. |

### 5.2 Locked dependencies (Phase 1)

**Core ML:**
- `torch >= 2.1`, `torchvision`, `torchaudio`
- `ultralytics` (YOLOv8m)
- `mmengine`, `mmcv >= 2.0`, `mmpose >= 1.3`, `mmaction2 >= 1.2` — pin exact patch versions in `uv.lock`
- `opencv-python-headless` (no GUI dep on Modal)
- `numpy < 2.0` (compatibility with mmcv at time of writing — verify)
- `scipy`

**Tracking + utilities:**
- `wandb`
- `hydra-core`, `omegaconf`
- `pydantic >= 2` for I/O schemas
- `tqdm`, `rich` (logging)
- `python-dotenv`

**Modal + data:**
- `modal`
- `huggingface_hub` (downloading MMAction2 checkpoints)

**Dev:**
- `pytest`, `pytest-cov`
- `ruff`, `mypy`
- `pre-commit`
- `jupyter`, `ipykernel` (for ad-hoc exploration)

**WASB** is installed from source (`git+https://github.com/nttcom/WASB-SBDT.git`) — it's not on PyPI.
**Pose2Sim** is installed from source (`git+https://github.com/perfanalytics/pose2sim.git`) for Stage 4 (stub for now, but list the dep).

---

## 6. Repository Layout

```
CourtVision/
├── README.md                          # Setup, usage, project context
├── pyproject.toml                     # Project metadata + deps (uv-managed)
├── uv.lock                            # Lockfile
├── .env.example                       # Required env vars (WANDB_API_KEY, MODAL_TOKEN_*, etc.)
├── .gitignore                         # Standard Python + .env + data/ + outputs/
├── .pre-commit-config.yaml            # ruff + mypy hooks
├── .python-version                    # 3.11
├── LICENSE                            # (TBD; user choice — leave empty until they decide)
│
├── configs/                           # Hydra configs
│   ├── config.yaml                    # Root config
│   ├── data/                          # Per-dataset configs
│   │   ├── trackid3x3.yaml
│   │   ├── multisubjects.yaml
│   │   ├── finesports.yaml
│   │   ├── u_arizona.yaml
│   │   └── own_footage.yaml           # (stub — paths TBD)
│   ├── stage1_detection/
│   │   ├── yolov8m_bytetrack.yaml     # Locked model + tracker config
│   │   └── eval.yaml                  # Eval-only run (TrackID3x3 MOTA)
│   ├── stage2_pose/
│   │   ├── rtmpose_l_halpe26.yaml
│   │   └── eval.yaml
│   ├── stage3_ball/
│   │   ├── wasb.yaml
│   │   └── eval.yaml
│   ├── stage4_triangulation/          # (stub for now)
│   │   └── pose2sim.yaml
│   ├── stage5_classifier/
│   │   ├── poseconv3d_slowonly_r50.yaml  # Architecture
│   │   ├── train_stage2_multisubjects.yaml
│   │   ├── train_stage3_finesports.yaml
│   │   ├── train_stage4_lpft.yaml     # (stub — needs own footage)
│   │   ├── train_stage5_ball_mlp.yaml # (stub — needs own footage)
│   │   └── infer.yaml                 # Sliding-window inference
│   ├── stage6_release/                # (stub)
│   ├── stage7_contest/                # (stub)
│   ├── stage8_demo/                   # (stub)
│   └── modal/
│       ├── image.yaml                 # Image definition shared across functions
│       ├── volumes.yaml               # Volume names
│       └── gpu.yaml                   # GPU type per function (T4 / L4 / A10G / A100)
│
├── src/courtvision/
│   ├── __init__.py
│   ├── schemas.py                     # All cross-stage I/O contracts (pydantic models)
│   ├── io_utils.py                    # JSON read/write with schema validation
│   ├── logging.py                     # rich + wandb wrappers
│   ├── paths.py                       # Centralized path resolution (local vs. Modal)
│   ├── modal_app.py                   # Single Modal app definition
│   │
│   ├── stage1_detection/
│   │   ├── __init__.py
│   │   ├── yolo_bytetrack.py          # Per-camera inference
│   │   ├── eval.py                    # MOTA / IDF1 on TrackID3x3
│   │   └── README.md
│   │
│   ├── stage2_pose/
│   │   ├── __init__.py
│   │   ├── rtmpose_inference.py       # Top-down on YOLO crops
│   │   ├── halpe_to_coco17.py         # Keypoint subsetting (used by classifier)
│   │   ├── eval.py                    # PCK on TrackID3x3, MS comparison
│   │   └── README.md
│   │
│   ├── stage3_ball/
│   │   ├── __init__.py
│   │   ├── wasb_inference.py
│   │   ├── eval.py                    # mAP + release-window recall
│   │   └── README.md
│   │
│   ├── stage4_triangulation/
│   │   ├── __init__.py
│   │   ├── triangulate.py             # STUB — raises NotImplementedError
│   │   ├── calibration.py             # STUB
│   │   └── README.md                  # Spec lives here
│   │
│   ├── stage5_classifier/
│   │   ├── __init__.py
│   │   ├── model/
│   │   │   ├── poseconv3d.py          # MMAction2 model factory
│   │   │   ├── ball_mlp.py            # Two-stream MLP head
│   │   │   └── two_stream.py          # Combined model
│   │   ├── data/
│   │   │   ├── kinetics.py            # (download only — no loading needed)
│   │   │   ├── multisubjects.py       # Dataloader for Stage 2 of transfer
│   │   │   ├── finesports.py          # Dataloader for Stage 3 of transfer
│   │   │   ├── own_footage.py         # STUB — Stage 4 of transfer
│   │   │   └── windowing.py           # 32-frame sliding-window logic, shared
│   │   ├── train.py                   # Hydra-driven training entrypoint
│   │   ├── infer.py                   # Sliding-window inference + ball-ascent gate + NMS
│   │   ├── eval.py                    # External benchmarks + ablation eval
│   │   └── README.md
│   │
│   ├── stage6_release/
│   │   ├── __init__.py
│   │   ├── rule_forward.py            # STUB — 4-condition rule
│   │   ├── rule_backward.py           # STUB — backward-trace-from-apex
│   │   ├── eval.py                    # STUB
│   │   └── README.md                  # Spec
│   │
│   ├── stage7_contest/
│   │   ├── __init__.py
│   │   ├── measurements.py            # STUB — 5 measurements
│   │   ├── eval.py                    # STUB
│   │   └── README.md                  # Spec
│   │
│   └── stage8_demo/
│       ├── __init__.py
│       ├── playback.py                # STUB
│       └── README.md                  # Spec
│
├── scripts/                           # CLI entrypoints, wrap Modal functions
│   ├── stage1_run.py
│   ├── stage1_eval.py
│   ├── stage2_run.py
│   ├── stage2_eval.py
│   ├── stage3_run.py
│   ├── stage3_eval.py
│   ├── stage5_train_multisubjects.py
│   ├── stage5_train_finesports.py
│   ├── stage5_eval_external.py
│   ├── download_checkpoints.py        # Pulls Kinetics PoseConv3D + RTMPose-l + WASB + YOLO weights to Volume
│   └── download_datasets.py           # Pulls TrackID3x3 / MultiSubjects / FineSports / U Arizona to Volume
│
├── notebooks/                         # Exploration
│   ├── 00_setup_check.ipynb           # Verify env, Modal connection
│   ├── 01_stage1_visualize.ipynb      # Plot tracks on a clip
│   ├── 02_stage2_visualize.ipynb      # Plot poses
│   ├── 03_stage3_visualize.ipynb      # Plot ball trajectory
│   └── 05_stage5_inference_check.ipynb
│
├── tests/
│   ├── conftest.py                    # Shared fixtures + tiny test clips
│   ├── fixtures/                      # 1-second 3-frame mini-clips, synthetic poses
│   ├── test_schemas.py
│   ├── test_stage1_smoke.py
│   ├── test_stage2_smoke.py
│   ├── test_stage3_smoke.py
│   └── test_stage5_smoke.py
│
├── docs/
│   ├── prd.md                         # This file
│   ├── decisions.md                   # Symlink or copy from Milestone 2 folder
│   ├── milestone2_technical_approach.pdf  # Reference
│   └── sprint_plan_week1.md           # Day-by-day plan
│
├── data/                              # gitignored. Local mirror of datasets (50-clip subsets).
│   └── .gitkeep
│
├── outputs/                           # gitignored. Local copies of inference / training outputs.
│   └── .gitkeep
│
└── modal_workspace/                   # (only used at runtime; mounted as Volume on Modal)
    ├── data/                          # Full datasets
    └── artifacts/                     # Model checkpoints, eval JSONs
```

---

## 7. Cross-Stage I/O Contracts (Pydantic Schemas)

These are the **stable interfaces** between stages. Define once in `src/courtvision/schemas.py`. All stage outputs are JSON files (one per session per camera where applicable), validated against these schemas on write and on read.

### 7.1 Stage 1 output: per-camera tracks

```python
# tracks_{session_id}_{camera_id}.json
class Track(BaseModel):
    track_id: int
    bbox: list[float]            # [x, y, w, h] in pixels
    confidence: float
    frame_idx: int

class TrackSequence(BaseModel):
    session_id: str
    camera_id: str
    fps: float
    n_frames: int
    tracks: list[Track]
```

### 7.2 Stage 2 output: per-camera 2D pose

```python
# pose2d_{session_id}_{camera_id}.json
class Keypoint2D(BaseModel):
    x: float
    y: float
    score: float

class Pose2D(BaseModel):
    frame_idx: int
    track_id: int
    keypoints: list[Keypoint2D]  # length 26 (Halpe-26)
    keypoint_format: Literal["halpe26"] = "halpe26"

class Pose2DSequence(BaseModel):
    session_id: str
    camera_id: str
    fps: float
    poses: list[Pose2D]
```

### 7.3 Stage 3 output: per-camera ball detections

```python
# ball2d_{session_id}_{camera_id}.json
class BallDetection2D(BaseModel):
    frame_idx: int
    x: float
    y: float
    score: float

class Ball2DSequence(BaseModel):
    session_id: str
    camera_id: str
    fps: float
    detections: list[BallDetection2D]
```

### 7.4 Stage 4 output (CONTRACT ONLY — stub implementation): 3D pose + ball + visibility

```python
# pose3d_{session_id}.json, ball3d_{session_id}.json
class Keypoint3D(BaseModel):
    x: float; y: float; z: float
    confidence: float            # mean of contributing 2D scores
    n_cameras: int               # 3 / 2 / 0

class Pose3D(BaseModel):
    frame_idx: int
    track_id: int
    keypoints: list[Keypoint3D]  # Halpe-26 in world coords (meters)
    reprojection_error_px: float

class Pose3DSequence(BaseModel):
    session_id: str
    fps: float
    poses: list[Pose3D]
    calibration_ref: str         # path to extrinsics file used

class BallDetection3D(BaseModel):
    frame_idx: int
    x: float; y: float; z: float
    n_cameras: int
    reprojection_error_px: float
```

### 7.5 Stage 5 output: shot events

```python
# shots_{session_id}.json
class ShotEvent(BaseModel):
    event_id: str
    shooter_track_id: int
    window_start_frame: int
    window_end_frame: int
    peak_frame: int              # frame of max P(shoot)+P(layup)
    p_shoot: float
    p_layup: float
    p_dribble: float
    p_background: float
    per_camera_probs: dict[str, dict[str, float]]   # camera_id -> class -> prob

class ShotEventSequence(BaseModel):
    session_id: str
    events: list[ShotEvent]
```

### 7.6 Stage 6 output: release frames

```python
# releases_{session_id}.json
class ReleaseFrame(BaseModel):
    event_id: str
    release_frame_idx: int
    method: Literal["forward_rule", "backward_trace", "fallback_apex"]
    confidence_flag: Literal["high", "low"]
    per_keypoint_confidence: dict[str, float]
    n_cameras_at_release: int
```

### 7.7 Stage 7 output: contest measurements

```python
# contest_{session_id}.json
class ContestMeasurement(BaseModel):
    event_id: str
    wrist_to_trajectory_distance_m: float | None      # signed; None if no defender within 3m
    wrist_height_advantage_m: float | None
    shooter_velocity_forward_mps: float
    shooter_velocity_lateral_mps: float
    shooter_velocity_vertical_mps: float
    floor_distance_m: float | None
    composite_score: float | None
    closest_defender_track_id: int | None
    flags: list[str]                                  # "uncontested" / "id_swap" / "weak_triangulation"
```

These schemas are the **contract Claude Code must implement faithfully**. Stage 1, 2, 3 producers and Stage 5 consumer all key off them in the sprint; Stages 4, 6, 7 outputs are written by stub functions that raise `NotImplementedError("Stage N — week 3-4")` but their schema modules exist.

---

## 8. Modal Infrastructure

### 8.1 Single app, functions per stage

```python
# src/courtvision/modal_app.py
import modal

app = modal.App("courtvision")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install_from_pyproject("pyproject.toml")    # syncs with local
    .pip_install(
        "git+https://github.com/nttcom/WASB-SBDT.git",
        "git+https://github.com/perfanalytics/pose2sim.git",
    )
    .run_commands(
        # MMAction2 checkpoint cache directory
        "mkdir -p /root/.cache/mmaction2",
    )
)

data_volume = modal.Volume.from_name("courtvision-data", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("courtvision-artifacts", create_if_missing=True)

VOLUMES = {
    "/data": data_volume,
    "/artifacts": artifacts_volume,
}

# ---------- Off-the-shelf inference (Stages 1, 2, 3) ----------

@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_yolo_bytetrack(video_relpath: str, config: dict) -> str:
    ...

@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_rtmpose(video_relpath: str, tracks_relpath: str, config: dict) -> str:
    ...

@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600)
def run_wasb(video_relpath: str, config: dict) -> str:
    ...

# ---------- Stage 5 training ----------

@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=3600 * 36)
def train_stage2_multisubjects(config: dict) -> str:
    ...

@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, timeout=3600 * 48)
def train_stage3_finesports(config: dict) -> str:
    ...

# ---------- Eval ----------

@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=3600 * 4)
def eval_external_benchmark(ckpt_relpath: str, dataset: str) -> dict:
    ...
```

### 8.2 GPU assignments

| Function | Default GPU | Rationale |
|---|---|---|
| Stage 1 (YOLO+BT) inference | T4 | YOLOv8m fits in <8GB VRAM easily. |
| Stage 2 (RTMPose) inference | T4 | RTMPose-l fits in <8GB VRAM. |
| Stage 3 (WASB) inference | T4 | HRNet backbone fits in T4. |
| Stage 5 Stage-2 training (MultiSubjects) | A10G | ~12-24h wall-clock; A10G is the sweet spot. |
| Stage 5 Stage-3 training (FineSports) | A100-40GB | ~24-48h on A10G → drop to ~12-24h on A100. User's "fast over cheap" budget supports this. |
| Stage 5 inference | T4 | Sliding window over a clip; not memory-bound. |
| External benchmark eval | T4 | Inference-only. |

### 8.3 Dataset storage strategy

**All large datasets live on Modal Volumes, not on the laptop.** Modal Volume storage is ~$0.15/GB/month, so the full ~100-180 GB of datasets cost ~$15-27/month — trivial relative to the GPU budget. Volumes mount at filesystem speed inside Modal functions (`/data` looks like local disk), persist across runs, and are versioned.

**Two Volumes:**

| Volume | Mount path | Contents | Approx size |
|---|---|---|---|
| `courtvision-data` | `/data` | Training + eval datasets | ~100-180 GB |
| `courtvision-artifacts` | `/artifacts` | Pretrained weights, trained checkpoints, eval JSONs, inference outputs | ~10-30 GB |

**Dataset breakdown on `courtvision-data`:**

| Dataset | Approx size | Path on Volume | Purpose |
|---|---|---|---|
| MultiSubjects | ~30-50 GB | `/data/multisubjects/` | Stage 5 transfer-stage-2 training; classifier benchmark |
| FineSports | ~50-100 GB | `/data/finesports/` | Stage 5 transfer-stage-3 training; temporal-localization benchmark |
| TrackID3x3 Indoor | ~5-10 GB | `/data/trackid3x3/` | Stage 1 IDF1 eval; Stage 2 wrist PCK eval |
| U Arizona Roboflow basketball | ~2 GB | `/data/u_arizona/` | Stage 3 mAP eval |
| WASB basketball subset | ~5-10 GB | `/data/wasb_basketball/` | Stage 3 release-window recall eval |
| Own footage | (TBD weeks 3-4) | `/data/own_footage/` | Stage 5 transfer-stage-4 LP-FT (deferred) |

**Pretrained checkpoints on `courtvision-artifacts`:**

| Checkpoint | Approx size | Path on Volume |
|---|---|---|
| YOLOv8m COCO weights | ~50 MB | `/artifacts/checkpoints/yolov8m.pt` |
| RTMPose-l Halpe-26 | ~300 MB | `/artifacts/checkpoints/rtmpose_l_halpe26.pth` |
| WASB basketball | ~200 MB | `/artifacts/checkpoints/wasb_basketball.pth` |
| PoseConv3D SlowOnly-R50 Kinetics-400 | ~250 MB | `/artifacts/checkpoints/poseconv3d_k400.pth` |

**One-time download pattern (runs ON Modal, never touches your laptop):**

```python
# scripts/download_datasets.py
@app.function(volumes={"/data": data_volume}, timeout=3600 * 6, cpu=4)
def download_multisubjects():
    # Pull dataset directly to /data/multisubjects/ on the Volume
    # Source: Hugging Face or original release URL
    ...
    data_volume.commit()    # persist
```

Run with `modal run scripts/download_datasets.py::download_multisubjects`. The download happens on Modal's bandwidth and writes straight to the Volume. Your laptop's network and disk are irrelevant.

One Modal function per dataset, each with appropriate timeout. The script also includes `download_pretrained_checkpoints()` for the artifacts Volume.

**Why NOT local disk:**
- ~180 GB exceeds typical laptop free space and eats SSD lifespan.
- Re-downloading after a `rm` is painful (FineSports especially).
- Modal jobs would have to re-upload from your laptop each run — slow and bandwidth-expensive.

**Why NOT S3/GCS:**
- Adds a third service to manage.
- Modal Volumes mount at filesystem speed; S3 needs explicit fetch/cache logic.
- Cost is comparable for this scale.

### 8.4 Local mirror

A tiny subset of each dataset is mirrored locally to your laptop for dataloader development — you don't need the full 50 GB of MultiSubjects to iterate on a `Dataset.__getitem__` method.

**Location:** `data/mirror/` in the repo (gitignored).

**Contents:**
- 5 short clips (~10s each) from each dataset, copied from the Volume.
- 1 placeholder file for own footage (replaced when actual footage exists).
- Total size: ~5-10 GB.

**Sync script:** `scripts/sync_mirror.py` uses `modal volume get courtvision-data <relpath> data/mirror/<relpath>` to pull specific files. Run once after Volume is populated; re-run only if you change which clips you want locally.

**Path resolution:** `src/courtvision/paths.py` exposes `dataset_path(name)` which returns `/data/<name>` when running on Modal and `data/mirror/<name>` when running locally. Stage code never hardcodes either path.

### 8.5 Secrets

`.env.example`:
```
WANDB_API_KEY=
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
HF_TOKEN=                 # if needed for gated MMAction2 checkpoints
```

Inside Modal functions, the `wandb` key is mounted via `modal.Secret.from_name("wandb-secret")`. Document the `modal secret create wandb-secret WANDB_API_KEY=...` step in the README.

---

## 9. Stage-by-Stage Specifications

### 9.1 Stage 1 — Player Detection + ByteTrack

**Model:** YOLOv8m, COCO-pretrained, `classes=[0]` (person), `conf=0.4`, `iou=0.5`.
**Tracker:** ByteTrack via `ultralytics`' built-in `tracker='bytetrack.yaml'` with defaults; will switch to MMTracking if integration is bumpy.
**Input:** single video file (one camera, one session).
**Output:** `tracks_{session_id}_{camera_id}.json` per the schema above.
**Datasets used for eval:**
- **TrackID3x3** (primary) — provides track ID ground truth. Compute MOTA and IDF1.
- **MultiSubjects** (qualitative) — visual ID-stability check.
- **FineSports** (qualitative stress test) — NBA broadcast video with camera motion.

**Acceptance criteria:**
- Runs end-to-end on a 10-second MultiSubjects clip and emits a valid `TrackSequence` JSON.
- IDF1 ≥ 0.70 on TrackID3x3 (off-the-shelf YOLO+BT is well-documented to clear this bar on half-court basketball).
- Smoke test passes (tiny clip → schema-valid output).

### 9.2 Stage 2 — RTMPose-l Halpe-26

**Model:** RTMPose-l, Halpe-26 keypoints, MMPose `rtmpose-l_8xb256-420e_coco-256x192` config adapted for Halpe-26 (use MMPose's Halpe checkpoint).
**Mode:** top-down. Crops come from Stage 1 bboxes.
**Output:** `pose2d_{session_id}_{camera_id}.json`.
**Helper:** `halpe_to_coco17.py` exports the COCO-17 subset for Stage 5 classifier consumption.
**Datasets used for eval:**
- **TrackID3x3** — PCK@0.05 on the 10 keypoints they label (verify wrist is included before treating as the headline number).
- **MultiSubjects** — convert their BlazePose-33 labels to COCO-17 and compare against RTMPose-l's COCO-17 subset output.

**Acceptance criteria:**
- Runs on Stage 1's output and emits a valid `Pose2DSequence`.
- PCK@0.05 on TrackID3x3 wrists ≥ 0.85 (Phase 1 gate; below this, trigger Phase 2 ★ fine-tune — out of scope for sprint).
- Smoke test passes.

### 9.3 Stage 3 — WASB Ball Detection

**Model:** WASB, basketball-pretrained, off-the-shelf. Code via `nttcom/WASB-SBDT`.
**Output:** `ball2d_{session_id}_{camera_id}.json`.
**Datasets used for eval:**
- **U Arizona Roboflow basketball** (~9.8K images) — held-out slice for mAP@0.5 and mAP@0.5:0.95. Also run YOLOv8 (ayushpai) as the baseline for the milestone comparison.
- **WASB basketball subset** — release-window recall: hand-pick ~30 release moments, compute detection rate in ±15-frame windows around each.
- **MultiSubjects + FineSports** — qualitative checks on continuous video.

**Acceptance criteria:**
- Runs on a clip and emits a valid `Ball2DSequence`.
- mAP@0.5 ≥ 0.85 on U Arizona slice.
- Release-window recall ≥ 90% on WASB basketball subset (Phase 1 → Phase 2 gate per Decision 3).

### 9.4 Stage 4 — Triangulation (STUB)

**Status:** I/O contract only. Implementation deferred until own footage exists.
**Stub behavior:** `src/courtvision/stage4_triangulation/triangulate.py` exports a function `triangulate_session(session_id: str) -> Pose3DSequence` that raises `NotImplementedError("Stage 4 — implement in weeks 3-4")`.
**README spec:** record Decision 4 verbatim — Pose2Sim DLT for poses, `cv2.triangulatePoints` for ball, Butterworth 6 Hz smoothing, per-frame visibility logging, ≤5px reprojection error target.

### 9.5 Stage 5 — Shot Classifier (Two-Stream PoseConv3D + Ball MLP)

**This is the only trained stage. It is the critical-path item for the sprint.**

#### 9.5.1 Architecture

**Pose stream:**
- PoseConv3D, SlowOnly-R50 backbone, MMAction2 config `slowonly_r50_8xb16-u48-240e_k400-keypoint`.
- Input: [T=32 frames @ 60fps ≈ 0.53s, K=17 channels (COCO-17 subset of Halpe-26), H, W] as Gaussian heatmaps.
- Per-player crops. Only the target player's pose is rendered into their input (Decision 6).

**Ball-trajectory stream (added in Sprint Day TBD, training in week 3+):**
- 2-3 layer MLP, hidden dim 64.
- Inputs (per window): ball-z time series (length 32) + scalar summaries (max_z, time-to-peak, peak_v_z, ascent rate, z at endpoints).
- 3D ball positions from Stage 4 (until Stage 4 exists, use 2D ball-z proxy from the best-visibility camera — flag in code).

**Head:**
- Concatenate pose-stream features + ball-stream features.
- 4-class linear head: `dribble`, `layup`, `shoot`, `background`.

#### 9.5.2 4-Stage Transfer Pipeline (Path C primary)

| Stage | Data | Recipe | Trainable params | Sprint? |
|---|---|---|---|---|
| 1 — Kinetics-400 | (download only) | Load MMAction2 PoseConv3D checkpoint | — | ✅ |
| 2 — MultiSubjects | 6,144 clips, BlazePose-33→COCO-17 | Full fine-tune. SGD, lr 0.01, cosine schedule, ~30 epochs. Extend head 3→4 classes. | All | ✅ |
| 3 — FineSports | 10K NBA clips, continuous video → synthesized 32-frame windows | Full fine-tune. SGD, lr 0.005, ~20 epochs. All 4 classes get explicit supervision. | All | ✅ (Path A fallback if access fails) |
| 4 — Own footage LP-FT | ~100-200 labeled shots → synthesized windows | 4a: linear probe (freeze backbone, head only, lr 1e-3, 10-20 epochs). 4b: full fine-tune at lr 1e-5, early-stop on F1. | Head, then all | ❌ (week 3-4) |
| 5 — Ball MLP head | same data as 4 | Freeze backbone + pose head. Train ball MLP + new combined head. | Ball MLP + combined head | ❌ (week 3-4) |

#### 9.5.3 Sprint deliverables for Stage 5

1. **Model factory** (`model/poseconv3d.py`) — load Kinetics checkpoint, swap head to 4-class.
2. **Dataloaders** for MultiSubjects + FineSports, including BlazePose-33 → COCO-17 conversion and 32-frame window sampling.
3. **`train.py` Hydra entrypoint** — handles Stage 2 and Stage 3 training via config swap.
4. **Stage 2 training run kicked off** on Modal (A10G), W&B logging.
5. **Stage 3 training run kicked off** on Modal (A100-40GB), W&B logging.
6. **External benchmark eval scripts:**
   - MultiSubjects top-1 accuracy on their 3-class test split (post-Stage-2 checkpoint).
   - FineSports mAP@tIoU={0.3, 0.5, 0.7} on their test videos (post-Stage-3 checkpoint, Path C only).
7. **Sliding-window inference scaffold** (`infer.py`) — 32-frame windows @ 60fps, stride 4, ball-ascent gate, late fusion across cameras, NMS. Returns `ShotEventSequence`. Tested on a MultiSubjects clip; not yet calibrated to own footage.
8. **Ablation harness** — script that runs eval on Row 1 (post-Stage-2) and Row 2 (post-Stage-3) checkpoints; Rows 3, 4 stubbed.

#### 9.5.4 W&B tracking

- One W&B project: `courtvision`.
- One W&B run per training stage. Run names: `stage5-stage2-multisubjects-{date}`, `stage5-stage3-finesports-{date}`.
- Logged per run: loss curves, val accuracy (top-1, top-5), per-class F1, sample input visualizations (heatmap stack), config (auto-via Hydra+W&B integration), final checkpoint as W&B artifact.
- Tag runs with the transfer stage number for ablation comparison views.

#### 9.5.5 Acceptance criteria (sprint)

- Kinetics checkpoint loads and forward-passes on a 32-frame synthetic input.
- MultiSubjects dataloader returns valid tensors of shape `[B, 17, 32, H, W]`.
- Stage 2 training completes 1 epoch on Modal without error (smoke test for full pipeline).
- Stage 3 dataloader returns valid tensors on FineSports windows.
- External benchmark eval scripts run on a checkpoint (even an untrained one) without error and emit a JSON results file.
- Sliding-window inference runs on a 10-second clip and emits a valid `ShotEventSequence`.

### 9.6 Stages 6, 7, 8 — STUBS

**Stage 6 (release frame):** Two implementations planned — `rule_forward.py` (4-condition) and `rule_backward.py` (backward-trace-from-apex). README captures the rationale and the head-to-head eval plan added in our discussion. Both stubs raise `NotImplementedError`.

**Stage 7 (contest metric):** `measurements.py` exports 5 functions (`wrist_to_trajectory_distance`, `wrist_height_advantage`, `shooter_velocity_decomposed`, `floor_distance`, `composite`) all raising `NotImplementedError`. README captures the spec from Decision 9.

**Stage 8 (3D demo):** README only — choice of 3D viz framework (Plotly 3D / three.js / aitviewer) deferred.

---

## 10. Phased Sprint Plan — Build in 5 Parts, One at a Time

**This project is built in 5 sequential phases. Each phase has explicit START gate, scope, acceptance checks (machine-verified), a user test step (your qualitative/quantitative assessment), and STOP gate. Claude Code must NOT begin a phase until the user explicitly approves the prior phase's STOP gate.**

The phases are:

| Phase | What gets built | Time |
|---|---|---|
| **A** | Scaffold repo + schemas + Modal app skeleton + dataset download | Day 0 |
| **B** | Stage 1 — YOLOv8m + ByteTrack | Day 1 |
| **C** | Stage 2 — RTMPose-l Halpe-26 | Day 2 |
| **D** | Stage 3 — WASB ball detection | Day 3 |
| **E** | Stage 5 — PoseConv3D classifier (scaffold + pretraining runs) | Days 4-10 |

### Commit policy (applies to every phase)

- **Commit frequently inside a phase** — at minimum after every meaningful unit of work (a new module compiles + passes its smoke test; a config file is added and parses; a dataloader returns valid tensors). Aim for 3-8 commits per phase, not one giant commit at the end.
- **Mandatory commit at the end of every phase** with message format: `Phase X: <one-line summary>`.
- **Never skip the end-of-phase commit** even if the phase was short.
- **Never commit `.env`, model weights, dataset files, or `__pycache__`.** `.gitignore` enforces this; verify with `git status` before committing.
- After each commit, run `git log --oneline -5` and report the commit hash + message in the response so the user can verify.

---

### PHASE A — Scaffold repo + schemas + Modal app skeleton

**START gate (do NOT start until all true):**
- [ ] User has explicitly said "begin Phase A" or "start Phase A".
- [ ] Claude Code is in plan mode and has proposed the file-level plan, and the user has approved.

**Scope:**
1. Initialize repo: `git init`, `.gitignore` (Python defaults + `.env`, `data/`, `outputs/`, `*.pth`, `*.pt`, `wandb/`, `__pycache__/`), `pyproject.toml` (Python 3.11, name `courtvision`, all deps from Section 5.2), `.python-version=3.11`, `uv.lock` via `uv lock`.
2. Create the full directory tree from Section 6, including all `__init__.py` files and `.gitkeep` files for empty dirs.
3. Implement `src/courtvision/schemas.py` with every Pydantic model from Section 7.
4. Implement `src/courtvision/paths.py` with environment-aware path resolution (local mirror vs. Modal Volume per Section 8.4).
5. Implement `src/courtvision/io_utils.py` with schema-validated JSON read/write helpers.
6. Implement `src/courtvision/logging.py` (rich + wandb wrappers).
7. Implement `src/courtvision/modal_app.py` with the app + image + volumes + function signatures from Section 8. **Function bodies are `raise NotImplementedError("Phase B/C/D/E")` for Stages 1, 2, 3, 5.**
8. Implement `scripts/download_datasets.py` and `scripts/download_checkpoints.py` per Section 8.3 (one Modal function per dataset/checkpoint).
9. Implement Hydra config tree under `configs/` per Section 6. Every config file must at least parse validly with `OmegaConf.load()`.
10. Write smoke tests under `tests/` — `test_schemas.py` (round-trip every schema), and stub `test_stage{1,2,3,5}_smoke.py` files that import the stage module and assert `NotImplementedError` is raised for now.
11. Write `README.md` with: project context, setup steps (`uv sync` → `cp .env.example .env` → `modal token new` → `modal deploy` → `modal run scripts/download_datasets.py::download_<name>`), how to run each stage locally and on Modal, how to run tests.
12. Write `.env.example` per Section 8.5.
13. Set up `.pre-commit-config.yaml` with ruff + mypy hooks.
14. Write `docs/sprint_plan.md` derived from this Section 10.
15. Copy `decisions.md` and `CourtVision_PRD.md` into `docs/` for reference.

**Out of scope for Phase A:**
- Do NOT implement Stage 1/2/3/5 inference or training logic.
- Do NOT deploy the Modal app yet — just write the code.
- Do NOT download datasets yet — just write the download script.

**Automated acceptance checks (Claude Code runs these and reports results):**
- [ ] `uv sync` succeeds.
- [ ] `uv run pytest` is green (smoke tests pass — they verify imports and the `NotImplementedError` stubs).
- [ ] `uv run python -c "import courtvision; from courtvision.schemas import TrackSequence, Pose2DSequence, Ball2DSequence, Pose3DSequence, ShotEventSequence, ReleaseFrame, ContestMeasurement"` succeeds.
- [ ] `uv run python -c "from omegaconf import OmegaConf; import glob; [OmegaConf.load(p) for p in glob.glob('configs/**/*.yaml', recursive=True)]"` succeeds.
- [ ] `uv run ruff check .` is clean.
- [ ] `uv run mypy src/courtvision` is clean.
- [ ] All directories and files from PRD Section 6 exist (verify with `find . -type d` against the spec).
- [ ] `git status` shows a clean working tree after the final commit.

**USER TEST STEP (Lukas runs this — Phase A):**

After Claude Code reports all automated checks pass and commits, you do the following:

1. **Clone-and-build check:** open a fresh terminal in the repo, run `uv sync`, `uv run pytest`. Should be green.
2. **Schema sanity check:** open `src/courtvision/schemas.py` and verify field names + types match PRD Section 7. (No qualitative judgment needed beyond "does this match the contract?")
3. **Modal deploy:** run `modal deploy src/courtvision/modal_app.py`. Should succeed and show the app + functions in `modal app list`.
4. **Dataset download (this is the big one):** run `modal run scripts/download_datasets.py::download_trackid3x3` (smallest dataset first as a smoke test). Once that succeeds, run the remaining downloads in any order. After all downloads: `modal volume ls courtvision-data` should show all dataset directories.
5. **Checkpoint download:** run `modal run scripts/download_checkpoints.py::download_all`. After: `modal volume ls courtvision-artifacts/checkpoints` should show all 4 weight files.
6. **Local mirror:** run `python scripts/sync_mirror.py`. Verify `data/mirror/` populates with the small subsets.

**STOP gate (commit + report + halt):**
- [ ] All automated checks pass.
- [ ] Final commit made: `Phase A: scaffold repo + schemas + Modal app skeleton`.
- [ ] Claude Code reports: commit hash, summary of files created (count by directory), and any items deferred or flagged.
- [ ] **Claude Code STOPS and does not proceed to Phase B until user explicitly says "begin Phase B" or "start Phase B".**

---

### PHASE B — Stage 1: YOLOv8m + ByteTrack

**START gate:**
- [ ] Phase A STOP gate cleared and user has explicitly said "begin Phase B".
- [ ] User has confirmed datasets and checkpoints are on Modal Volumes (Phase A user test step 4-5).
- [ ] Claude Code is in plan mode and has proposed the file-level plan.

**Scope:**
1. Implement `src/courtvision/stage1_detection/yolo_bytetrack.py` — main inference function per Section 9.1. Public entrypoint: `def run_stage1(video_path: Path, config: dict) -> TrackSequence`.
2. Implement `src/courtvision/stage1_detection/eval.py` — MOTA / IDF1 computation against TrackID3x3 ground truth. Use `motmetrics` or `trackeval` library (add to deps if needed — flag and stop if so).
3. Fill in `modal_app.py::run_yolo_bytetrack` to call `run_stage1` inside the Modal function with proper Volume paths.
4. Implement `scripts/stage1_run.py` (Hydra entrypoint, runs locally or via `modal run`).
5. Implement `scripts/stage1_eval.py` (Hydra entrypoint, runs TrackID3x3 eval).
6. Update `configs/stage1_detection/yolov8m_bytetrack.yaml` and `configs/stage1_detection/eval.yaml` with real values (confidence 0.4, IoU 0.5, class 0, etc.).
7. Implement `tests/test_stage1_smoke.py` — runs Stage 1 on a 5-frame synthetic clip (fixture), validates output against `TrackSequence` schema.

**Out of scope for Phase B:**
- Do NOT touch Stage 2/3/5 code beyond what already exists from Phase A.
- Do NOT change any I/O schemas.
- Do NOT add new top-level dependencies without flagging and stopping.

**Automated acceptance checks:**
- [ ] `uv run pytest tests/test_stage1_smoke.py` passes.
- [ ] `uv run python scripts/stage1_run.py --help` shows Hydra help text.
- [ ] `uv run python scripts/stage1_eval.py --help` shows Hydra help text.
- [ ] `uv run ruff check src/courtvision/stage1_detection/ scripts/stage1_*.py` clean.
- [ ] `uv run mypy src/courtvision/stage1_detection/` clean.
- [ ] Local dry run on a single mirror clip succeeds: `uv run python scripts/stage1_run.py +input=data/mirror/multisubjects/<one_clip>.mp4` produces a schema-valid output JSON.

**USER TEST STEP (Lukas runs this — Phase B):**

You will assess Stage 1 quantitatively and qualitatively. Claude Code provides the scripts; you run them.

**Quantitative (Modal):**
1. `modal run scripts/stage1_run.py -- input=/data/trackid3x3/<test_split> session_id=trackid_test` — produces tracks JSONs on the artifacts Volume.
2. `modal run scripts/stage1_eval.py -- tracks_dir=/artifacts/stage1/trackid_test gt_dir=/data/trackid3x3/annotations` — outputs `eval_stage1_trackid3x3.json` with MOTA + IDF1.
3. Pull results locally: `modal volume get courtvision-artifacts eval_stage1_trackid3x3.json outputs/`.
4. **Your check:** open the JSON and confirm IDF1 ≥ 0.70. Report the number.

**Qualitative (Modal + local):**
5. `modal run scripts/stage1_run.py -- input=/data/multisubjects/<one_clip>.mp4 session_id=ms_demo` — produces tracks JSON.
6. Pull the original clip + tracks locally: `modal volume get courtvision-data multisubjects/<one_clip>.mp4 outputs/`; `modal volume get courtvision-artifacts stage1/ms_demo/ outputs/stage1_ms_demo/`.
7. Run a small visualization script (Claude Code provides it in `notebooks/01_stage1_visualize.ipynb`) that overlays bboxes + track IDs on the clip frames.
8. **Your check:** does each player keep the same track ID across frames? Are there spurious detections? Note any obvious failures.
9. Repeat on 1 FineSports clip (NBA broadcast — harder camera motion) as a stress test. Note whether tracking degrades.

**STOP gate:**
- [ ] All automated checks pass.
- [ ] User has confirmed quantitative IDF1 ≥ 0.70 OR has chosen to accept a lower number with documented rationale.
- [ ] User has confirmed qualitative assessment is acceptable.
- [ ] Final commit: `Phase B: Stage 1 — YOLO+ByteTrack`.
- [ ] Claude Code reports: commit hash, summary of changes, eval results JSON contents.
- [ ] **Claude Code STOPS and does not proceed to Phase C until user explicitly says "begin Phase C".**

---

### PHASE C — Stage 2: RTMPose-l Halpe-26

**START gate:**
- [ ] Phase B STOP gate cleared and user has explicitly said "begin Phase C".
- [ ] Claude Code is in plan mode and has proposed the file-level plan.

**Scope:**
1. Implement `src/courtvision/stage2_pose/rtmpose_inference.py` — top-down inference on YOLO crops. Public entrypoint: `def run_stage2(tracks: TrackSequence, video_path: Path, config: dict) -> Pose2DSequence`.
2. Implement `src/courtvision/stage2_pose/halpe_to_coco17.py` — keypoint subsetting helper. Public function: `def halpe26_to_coco17(pose: Pose2D) -> Pose2D` with `keypoint_format` updated.
3. Implement `src/courtvision/stage2_pose/eval.py` — wrist PCK@0.05 on TrackID3x3; PCK comparison against MultiSubjects BlazePose-33 → COCO-17 subset.
4. Fill in `modal_app.py::run_rtmpose`.
5. Implement `scripts/stage2_run.py` and `scripts/stage2_eval.py`.
6. Update `configs/stage2_pose/*.yaml`.
7. Implement `tests/test_stage2_smoke.py` — runs Stage 2 on synthetic crops (fixture), validates against `Pose2DSequence` schema and the COCO-17 conversion.
8. **Day 2 verification (from Risk Register and Section 13.1):** before running PCK eval, load TrackID3x3 annotation files and verify wrist keypoints are present. If not, switch PCK eval target to MultiSubjects-only and flag in a commit message.

**Out of scope:**
- Do NOT touch Stage 1/3/5 code.
- Do NOT change schemas.

**Automated acceptance checks:**
- [ ] `uv run pytest tests/test_stage2_smoke.py` passes.
- [ ] `uv run python scripts/stage2_run.py --help` works.
- [ ] `uv run ruff check src/courtvision/stage2_pose/ scripts/stage2_*.py` clean.
- [ ] `uv run mypy src/courtvision/stage2_pose/` clean.
- [ ] Halpe-26 → COCO-17 conversion test passes (`tests/test_stage2_smoke.py::test_halpe_to_coco17`).
- [ ] Local dry run on a single mirror clip + its Phase B tracks produces a schema-valid `Pose2DSequence`.

**USER TEST STEP (Lukas runs this — Phase C):**

**Quantitative (Modal):**
1. `modal run scripts/stage2_run.py -- input=/data/trackid3x3/<test_split> tracks=/artifacts/stage1/trackid_test session_id=trackid_test` — produces 2D pose JSONs.
2. `modal run scripts/stage2_eval.py -- poses_dir=/artifacts/stage2/trackid_test gt_dir=/data/trackid3x3/annotations` — outputs `eval_stage2_trackid3x3.json`.
3. Repeat for MultiSubjects: `modal run scripts/stage2_eval.py -- poses_dir=... gt_dir=/data/multisubjects/annotations subset=coco17` — outputs `eval_stage2_multisubjects.json`.
4. **Your check:** confirm wrist PCK@0.05 ≥ 0.85 on TrackID3x3 (if wrists are in their schema; otherwise fall back to MultiSubjects). Report the numbers per keypoint.

**Qualitative:**
5. Run `notebooks/02_stage2_visualize.ipynb` — overlays Halpe-26 skeletons on a MultiSubjects clip and a FineSports clip.
6. **Your check:** are wrists and elbows tracked smoothly across frames? Any obvious limb flips or missed keypoints during shot motion?

**STOP gate:**
- [ ] All automated checks pass.
- [ ] User confirmed quantitative wrist PCK target met (or documented acceptable deviation).
- [ ] User confirmed qualitative assessment acceptable.
- [ ] Final commit: `Phase C: Stage 2 — RTMPose-l Halpe-26`.
- [ ] Claude Code reports: commit hash, eval JSON contents, any flagged risks.
- [ ] **STOPS until user says "begin Phase D".**

---

### PHASE D — Stage 3: WASB ball detection

**START gate:**
- [ ] Phase C STOP gate cleared; user explicitly said "begin Phase D".
- [ ] Claude Code in plan mode with approved plan.

**Scope:**
1. Implement `src/courtvision/stage3_ball/wasb_inference.py` — wraps WASB inference, returns `Ball2DSequence`.
2. Implement `src/courtvision/stage3_ball/eval.py` — mAP@0.5 / mAP@0.5:0.95 on U Arizona slice; release-window recall on WASB basketball subset.
3. Fill in `modal_app.py::run_wasb`. **WASB integration risk (Section 13.3):** if `pip install git+...` fails in the shared image, build a dedicated image with WASB's exact deps and split the function into its own Modal app or function with a separate image. Flag and stop if this happens.
4. Implement `scripts/stage3_run.py` and `scripts/stage3_eval.py`.
5. Update `configs/stage3_ball/*.yaml`.
6. Implement `tests/test_stage3_smoke.py` — runs WASB on a synthetic 5-frame clip (fixture).

**Out of scope:**
- Do NOT touch Stage 1/2/5 code.

**Automated acceptance checks:**
- [ ] `uv run pytest tests/test_stage3_smoke.py` passes.
- [ ] `uv run python scripts/stage3_run.py --help` works.
- [ ] `uv run ruff check src/courtvision/stage3_ball/ scripts/stage3_*.py` clean.
- [ ] `uv run mypy src/courtvision/stage3_ball/` clean.
- [ ] Local dry run on a single mirror clip emits a schema-valid `Ball2DSequence`.

**USER TEST STEP (Lukas runs this — Phase D):**

**Quantitative (Modal):**
1. `modal run scripts/stage3_run.py -- input=/data/u_arizona session_id=u_arizona_eval` — produces ball JSONs on a held-out slice.
2. `modal run scripts/stage3_eval.py -- preds_dir=/artifacts/stage3/u_arizona_eval gt_dir=/data/u_arizona/annotations` — outputs `eval_stage3_u_arizona.json` with mAP@0.5 and mAP@0.5:0.95.
3. `modal run scripts/stage3_eval.py -- preds_dir=/artifacts/stage3/wasb_basketball gt_dir=/data/wasb_basketball/annotations mode=release_window_recall` — outputs `eval_stage3_wasb_recall.json`.
4. **Your check:**
   - mAP@0.5 ≥ 0.85 on U Arizona slice.
   - Release-window recall ≥ 90% on WASB basketball subset.
   - If recall < 90%, this triggers the Phase 2 ★ gate (Decision 3) — note it but do NOT pivot in this sprint; flag for week 3-4 decision.

**Qualitative:**
5. Run `notebooks/03_stage3_visualize.ipynb` — overlays ball detection on a MultiSubjects clip and a FineSports clip with a known shot.
6. **Your check:** is the ball tracked through the release motion? Lost during occlusion behind the shooter? Notes for the final report.

**STOP gate:**
- [ ] All automated checks pass.
- [ ] User confirmed quantitative gates (or documented acceptable miss with Phase 2 ★ trigger flagged).
- [ ] User confirmed qualitative assessment acceptable.
- [ ] Final commit: `Phase D: Stage 3 — WASB ball detection`.
- [ ] Claude Code reports: commit hash, eval JSON contents.
- [ ] **STOPS until user says "begin Phase E".**

---

### PHASE E — Stage 5: PoseConv3D classifier (scaffold + pretraining runs)

**This phase is significantly larger than B/C/D. It is broken into sub-phases E1-E5 with intermediate user-test checkpoints, not a single end-of-phase gate.**

**START gate:**
- [ ] Phase D STOP gate cleared; user explicitly said "begin Phase E".
- [ ] Claude Code in plan mode with approved plan for Sub-Phase E1.

#### E1 — Model factory + Kinetics checkpoint load

**Scope:**
1. Implement `src/courtvision/stage5_classifier/model/poseconv3d.py` — MMAction2 model factory. Public function: `def build_poseconv3d(num_classes: int = 4, pretrained_path: Path | None = None) -> nn.Module`.
2. Implement `src/courtvision/stage5_classifier/model/ball_mlp.py` — small MLP for ball-trajectory stream (used in week 3-4, but architecture defined now).
3. Implement `src/courtvision/stage5_classifier/model/two_stream.py` — combined two-stream model. Forward pass uses only pose stream until ball MLP is trained (week 3-4); ball stream returns zeros until then.
4. Commit after this sub-phase.

**Automated acceptance checks (E1):**
- [ ] `uv run python -c "from courtvision.stage5_classifier.model import build_poseconv3d; m = build_poseconv3d(num_classes=4, pretrained_path='/artifacts/checkpoints/poseconv3d_k400.pth'); import torch; x = torch.zeros(1, 17, 32, 56, 56); print(m(x).shape)"` produces shape `[1, 4]`.
- [ ] `uv run pytest tests/test_stage5_smoke.py::test_model_forward` passes.

**USER TEST STEP (E1):** Lukas runs the forward-pass command above on Modal (T4) and confirms it returns `[1, 4]` without NaN. Quick check, no qualitative judgment.

**Sub-phase commit:** `Phase E1: PoseConv3D model factory + Kinetics checkpoint load`.

#### E2 — MultiSubjects dataloader + train entrypoint

**Scope:**
1. Implement `src/courtvision/stage5_classifier/data/windowing.py` — shared 32-frame window sampling logic.
2. Implement `src/courtvision/stage5_classifier/data/multisubjects.py` — `MultiSubjectsDataset` returning `[B, 17, 32, H, W]` tensors. Includes BlazePose-33 → COCO-17 conversion and head extension to 4 classes (add `background` zero-pad).
3. Implement `src/courtvision/stage5_classifier/train.py` — Hydra-driven training entrypoint. Generic enough to handle both MultiSubjects and FineSports via config swap.
4. Wire W&B logging.
5. Update `configs/stage5_classifier/train_stage2_multisubjects.yaml`.
6. Verify 1 training iteration runs locally on a mirror subset (no Modal yet).

**Automated acceptance checks (E2):**
- [ ] `uv run pytest tests/test_stage5_smoke.py::test_multisubjects_dataloader` passes — returns tensor of shape `[B, 17, 32, H, W]`.
- [ ] `uv run python scripts/stage5_train_multisubjects.py +debug=true` runs 1 train step locally without error; loss is finite.

**USER TEST STEP (E2):** Lukas reviews the W&B run from the 1-step debug run, confirms loss is finite, config logged correctly, model gradients flow.

**Sub-phase commit:** `Phase E2: MultiSubjects dataloader + train entrypoint`.

#### E3 — Kick off MultiSubjects training (transfer-stage-2)

**Scope:**
1. Launch full MultiSubjects training on Modal A10G with `modal run scripts/stage5_train_multisubjects.py`.
2. Monitor via W&B. Expected wall-clock: ~12-24h.
3. On completion: save checkpoint to `/artifacts/checkpoints/stage5/stage2_multisubjects.pth`.
4. Implement `scripts/stage5_eval_external.py` — MultiSubjects top-1 accuracy on held-out split.
5. Run external benchmark eval on the Stage-2 checkpoint.

**Automated acceptance checks (E3):**
- [ ] Training run completes ~30 epochs or hits early-stop.
- [ ] W&B logs show monotonically decreasing training loss and stable val accuracy.
- [ ] Checkpoint exists on the Volume.
- [ ] External benchmark eval emits `eval_stage5_multisubjects.json`.

**USER TEST STEP (E3) — quantitative + qualitative:**
- **Quantitative:** open `eval_stage5_multisubjects.json` and compare top-1 accuracy against MultiSubjects' published numbers (in their paper). You expect to match or beat their pose-only baseline.
- **Qualitative:** in W&B, inspect train/val loss curves for stability. Inspect class-wise F1 — `shoot` and `layup` classes should have high F1; `dribble` may be weaker (acceptable).

**Sub-phase commit:** `Phase E3: MultiSubjects training + external benchmark (Row 1 of ablation)`.

#### E4 — FineSports dataloader + training (transfer-stage-3, Path C)

**Scope:**
1. Implement `src/courtvision/stage5_classifier/data/finesports.py` — synthesizes 32-frame windows from continuous video using temporal labels. Positives ±5-frame jitter; negatives from non-action portions labeled `background`.
2. Update `configs/stage5_classifier/train_stage3_finesports.yaml`.
3. Launch FineSports training on Modal A100-40GB. Initialize from Stage-2 checkpoint.
4. Run FineSports external benchmark eval: mAP@tIoU={0.3, 0.5, 0.7} on their test videos.
5. **Path A fallback (Risk Register §13.2):** if FineSports access is unavailable, skip this sub-phase, mark Row 2 ablation as N/A, and proceed directly to E5. Flag in commit.

**Automated acceptance checks (E4):**
- [ ] FineSports dataloader returns valid tensors.
- [ ] Training run completes (or stable in progress) and saves checkpoint.
- [ ] External benchmark eval emits `eval_stage5_finesports.json`.

**USER TEST STEP (E4):**
- **Quantitative:** compare mAP@tIoU values in `eval_stage5_finesports.json` against FineSports paper's published numbers.
- **Qualitative:** W&B loss curves stable; class-wise F1 reasonable across all 4 classes (background now actively supervised).

**Sub-phase commit:** `Phase E4: FineSports training + external benchmark (Row 2 of ablation)` (or `Phase E4: FineSports skipped — Path A fallback` if access fails).

#### E5 — Sliding-window inference scaffold + ablation harness

**Scope:**
1. Implement `src/courtvision/stage5_classifier/infer.py` — sliding window (32 frames, stride 4), ball-ascent gate (3D ball z > 3m within ±20 frames; uses 2D best-camera z-proxy until Stage 4 exists), late fusion across cameras at probability level, NMS on peak P(shoot)+P(layup). Returns `ShotEventSequence`.
2. Implement `src/courtvision/stage5_classifier/eval.py` — ablation harness that evaluates a list of checkpoints (Row 1, Row 2, plus stubs for Row 3, Row 4) on the same test set.
3. Verify inference runs on a MultiSubjects clip and emits valid `ShotEventSequence`.

**Out of scope (deferred to weeks 3-4):**
- Do NOT implement LP-FT on own footage (transfer-stage-4).
- Do NOT train the ball-trajectory MLP head (transfer-stage-5).
- Do NOT run Rows 3, 4 of the ablation (need own footage).

**Automated acceptance checks (E5):**
- [ ] `uv run pytest tests/test_stage5_smoke.py::test_sliding_window_inference` passes.
- [ ] `uv run python scripts/stage5_eval_external.py --ckpt=/artifacts/checkpoints/stage5/stage2_multisubjects.pth` produces results JSON.
- [ ] Sliding-window inference on a test clip produces a schema-valid `ShotEventSequence`.

**USER TEST STEP (E5) — final Phase E assessment:**

**Quantitative:**
1. Run sliding-window inference on a held-out MultiSubjects clip with known shot timestamps.
2. **Your check:** does the classifier peak `P(shoot)+P(layup)` align (within a few frames) with the known shot moments? Inspect the probability time-series plot.

**Qualitative:**
3. Run inference on a FineSports clip with known shots. Visualize the probability stream over time (notebook).
4. **Your check:** does the classifier fire on actual shots? Any obvious false positives on passes or rebounds? This is your first end-to-end signal of whether the architecture is going to work — even pre-LP-FT.

**FINAL STOP gate for Phase E (and the sprint):**
- [ ] All E1-E5 sub-phase checks pass.
- [ ] User has reviewed E3 and E4 quantitative results (MultiSubjects top-1, FineSports mAP if Path C).
- [ ] User has reviewed E5 qualitative inference on at least one MultiSubjects + one FineSports clip.
- [ ] All Row 1 and Row 2 (or Row 1 only if Path A) ablation evals committed to W&B.
- [ ] Final phase commit: `Phase E: Stage 5 — PoseConv3D scaffold + Stage-2/3 pretraining`.
- [ ] Claude Code reports: all commit hashes from E1-E5, full eval JSON paths, W&B run links.
- [ ] **Sprint complete. Claude Code STOPS until user provides week 3-4 instructions.**

---

### Risk register

| Risk | Mitigation |
|---|---|
| MMAction2/MMPose/mmcv version hell | Pin exact patch versions in `uv.lock`. Use the Modal image as canonical; replicate locally only as needed. If pinning fails, fall back to MMAction2's official Docker image as the Modal base. |
| FineSports data access falls through | Switch to Path A (3-stage transfer, no FineSports). Already specified in Decision 7. |
| Modal GPU queue wait | A10G and T4 have near-instant availability; A100 may queue. Schedule FineSports training to start by Day 9 end-of-day. |
| WASB integration friction | Has its own conda env in its repo. If `pip install git+...` fails, fall back to a separate Modal function with a custom image just for WASB; pass data via the shared Volume. |
| TrackID3x3 keypoint schema doesn't include wrists | Verify on Day 2 first thing by loading their annotation files. If wrist isn't labeled, swap PCK eval to MultiSubjects-only. |

### Budget tracking

| Item | Est. cost |
|---|---|
| Stages 1-3 inference (T4, sporadic) | ~$5 |
| MultiSubjects training (A10G, ~24h) | ~$30 |
| FineSports training (A100-40GB, ~24h) | ~$70 |
| External benchmark eval (T4, ~4h) | ~$3 |
| Debugging re-runs (mixed) | ~$30 |
| **Buffer** | ~$50 |
| **Total estimated** | **~$190** |

Within $200 budget. Wall-clock-optimized choices (A100 over A10G for FineSports) cost ~$40 extra but save ~24h.

---

## 11. Rolled-up Sprint Acceptance Criteria

This is the single checklist mirroring Section 10's per-phase STOP gates. By the end of Phase E, all of the following must be true.

- [ ] **Phase A:** Repo scaffolded per Section 6; `uv sync` works; `pytest` is green; Modal app deployed; both Volumes populated with datasets + pretrained checkpoints; local mirror synced.
- [ ] **Phase B (Stage 1):** Produces valid `TrackSequence` JSONs; IDF1 ≥ 0.70 on TrackID3x3 (or documented acceptable deviation); user-confirmed qualitative pass on MultiSubjects + FineSports.
- [ ] **Phase C (Stage 2):** Produces valid `Pose2DSequence` JSONs; wrist PCK@0.05 ≥ 0.85 on TrackID3x3 (or MultiSubjects fallback per Risk Register §13.1); user-confirmed qualitative pass.
- [ ] **Phase D (Stage 3):** Produces valid `Ball2DSequence` JSONs; mAP@0.5 ≥ 0.85 on U Arizona; release-window recall ≥ 90% on WASB basketball subset (or Phase 2 ★ trigger flagged per Decision 3).
- [ ] **Phase E (Stage 5):**
  - E1: Kinetics checkpoint loads, forward-pass on synthetic input returns shape `[1, 4]`.
  - E2: MultiSubjects dataloader returns valid tensors; train entrypoint runs 1 step locally without error.
  - E3: MultiSubjects training completed (or in stable progress); MultiSubjects external benchmark JSON emitted; Row 1 ablation checkpoint saved.
  - E4: FineSports training completed (or in stable progress, or Path A fallback documented); FineSports external benchmark JSON emitted; Row 2 ablation checkpoint saved.
  - E5: Sliding-window inference produces valid `ShotEventSequence` on a test clip; ablation harness runs Row 1 + Row 2 evals successfully; user-confirmed qualitative inference plot on MultiSubjects + FineSports.
- [ ] W&B project shows all training runs with logged metrics.
- [ ] All schemas in Section 7 are implemented and validated on stage outputs.
- [ ] Stages 4, 6, 7, 8 have stub modules + README specs (delivered in Phase A).
- [ ] Every phase ended with the mandatory commit per Section 15.5.

---

## 12. Style and Guardrails (cross-cutting rules)

These rules apply across every phase. Per-phase scope and stop gates live in Section 10; the Execution Protocol lives in Section 15.

### 12.1 Code style

- Type-annotate every public function. `mypy --strict` clean on `src/courtvision/`.
- Use `pydantic` v2 for all I/O schemas. No raw `dict` returns from stage entrypoints.
- Use `rich.logging` + `wandb` for logs. No `print` in library code.
- All file paths through `courtvision.paths` — never hardcode `/data`, `/artifacts`, or `data/mirror/`.
- Every stage's main entrypoint is callable both as a Python function (for testing + local) and via a Modal function (for production).
- One concept per file. If a module exceeds ~300 lines, split.
- Docstrings on every public function — Google style.

### 12.2 What NOT to do (in any phase)

- Do not implement Stages 4, 6, 7, 8 logic — only stubs + README specs.
- Do not implement Stage 5 transfer-stage-4 (LP-FT) or transfer-stage-5 (ball MLP head training) — those wait for own footage in weeks 3-4.
- Do not add swappable backends — locked tools only (Section 5.2).
- Do not commit `.env`, dataset files, or model checkpoints.
- Do not invent new dependencies. If something is missing, flag it and STOP — wait for explicit user approval before adding a dep.
- Do not change any decision from `decisions.md` without flagging it and STOPping.
- Do not deploy to Modal or burn GPU hours on training without explicit user instruction (e.g., E3 and E4 sub-phases authorize specific training runs; nothing else).
- Do not begin a new phase until the prior phase's STOP gate has been cleared by explicit user approval.

---

## 13. Open Items (to revisit during the sprint)

1. **TrackID3x3 keypoint schema** — confirm wrist is in their 10 keypoints. Verify at start of Phase C.
2. **FineSports access** — confirm data access during Phase A (download step); if blocked, switch to Path A and skip sub-phase E4.
3. **WASB integration** — confirm `pip install git+...` works in the Modal image during Phase D; otherwise build separate WASB image (flag and stop per Section 15.6).
4. **MMAction2 PoseConv3D head extension 3→4 classes** — confirm the canonical MMAction2 pattern during sub-phase E1; may need a custom head module.
5. **License** — defer choice until weeks 3-4 (MIT or Apache 2.0 likely).
6. **W&B availability** — if W&B is unavailable, fall back to TensorBoard. Adds complexity to ablation comparison; not preferred. Resolve in Phase A.

---

## 14. References

- `decisions.md` (Milestone 2 working doc) — source of truth for all technical choices.
- `CourtVision_Milestone2.pdf` (Milestone 2 deck) — visual summary.
- Duan et al. 2022 — PoseConv3D (CVPR).
- Kumar et al. 2022 — LP-FT (ICLR).
- Daly-Grafstein & Bornn 2020 — NBA contest baseline.
- Mikic et al. 2024 — basketball release biomechanics under defense.
- Pagnon et al. — Pose2Sim.
- Yamada et al. — TrackID3x3 dataset.
- Xu et al. — MultiSubjects + FineSports datasets.

---

## 15. Execution Protocol (for Claude Code)

This section governs *how* Claude Code executes the work specified in Sections 9 and 10. It is binding.

### 15.1 The 5-phase rule

The project is built in 5 sequential phases — A (scaffold), B (Stage 1), C (Stage 2), D (Stage 3), E (Stage 5). Phases are executed strictly in order, one at a time. Phase E itself is broken into sub-phases E1-E5 with intermediate user-test checkpoints.

### 15.2 Plan mode before every phase

Before starting **any** phase, Claude Code enters plan mode (`/plan` or equivalent), reads the relevant PRD sections, and proposes a file-level implementation plan: which files will be created, modified, or deleted; which deps (if any) need to be added; which automated checks will be run. **Claude Code must wait for the user's explicit plan approval before writing any code for the phase.** Sub-phases of E follow the same rule (plan E1, get approval, execute, stop; plan E2, get approval, etc.).

### 15.3 START gate, STOP gate

Every phase has explicit START and STOP gates in Section 10. Claude Code must not start a phase until its START gate is true. Claude Code must not proceed past a phase until its STOP gate is cleared by explicit user approval (the user says "begin Phase X" or "proceed to Phase X").

If the user has not yet provided that explicit signal, Claude Code waits. It does not infer continuation from the user's other messages, even if the user is asking related questions.

### 15.4 Acceptance checks and the user test step

Every phase has two layers of acceptance:
1. **Automated checks** — Claude Code runs these and reports results. If any fails, Claude Code STOPS immediately and reports the failure. No partial-success advances to the user test step.
2. **User test step** — the user runs the steps documented in Section 10's USER TEST STEP for that phase. The user is the gate. Claude Code does not run the user test step itself; it provides scripts and notebooks that make the user's run easy.

### 15.5 Commit policy

- **Frequent commits inside a phase** — at least 3-8 per phase, each at a meaningful unit of work (new module + passing smoke test; new config + parses; new dataloader + returns valid tensors). Never bundle unrelated changes into one commit.
- **Mandatory commit at the end of every phase**, message format: `Phase X: <one-line summary>` or `Phase EN: <one-line summary>` for sub-phases.
- **No commit ever contains** `.env`, dataset files, or model checkpoints — `.gitignore` enforces this; verify with `git status` before each commit.
- **Always run `git log --oneline -5` after committing** and include the output in the response so the user can verify.

### 15.6 Stop-on-failure rule

Claude Code STOPS immediately and asks for user input in any of the following cases:
- An automated acceptance check fails.
- A dependency that is not already in `pyproject.toml` is needed.
- A decision in `decisions.md` would need to change for the phase to succeed.
- A Modal training run would burn more than ~$10 of GPU time and has not been explicitly authorized for this phase.
- An I/O schema in Section 7 would need to change.
- Phase scope creep — a task in the phase appears to require code outside that phase's named scope.

In all of these cases, Claude Code reports the issue and waits for instructions. It does NOT attempt to work around the issue silently.

### 15.7 GPU-burn authorization

Claude Code is authorized to spend Modal GPU credit only for:
- Phase A: dataset downloads (CPU-only) and checkpoint downloads (CPU-only).
- Phase B/C/D user test steps: short Modal inference runs (estimated <$2 each) on T4 GPUs.
- Phase E3: the MultiSubjects training run on A10G (~$30 estimated).
- Phase E4: the FineSports training run on A100-40GB (~$70 estimated).
- Phase E user test step: short inference runs on T4 (estimated <$5 total).

Any GPU burn outside this list requires explicit user approval before launch.

### 15.8 Context discipline

Across the 5 phases, Claude Code may be restarted with fresh context (via `/clear` or new session). When starting a new session for a later phase, Claude Code should:
- Re-read `CourtVision_PRD.md`, especially Sections 7 (schemas), 10 (phases), and 15 (this protocol).
- Re-read `decisions.md` for any decision-context not in the PRD.
- Run `git log --oneline -20` to understand recent work.
- Confirm the prior phase's STOP gate was cleared before assuming it was.

### 15.9 Reporting format at end of each phase

At every phase STOP, Claude Code's report includes:
1. List of commits made in the phase (hash + message, from `git log`).
2. List of files created / modified (one line each).
3. Automated check results (each check + pass/fail + key numeric outputs).
4. Any items flagged for user attention (deps requested, schema concerns, deferred decisions).
5. The exact user test step commands to run.
6. An explicit "STOPPING — awaiting user signal to begin Phase Y" closer.

---

**End of PRD.**

This document is sufficient for Claude Code to (a) scaffold the entire repository in one pass and (b) implement Stages 1, 2, 3, and 5-scaffolding over the 2-week sprint. Stages 4, 6, 7, 8 are interface-stable so the sprint code won't need to be refactored when those stages are implemented in weeks 3-4. Execution is governed by the 5-phase protocol in Section 15.
