# CourtVision

Multi-view 3D skeletal tracking + shot-contest measurement for half-court 2-on-2 basketball. Stanford CS231N, Spring 2026.

The headline novel metric is **3D defender-wrist distance to the actual ball trajectory** at shot release — a refinement of NBA SportVU's floor-distance convention enabled by 3D hand tracking from 3 synchronized iPhone views.

See `docs/prd.md` for the full PRD (definitive source for architecture, schemas, and execution protocol) and `docs/sprint_plan.md` for the 5-phase build plan.

## Setup

```bash
# 1. Sync deps. Python 3.11 via uv.
uv sync --extra dev

# 2. Local env — fill in WANDB / Modal / HF tokens.
cp .env.example .env

# 3. Modal auth + secret.
modal token new
modal secret create wandb-secret WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d= -f2)

# 4. Pre-commit (optional but recommended).
uv run pre-commit install

# 5. Run tests.
uv run pytest
```

## Modal infrastructure

Single Modal app, one function per stage. Two persistent Volumes (`courtvision-data`, `courtvision-artifacts`) hold datasets and checkpoints respectively.

```bash
# Deploy.
modal deploy src/courtvision/modal_app.py

# One-time downloads (runs on Modal, not locally).
modal run scripts/download_datasets.py::download_trackid3x3
modal run scripts/download_datasets.py::download_multisubjects
modal run scripts/download_datasets.py::download_finesports
modal run scripts/download_datasets.py::download_u_arizona
modal run scripts/download_datasets.py::download_wasb_basketball
modal run scripts/download_checkpoints.py::download_all

# Pull a small local mirror for dataloader iteration.
uv run python scripts/sync_mirror.py
```

## Running a stage

Every stage has a Hydra-driven runner. Local example (Stage 1, after Phase B is implemented):

```bash
uv run python scripts/stage1_run.py input=data/mirror/multisubjects/clip_001.mp4 output=outputs/tracks_demo.json
```

Modal example:

```bash
modal run scripts/stage1_run.py -- input=/data/trackid3x3/test session_id=trackid_test
```

## Repository layout

```
src/courtvision/        package code (cross-stage primitives + stageN_* subpackages)
configs/                Hydra config tree (root + data + per-stage + modal)
scripts/                CLI entrypoints (Hydra-decorated; wrap Modal functions)
tests/                  pytest suite (schema round-trips + per-stage smoke)
notebooks/              ad-hoc viz (added Phase B+)
docs/                   PRD, decisions, sprint plan
data/                   local mirror of datasets (gitignored)
outputs/                local mirror of inference outputs (gitignored)
modal_workspace/        Modal-only mount points (gitignored)
```

## Execution protocol

The PRD §15 binds Claude Code to a strict 5-phase sequential build (A → B → C → D → E), with explicit START/STOP gates between phases. See `docs/sprint_plan.md` for the phase table and acceptance criteria.

| Phase | What gets built |
|---|---|
| A | Scaffold + schemas + Modal app skeleton |
| B | Stage 1 — YOLOv8m + ByteTrack |
| C | Stage 2 — RTMPose-l Halpe-26 |
| D | Stage 3 — WASB ball detection |
| E | Stage 5 — PoseConv3D classifier (sub-phases E1-E5) |

Stages 4, 6, 7, 8 stay as I/O-contracted stubs through the sprint; their full implementations land in weeks 3-4 with own footage.

## Status

Phase A complete. Phase B not started.
