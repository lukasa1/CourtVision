# Sprint Plan (Weeks 1-2)

Distilled from PRD §10 and §15. The PRD is the source of truth; this file is a quick-reference index.

## Phases

| Phase | Scope | Time |
|---|---|---|
| **A** | Scaffold repo, schemas, Modal app skeleton, dataset download scripts, stub stage modules | Day 0 |
| **B** | Stage 1 — YOLOv8m + ByteTrack inference + MOTA/IDF1 eval on TrackID3x3 | Day 1 |
| **C** | Stage 2 — RTMPose-l Halpe-26 inference + wrist PCK eval | Day 2 |
| **D** | Stage 3 — WASB ball detection + mAP eval on DeepSportradar (primary) + Roboflow basketball (secondary); WASB-basketball recall deferred (F11) | Day 3 |
| **E** | Stage 5 — PoseConv3D classifier scaffold + transfer-stage 2 (MultiSubjects, A10G) + transfer-stage 3 (FineSports, A100-40GB) + sliding-window inference + ablation harness | Days 4-10 |

Phase E is sub-phased E1-E5 — see PRD §10 Phase E.

**Status:** Phase A ✅ · Phase B ✅ (closed 2026-05-27 — IDF1 0.785 ± 0.083 on TrackID3x3 test, gate ≥ 0.70 met; persistence layers A1/B1/C1 landed, A2/A3/B2/C2/D1/D2 deferred per flags.md F15) · Phase C ✅ (closed 2026-05-28 — RTMPose-l Halpe-26 via rtmlib/ONNX, PRD §5.2 engine deviation; wrist PCK 0.655 on 13 TrackID3x3 clips accepted via documented deviation + qualitative review, see decisions.md Decision 2 / flags.md F20–F21) · Phases D–E pending.

## Phase boundaries (PRD §15.3)

Phases run strictly in order. Claude Code does **not** advance past a STOP gate until the user explicitly says "begin Phase X". Sub-phases of E follow the same rule.

## Commit policy (PRD §15.5)

- 3-8 commits per phase, each at a meaningful unit of work.
- Mandatory end-of-phase commit: `Phase X: <one-line summary>`.
- Never commit `.env`, dataset files, model weights, or `__pycache__`. `.gitignore` enforces this.
- After each commit, run `git log --oneline -5` and report the hash.

## Stop-on-failure rule (PRD §15.6)

Claude Code STOPS immediately and asks for input if:

- An automated acceptance check fails.
- A dependency not in `pyproject.toml` is needed.
- A decision in `decisions.md` would need to change.
- A Modal training run would burn > ~$10 of un-authorized GPU.
- An I/O schema in PRD §7 would need to change.
- The phase requires code outside its named scope.

## GPU-burn authorization (PRD §15.7)

| Allowed | Notes |
|---|---|
| Phase A: CPU-only downloads | datasets + checkpoints |
| Phase B/C/D user tests | short T4 inference (< $2 each) |
| Phase E3 | MultiSubjects training on A10G (~$30) |
| Phase E4 | FineSports training on A100-40GB (~$70) |
| Phase E5 user tests | short T4 inference (< $5 total) |

Anything else needs explicit user approval.

## Acceptance gates per phase (cross-references to PRD §10)

- **A:** `uv sync` works; `pytest` green; all configs parse; ruff + mypy clean.
- **B:** valid `TrackSequence` JSONs; IDF1 ≥ 0.70 on TrackID3x3.
- **C:** valid `Pose2DSequence` JSONs; wrist PCK@0.05 ≥ 0.85 (or MultiSubjects fallback).
- **D:** valid `Ball2DSequence` JSONs; mAP@0.5 ≥ 0.85 on DeepSportradar Basketball Instants test split (primary; threshold may need calibration). Secondary mAP on Roboflow basketball. Release-window recall on WASB basketball deferred (F11).
- **E1-E5:** see PRD §10 sub-phase tables; final gate requires user-confirmed E5 qualitative inference on MultiSubjects + FineSports.

## Risk register (PRD §13)

| Risk | Mitigation |
|---|---|
| MMAction2/MMPose/mmcv version hell | Pin in `uv.lock`; Modal image is canonical; openmm stack is Modal-only (Phase A applied this). |
| FineSports access falls through | Switch to Path A (3-stage transfer; skip transfer-stage-3). |
| Modal A100 queue wait | Schedule FineSports training to start by Day 9 EOD. |
| WASB integration friction | Has its own conda env; if pip install fails in shared image, use a dedicated WASB Modal function/image. |
| TrackID3x3 keypoint schema doesn't include wrists | Verify Day 2; if missing, swap PCK eval to MultiSubjects-only. |
