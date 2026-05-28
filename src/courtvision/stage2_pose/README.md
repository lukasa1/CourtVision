# Stage 2 — RTMPose-l Halpe-26 (2D pose)

Top-down 2D pose on the player crops from Stage 1. One `Pose2DSequence` JSON
per (session, camera). See PRD §9.2.

## Public API

```python
from courtvision.stage2_pose.rtmpose_inference import run_stage2
from courtvision.stage2_pose.halpe_to_coco17 import halpe26_to_coco17
from courtvision.stage2_pose.eval import evaluate_trackid3x3, evaluate_multisubjects
```

- **`run_stage2(tracks, video_path, config) -> Pose2DSequence`** — RTMPose-l
  top-down inference. For each frame with tracks, runs `inference_topdown` on
  that frame's bboxes (one instance per track) and records a 26-keypoint
  Halpe pose per track. `mmpose` is imported lazily (Modal-only — see F1/F17).
- **`halpe26_to_coco17(pose) -> Pose2D`** — the COCO-17 subset (first 17 Halpe
  indices) for the Stage 5 classifier; the geometric pipeline keeps full Halpe-26.
- **`evaluate_trackid3x3(poses_dir, gt_dir) -> dict`** — wrist PCK@0.05 (primary gate).
- **`evaluate_multisubjects(poses_dir, gt_dir) -> dict`** — COCO-17 PCK (secondary).

## Locked knobs (PRD §9.2)

| Knob | Value |
|---|---|
| Model | RTMPose-l, Halpe-26 head (`rtmpose-l_simcc-body7…halpe26`) |
| Mode | top-down on Stage 1 bboxes |
| Keypoints | 26 (Halpe-26); COCO-17 subset for the classifier |
| Checkpoint | `/artifacts/checkpoints/rtmpose_l_halpe26.pth` |
| Input size | 192×256 (W×H) |

The model **config** is resolved at runtime from the installed mmpose's
`.mim/configs` tree — it must be the *halpe26* (26-kpt) config, not the
coco-256x192 (17-kpt) one. Override with `config['model']['cfg_path']` if a
given mmpose release names it differently. See `_resolve_model_config`.

## Where it runs

Stage 2 inference is **Modal-only** — the OpenMMLab stack doesn't build on
macOS arm64 (F1). Local dev runs the pure-Python parts (eval, conversion,
tests); inference + the schema-valid dry run happen on Modal T4 (F17).

```bash
# Inference (Modal T4):
uv run modal run scripts/stage2_run.py::run \
    --video-relpath trackid3x3/videos/Indoor/raw/<clip>.mp4 \
    --tracks-relpath stage1/<session>/tracks_<session>_<cam>.json

# Eval (Modal CPU):
uv run modal run scripts/stage2_eval.py::run \
    --poses-relpath stage2/<session> \
    --gt-relpath trackid3x3/ground_truth/Indoor/pose
```

## Evaluation — wrist PCK@0.05 (PRD §9.2)

**Gate: wrist PCK@0.05 ≥ 0.85 on TrackID3x3** (the deployment domain). Below
it, the PRD escalates to a Phase 2 ★ RTMPose fine-tune (out of sprint scope).

TrackID3x3 pose GT (verified — F18) is per-frame, **jersey-colour-keyed**
(not track-id), dense on the Indoor clips, 10 keypoints with wrists at GT
index 5 (R) / 6 (L); predictions are Halpe with wrists at 10 (R) / 9 (L).
The eval therefore:
1. **Spatially matches** predicted players to GT players per frame (Hungarian
   on keypoint centroids, capped at half the court spread).
2. Scores wrists with a **side-agnostic 2×2 min-cost assignment** — measures
   wrist *localisation* regardless of L/R label (L/R is ambiguous ~10-20% of
   the time when hands meet at the body). A strict same-side `wrist_pck_sided`
   is reported alongside.
3. Normalises the PCK threshold by the GT skeleton's keypoint bbox diagonal
   (the GT JSON carries no person bbox).

**Caveat:** TrackID3x3 pose labels are the dataset's baseline-estimator output,
not hand truth, so the number is *agreement with their baseline*, in-domain —
read it together with the qualitative overlay review (F18).

MultiSubjects is a secondary, single-player COCO-17 comparison
(BlazePose-33 → COCO-17); its GT-coordinate scale is assumed normalised and
**needs confirmation on the first Modal run** (F18).

## Tests

`tests/test_stage2_smoke.py` — Halpe→COCO17 + BlazePose33→COCO17 maps,
side-agnostic vs strict wrist scoring, PCK threshold boundary, player matching
regardless of order, an **end-to-end run against the real example GT** (perfect
synthetic prediction → wrist_pck ≈ 1.0), and a `Pose2DSequence` round-trip.
The real-inference test is `importorskip("mmpose")`-guarded. `uv run pytest
tests/test_stage2_smoke.py`.
