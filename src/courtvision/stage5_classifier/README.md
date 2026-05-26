# Stage 5 — Two-Stream Shot Classifier (PoseConv3D + Ball MLP)

**Status:** scaffold + Stage-2/3 pretraining in sprint scope; LP-FT (transfer-stage-4) and ball-MLP-head training (transfer-stage-5) deferred to weeks 3-4 after own footage exists.

## Architecture (PRD §9.5.1)

**Pose stream:** PoseConv3D SlowOnly-R50, MMAction2 config `slowonly_r50_8xb16-u48-240e_k400-keypoint`. Input `[T=32 @60fps, K=17 (COCO-17 subset), H, W]` as Gaussian heatmaps. Per-player crops; only the target player's pose rendered (Decision 6).

**Ball stream:** 2-3 layer MLP, hidden dim 64. Input: ball-z time series (len 32) + scalar summaries (max_z, time-to-peak, peak_v_z, ascent rate, endpoint zs). Until Stage 4 exists, uses 2D best-camera z-proxy (flagged in code).

**Head:** concatenate pose + ball features → 4-class linear head: `dribble`, `layup`, `shoot`, `background`.

## 4-stage transfer pipeline (Path C primary; Path A is fallback if FineSports inaccessible)

| Stage | Data | Recipe | Sprint? |
|---|---|---|---|
| 1 | Kinetics-400 | Load MMAction2 checkpoint | ✅ E1 |
| 2 | MultiSubjects (6,144 clips) | Full fine-tune, SGD lr 0.01, cosine, ~30 epochs. Extend head 3→4 | ✅ E3 |
| 3 | FineSports (10K NBA) | Full fine-tune, lr 0.005, ~20 epochs | ✅ E4 (Path C) |
| 4 | Own footage (~100-200 shots) | LP-FT: 4a freeze backbone, head only, lr 1e-3, 10-20 ep. 4b full FT lr 1e-5, early-stop on F1 | ❌ weeks 3-4 |
| 5 | Same data as 4 | Ball MLP + new combined head; freeze backbone + pose head | ❌ weeks 3-4 |

## Sprint deliverables (Phase E)

- `model/poseconv3d.py` — load Kinetics ckpt, swap head to 4-class
- `data/multisubjects.py`, `data/finesports.py` — dataloaders with BlazePose-33 → COCO-17 conversion and 32-frame windowing
- `train.py` — Hydra entrypoint for transfer-stage 2 and 3 via config swap
- W&B-tracked Stage-2 (A10G) and Stage-3 (A100-40GB) training runs
- External benchmark eval: MultiSubjects top-1, FineSports mAP@tIoU
- `infer.py` — sliding-window inference with ball-ascent gate + late fusion + NMS
- Ablation harness covering Row 1 (post-MS) and Row 2 (post-FS); Rows 3, 4 stubbed
