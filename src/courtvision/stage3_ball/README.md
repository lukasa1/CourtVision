# Stage 3 — WASB Ball Detection

**Status:** stub (Phase A). Real implementation: Phase D.

**Model:** WASB (BMVC 2023), basketball-pretrained, off-the-shelf.
**Output:** `ball2d_{session_id}_{camera_id}.json` matching `Ball2DSequence` (§7.3).
**Eval:** mAP@0.5 / mAP@0.5:0.95 on U Arizona Roboflow slice; release-window recall on WASB basketball subset.
**Acceptance:** mAP@0.5 ≥ 0.85 on U Arizona; release-window recall ≥ 90% (else triggers Phase 2 ★ per Decision 3).
**Integration risk:** WASB is git-only (`nttcom/WASB-SBDT`). If `pip install git+...` fails in the shared Modal image, build a dedicated WASB image (PRD §13 risk #3).
