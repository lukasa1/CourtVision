# Stage 2 — RTMPose-l Halpe-26

**Status:** stub (Phase A). Real implementation: Phase C.

**Model:** RTMPose-l, Halpe-26 keypoints, top-down on Stage 1 crops.
**Output:** `pose2d_{session_id}_{camera_id}.json` matching `Pose2DSequence` (§7.2).
**Helper:** `halpe_to_coco17.py` exports the COCO-17 subset for Stage 5.
**Eval:** wrist PCK@0.05 on TrackID3x3 (verify wrists are labeled first); COCO-17 comparison on MultiSubjects.
**Acceptance:** wrist PCK@0.05 ≥ 0.85 (Phase 1 gate; below this triggers Phase 2 ★ fine-tune, out of sprint scope).
