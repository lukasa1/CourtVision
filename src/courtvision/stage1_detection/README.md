# Stage 1 — Player Detection + ByteTrack

**Status:** stub (Phase A). Real implementation: Phase B.

**Model:** YOLOv8m (COCO-pretrained, person class only, `conf=0.4`, `iou=0.5`).
**Tracker:** ByteTrack via ultralytics' built-in.
**Input:** single-camera video file.
**Output:** `tracks_{session_id}_{camera_id}.json` matching `TrackSequence` schema (§7.1).
**Eval datasets:** TrackID3x3 (primary, MOTA + IDF1), MultiSubjects (qualitative), FineSports (stress test).
**Acceptance:** IDF1 ≥ 0.70 on TrackID3x3 (PRD §9.1).
