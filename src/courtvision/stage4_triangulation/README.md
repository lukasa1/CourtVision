# Stage 4 — Multi-View Triangulation

**Status:** stub — full implementation deferred to weeks 3-4 (requires own footage + calibration).

**Spec (Decision 4):**
- **Pose:** Pose2Sim DLT triangulation across 3 cameras.
- **Ball:** `cv2.triangulatePoints` (simpler than DLT, fine for a point-target).
- **Smoothing:** Butterworth 6 Hz low-pass on 3D pose trajectories (skating reduction).
- **Visibility:** log per-frame which cameras contributed (Halpe-26 keypoints can be visible in 0/2/3 cameras).
- **Calibration:**
  - Per-phone chessboard intrinsics (one-time).
  - Per-session court-landmark extrinsics (rerun any time the camera moves).
  - AE/AF lock per session.
- **Target:** ≤5 px reprojection error.

**I/O contract (PRD §7.4):** writes `pose3d_{session_id}.json` and `ball3d_{session_id}.json`. Schemas: `Pose3DSequence`, `Ball3DSequence`. Per-frame visibility logged via `Keypoint3D.n_cameras`.

**Risk:** triangulation quality depends entirely on calibration accuracy. Mitigation: redo extrinsics if any camera shifts between sessions.
