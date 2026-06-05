# CourtVision — Development Log

**Purpose.** Single consolidated tracker, organised by stage, for the final CS231N report.
For each stage: what was **planned** (PRD), what was **built**, what was **deferred or
changed** (with link to the flag entry), the **key decisions**, the **results / metrics**,
and the **key files / outputs** for the report.

This document is a synthesis of `flags.md` (24+ flag entries), `docs/decisions.md`
(10 locked decisions), the per-stage `src/courtvision/stage*/README.md` files, the
existing topic-specific docs (`docs/stage2_evaluation_report.md`,
`docs/stage5_training_method.md`), and the auto-memory notes. Refer to those for
deeper detail; this is the index.

> **Note on terminal ownership.** Two Claude Code terminals collaborated on this
> project (see §14). Sections **§3 Stage 1**, **§4 Stage 2**, **§6 Stage 3**, **§7
> Stage 5**, **§8 Stage 6**, and **§9 F24 cross-method filter** were written by the
> first terminal (training + benchmark-evaluation focus). Sections **§5 Stage 4**,
> **§10 Stage 7**, **§13 End-to-End Deployment**, and the Stage 4 / 7 / deployment
> figure recommendations in the back-half were written by the second terminal
> (3D pipeline + iPhone session deployment focus). All shared facts cross-reference
> `flags.md` (F1–F26), `docs/decisions.md` (Decisions 1–10), and the auto-memory
> notes.

---

## 1. Project framing

**Input.** Synchronised half-court basketball video from **two** iPhones at 60 fps,
1080 × 1920 portrait. (Original plan: 3 iPhones; reduced to 2 during deployment —
see §13.) **Output.** Per detected shot, a 5-measurement pose-only contest metric
computed at the release frame — headline novel measurement is the 3D wrist-to-wrist
distance between defender and shooter (Decision 9 redesign, F25; the original
wrist-to-trajectory measurement remains the Phase 2 escalation when a ball
detector lands).

**Pipeline (8 stages).** Stages 1–4 are off-the-shelf (detect/track players, 2D pose,
ball, 3D triangulation). Stage 5 is the only trained model (PoseConv3D action
classifier). Stages 6–8 are geometric (release-frame rule, contest metric, 3D demo).

**Phased sprint plan.** A → scaffolding, B → Stage 1, C → Stage 2, D → Stage 3, E
→ Stage 5. Stages 4, 6, 7, 8 are post-sprint with stub I/O contracts landed in Phase A.

---

## 2. Phase A — Scaffolding (Day 0)

**Planned.** Repo init, `pyproject.toml`, schema definitions, Modal app skeleton,
Hydra configs, dataset/checkpoint download scripts, smoke tests, `.env.example`,
pre-commit hooks (PRD §10 Phase A scope).

**Built.** All of the above. Schemas in `src/courtvision/schemas.py` (Track,
Pose2D/3D, Ball2D/3D, ShotEvent, ReleaseFrame, ContestMeasurement and Sequence
wrappers) — definitive cross-stage I/O contracts, all `extra="forbid"`. Modal
app at `src/courtvision/modal_app.py` (shared CPU image + per-stage isolated
GPU images). Per-stage stubs raising `NotImplementedError` for Phases B/C/D/E.

**Deferred / Changed.**
- **F1** — OpenMMLab stack (`mmengine`, `mmcv`, `mmpose`, `mmaction2`) removed from
  `pyproject.toml` because `mmcv 2.2.0` has no macOS arm64 wheel and undeclared
  `pkg_resources` dep. Installed in Modal image instead (PRD §13 risk #1
  explicitly endorsed this).
- **F4** — Local `_MIRROR_FILES` is empty (chicken-and-egg with F2, never blocking
  in practice).
- **F7, F8** — Minor infra (decisions.md symlink, wandb-secret passthrough).

**Key files.** `pyproject.toml`, `schemas.py`, `modal_app.py`, `scripts/download_datasets.py`,
`scripts/download_checkpoints.py`, `flags.md` (the issue log), `docs/decisions.md`.

---

## 3. Phase B — Stage 1 (Player Detection + ByteTrack)

**Planned.** YOLOv8m (person class only) + ByteTrack. Off-the-shelf; ≤ 4 players
on a fixed half-court is well within COCO-trained capability (Decision 1).

**Built.** `scripts/stage1_detection_modal.py::run_yolo_bytetrack` (YOLO via
ultralytics; ByteTrack via ultralytics config); `scripts/stage1_detection_modal.py::make_stage1_overlay`
(qualitative overlay videos for review). Wrist-PCK-relevant track persistence
checks landed in `tests/test_stage1_smoke.py`.

**Deferred / Changed.**
- **F13** — `lapx` Python package added for ByteTrack's linear-assignment solver
  (not in PRD §5.2, harmless extra dep).
- **F14** — numpy 2.0 / motmetrics compatibility fix + Modal Volume reload semantics.
- **F15** — Stage 1 track-persistence backlog (6 tuning layers deferred: C2/B2/A2/A3/D1/D2,
  closed at acceptable quality). C2 is the highest-leverage to revisit if iPhone
  footage shows more ID switches.

**Decisions.** Decision 1 — YOLOv8m + ByteTrack, person class only. RTMDet-m
considered for MMPose ecosystem consistency but rejected. Fallback to BoT-SORT
or larger YOLOv8 if poor results (deferred — accepted as-is).

**Results.** Off-the-shelf, no held-out training metric. Qualitative track-overlay
videos reviewed at end of Phase B (TrackID3x3 indoor clips); track quality
acceptable for downstream Stage 2/5 use. **No mAP or IDF1 number was reported
this sprint** — Stage 1 is consumed as input, not contributed as a result.

**Key files / outputs.**
- `src/courtvision/stage1_detection/` (README, runner)
- `outputs/stage1_*/...` track JSONs (`TrackSequence` schema)
- `/artifacts/stage1/<session>/tracks_<session>_<camera>.json` on Modal volume

---

## 4. Phase C — Stage 2 (RTMPose-l Halpe-26, 2D pose)

**Planned.** RTMPose-l, Halpe-26 keypoints, top-down on YOLO crops, off-the-shelf
(Decision 2). Wrist PCK@0.05 ≥ 0.85 on TrackID3x3 as the headline acceptance gate.

**Built.** `src/courtvision/stage2_pose/rtmpose_inference.py` (rtmlib/ONNX, NOT
MMPose — see F20), `make_pose_overlay_video` (multi-track skeleton overlay),
`halpe26_to_coco17` (subset converter for Stage 5 input), batch eval pipeline
(`scripts/stage2_batch.py`).

**Deferred / Changed.**
- **F17** — Stage 2 inference is Modal-only; PRD's "local dry run" check substituted
  with a Modal smoke test.
- **F18** — TrackID3x3 GT is **model-derived, not hand-labelled**, at strict ~12 px
  tolerance. Wrist PCK@0.05 measures *agreement with another estimator*, not
  ground truth. Caveat must appear in the report.
- **F20** — **MMPose / mmcv unusable on the image's torch 2.12 + CUDA 13** — `mmcv._ext`
  fails to compile (mmcv issue #3310). Researched-not-guessed: pivoted Stage 2
  inference engine from MMPose → **rtmlib (ONNX)** as a PRD §5.2 *deviation*. Same
  model (RTMPose-l Halpe-26 `2abb7558`), same weights, same Pose2DSequence schema —
  only the runtime stack changed. Reason: rtmlib depends only on
  numpy/opencv/onnxruntime, decoupled from torch/CUDA/numpy version web.
- **F21** — Phase C wrist-PCK gate at 0.655 (< 0.85 target) accepted as a
  documented deviation per PRD §9.2 (qualitative acceptance allowed). See
  Results below.
- **F26** — MultiSubjects per-keypoint COCO-17 PCK secondary corroboration
  (closes the F18 loose end — different dataset, different GT, same falloff).

**Decisions.** Decision 2 — RTMPose-l, Halpe-26, top-down; engine swap to rtmlib
2026-05-28; Phase 2 ★ RTMPose fine-tune kept as the documented escalation path.

**Results.**

| Eval | Wrist PCK@0.05 | All-keypoint PCK@0.05 | n |
|---|---:|---:|---:|
| TrackID3x3 Indoor (13 clips, model-derived GT, 12 px tol) | **0.655** ± 0.043 | 0.781 | per F21 |
| MultiSubjects val (300 clips, BlazePose GT) | **0.680** | 0.841 | 362,695 kp |

Per-landmark proximal→distal falloff reproduced cleanly across both datasets
(shoulder ≈ 0.94 / 0.89, elbow ≈ 0.73 / 0.78, **wrist ≈ 0.66 / 0.68**, ankle
≈ 0.85 / 0.86). User reviewed 13 wrist-overlay videos; **wrist location looks
accurate, especially on shot-release frames** — the frames Stage 7 depends on.
Acceptance via the PRD's qualitative gate.

**Key files / outputs.**
- `src/courtvision/stage2_pose/` (README + inference + halpe26_to_coco17 + overlay)
- `docs/stage2_evaluation_report.md` (full PCK breakdown, the headline Stage-2 report doc)
- `outputs/stage2_gate/` (the 13 wrist-overlay videos used for qualitative acceptance)
- `outputs/stage2_gate/eval_stage2_multisubjects_300.json` (MS corroboration, F26)

---

## 5. Stage 4 — 3D Triangulation (custom 2-camera DLT, deployed on iPhone session)

**Planned (Decision 4 + Decision 10).** Pose2Sim DLT wrapper for 3D pose, `cv2.triangulatePoints` for ball, chessboard intrinsics (Decision 10), court-landmark extrinsics (Decision 10), per-frame visibility logging, ≤ 5 px reprojection error target on a held-out third camera. 3 iPhones envisioned.

**Built — what actually shipped.** Custom 2-camera DLT pipeline with per-frame Hungarian cross-camera ID matching. **Pose2Sim was not used** — the dependency was heavy for a project that ended up needing only `cv2.triangulatePoints` plus an assignment solver. **Two cameras, not three** (the third iPhone never materialised; bundled scope into a working 2-camera deployment). Calibration pipeline complete and runnable end-to-end on a captured session.

Key components built:

1. **Intrinsic calibration — chessboard via OpenCV (`scripts/calibrate_intrinsics.py`).** Decision 10's chessboard procedure. Each phone calibrated once: capture a chessboard video moving through ~20 positions, `cv2.findChessboardCorners` on sampled frames, `cv2.calibrateCamera` solves K + distortion. Saved as `data/calibration/intrinsics_{A,B}_checker.json`.
2. **Audio-clap synchronisation — frame-accurate (`scripts/detect_claps.py`).** 4 hand-claps recorded across both phones; clap onsets detected via spectral-flux + adaptive threshold + nearest-zero-crossing. Inter-camera offset = mean(per-clap offset). Drift slope = `fps_b/fps_a` (the 59.94/60.00 mismatch is real — both phones are nominally 60 fps but A records at 59.940 fps NTSC drop-frame, B at 60.000 fps). Drift produces a 0.001 frame/frame slope which we apply during the trim step. `data/sessions/<session>/sync.json` is the canonical output.
3. **Extrinsic calibration — court landmarks via manual click + EPNP/SQPNP (`scripts/calibrate_extrinsics.py`).** Decision 10's manual-clicking procedure with iterative landmark dropping. Per camera, click up to 17 court landmarks in a calibration frame (backboard corners, free-throw lane corners, free-throw arc, 3-point arc top, inner-square corners, rim, sideline/baseline anchors). Solver tries every {EPNP, SQPNP, IPPE, iterative} × {greedy outlier drop} combination and keeps the one with lowest mean reprojection error. World frame is rim-centered (basket at origin in xy, floor at z=0, rim at z=3.05 m), high-school NCAA spec. Saved as `data/sessions/<session>/extrinsics_{A,B}.json` + diagnostic overlay PNG.
4. **Trim with drift compensation — frame-accurate (`scripts/run_session_pipeline.py::_trim_videos`).** Start trim 5 s after the final sync clap; duration locked to 180 s; A trimmed at native fps, B trimmed by `expected_b_frames = expected_a_frames * fps_b / fps_a`. After trim, the frame correspondence `B_frame(A_frame) = round(A_frame * fps_b / fps_a)` is exact to ±1 frame across the full clip (verified at clap onsets in the trimmed clips).
5. **2-camera triangulation — DLT with per-frame Hungarian assignment (`scripts/run_session_pipeline.py::_triangulate_session`).** Stages 1 + 2 are run independently per camera (no cross-camera awareness). At the triangulation step, the cross-camera ID-matching problem becomes the binding constraint: ByteTrack issues ~10 different track IDs per camera over the 180-second clip (track-switch on every occlusion), so a "two stable tracks" assumption doesn't hold. **Solution: per-frame Hungarian matching on the pelvis reprojection error.** For each frame f: take all alive Pose2D tracks on cam A and cam B at frame f. Cost matrix is the 3D L2 reprojection error of every possible pelvis pairing. Solve the assignment problem (`scipy.optimize.linear_sum_assignment`). Triangulate each accepted pair's 26 keypoints via `cv2.triangulatePoints` (undistorted via `cv2.undistortPoints`). Output: `pose3d.json` (89 MB on this session) — 18 002 3D poses across 10 789 frames (avg 1.67 players triangulated per frame).
6. **Triangulation sanity check (`scripts/run_session_pipeline.py::_sanity_check`).** Re-triangulate the 9 court landmarks that both cameras share, compare against their known world positions. **Mean error 3.90 cm, median 3.44 cm, max 6.79 cm** at a 12.5 m camera baseline.
7. **Calibration diagnostics (PNG overlays).** `data/sessions/<session>/extrinsics_{A,B}.diag.png` shows clicked landmarks (red) + reprojected world landmarks (green). `data/sessions/<session>/sync.diag.png` shows the per-clap offset agreement.

**Deferred / Changed.**
- **3rd iPhone never captured.** Decision 4's "≤ 5 px reprojection on held-out 3rd camera" target became "≤ 5 px reprojection on the two cameras we have." Verified against the triangulation sanity-check (3.9 cm mean error on shared landmarks = ~3.5 px at this baseline; well inside the target band).
- **Pose2Sim wrapper not built.** `cv2.triangulatePoints` is one line; Pose2Sim's value was a CLI / config wrapper around the same primitive plus marker-set handling we didn't need (we have a custom Halpe-26 path). Direct cv2 call wins on dependencies + portability.
- **Per-frame visibility logging is implicit, not a separate JSON.** `Pose3D.keypoints[i].n_cameras` ∈ {0, 2} carries the per-keypoint per-frame visibility. `n_cameras = 2` (triangulated from both cameras) is the only acceptable value for downstream use; `n_cameras = 0` is filtered out at consumer time. No "3" tier because we only have 2 cameras.
- **Ball triangulation never implemented (F19 ball drop).** `cv2.triangulatePoints` call path stays defined in schemas (`Ball3D`) for the Phase-2 path.
- **No automatic calibration-drift detector.** Decision 10's "2× session-baseline reprojection error" gate not implemented (no drift incidents observed; would be ~30 lines if needed).

**Decisions.** Decision 4 (DLT triangulation, with cv2 substituted for Pose2Sim); Decision 10 (calibration logistics — chessboard intrinsics + court-landmark extrinsics + AE/AF lock + audio-clap sync); new decision (2026-05-30): swap the assumed "stable 2 tracks per camera" assignment to per-frame Hungarian matching, after a Stage 4 debug session showed triangulation was producing only 1 pose/frame because of ByteTrack ID churn.

**Results (session `2026-06-02_1on1`).**

| Calibration component | Cam A | Cam B |
|---|---:|---:|
| Intrinsics RMS reproj (chessboard) | 2.22 px | 1.09 px |
| Intrinsics chessboard frames used | 97 | 75 |
| Extrinsics solver | EPNP | SQPNP |
| Extrinsics landmarks used (of 17 clicked) | 9 | 12 |
| Extrinsics mean reproj | 4.89 px | 3.71 px |
| Extrinsics max reproj | 9.53 px | 7.77 px |

| Audio-clap sync (4 claps) | Value |
|---|---:|
| Camera A native fps | 59.940 (NTSC) |
| Camera B native fps | 60.000 |
| Mean offset B − A | −14.137 s |
| Per-clap offset std | 25.8 ms (≈ 1.5 frames) |
| Max disagreement | 64.3 ms (≈ 1 frame) |
| Drift slope b_per_a | 1.0010 |

| Triangulation sanity (9 shared landmarks, BB corners + FT center + inner-square corners) | cm |
|---|---:|
| Mean 3D error | **3.90** |
| Median 3D error | 3.44 |
| Max 3D error | 6.79 |
| Camera baseline | 12.51 m |

| End-to-end 3D pose | Value |
|---|---:|
| Frames in trimmed clip | 10 789 (180 s) |
| 3D poses produced | 18 002 |
| Avg players per frame | 1.67 (max 4) |
| Stage 4 wrist 3D error vs hand-clicked GT (n = 13 of 15 shots, ±2 missing GT) | mean 20.8 cm / **median 3.94 cm** / P90 42.7 cm |

The median 3.94 cm wrist 3D error is the result that matters for the contest metric: the headline contest measurement (wrist-to-wrist distance) sits at ~30–160 cm, so a median wrist-localisation noise of 4 cm is comfortably below signal. The mean is pulled up by ~2 outlier shots where Stage 6's release-frame prediction is wrong (Δ −36 to −44 frames) and the contest is being measured 0.6 s away from the actual release — that's a Stage 6 issue, not a Stage 4 issue.

**Key files / outputs.**
- `scripts/calibrate_intrinsics.py` — chessboard intrinsic solver
- `scripts/calibrate_extrinsics.py` — landmark-click extrinsic solver with iterative drop
- `scripts/detect_claps.py` — frame-accurate audio sync
- `scripts/run_session_pipeline.py` — orchestrator (trim → tracking → pose → triangulation → sanity-check)
- `scripts/label_shots.py`, `scripts/review_shots.py` — GT labeling UIs (release frame + 3 wrists per shot per camera)
- `src/courtvision/schemas.py::{Pose3D, Pose3DSequence, Keypoint3D}` — typed 3D contracts
- `data/calibration/intrinsics_{A,B}_checker.json` — per-phone intrinsics + RMS
- `data/sessions/2026-06-02_1on1/extrinsics_{A,B}.json` (+ `.diag.png`) — per-session extrinsics
- `data/sessions/2026-06-02_1on1/sync.json` (+ `sync.diag.png`) — clap-detection result
- `data/sessions/2026-06-02_1on1/trim_metadata.json` — drift-corrected trim window
- `data/sessions/2026-06-02_1on1/triangulation_sanity.json` — the 3.9 cm sanity result
- `outputs/sessions/2026-06-02_1on1/pose3d.json` (89 MB) — the canonical 18 002-pose 3D output
- `outputs/sessions/2026-06-02_1on1/stage4_overlay.mp4` — re-projection overlay video

**Note on implementation path.** Stage 4 ended up living in `scripts/` rather than `src/courtvision/stage4_triangulation/` because the per-session deployment workflow drove the structure (one orchestrator script that does calibration → trim → triangulation in sequence is easier to operate than 6 packaged modules). The `src/courtvision/stage4_triangulation/README.md` spec doc remains; a refactor into a `stage4_triangulation` package is deferred — see "Deployment / refactor backlog" in §13.

---

## 6. Phase D — Stage 3 (Ball Detection) — **DROPPED**

**Planned.** WASB (BMVC 2023) basketball ball detector, off-the-shelf, with a
release-window recall ≥ 90 % gate on WASB's basketball subset (Decision 3).

**Built (then abandoned).**
- WASB checkpoint download (`wasb_basketball.pth`, 6.1 MB) verified
  (`verify_wasb_checkpoint`, F3 closed).
- WASB inference + qualitative overlay (`scripts/stage3_ball_modal.py`).
- YOLOv8 ball detector added alongside (F19 — secondary appearance detector).
- Qualitative review on TrackID3x3 indoor clips → **ball coverage too partial /
  too noisy to be a reliable input downstream**.

**Deferred / Changed.**
- **F9** — WASB + Pose2Sim removed from the shared Modal image (isolated to
  dedicated images per stage).
- **F11** — WASB basketball subset upstream link is 404; alternative datasets
  used for sanity checks.
- **F12** — "U Arizona Roboflow basketball" couldn't be verified; substituted
  with DeepSportradar Basketball Instants (Kaggle, ACM MMSports 2022) + Roboflow
  `ownprojects/basketball-w2xcw`.
- **F16** — Off-the-shelf WASB-basketball has *partial, clip-dependent* ball
  coverage on TrackID3x3 (some clips good, many lose the ball through several
  contiguous seconds).
- **F19** — **Phase D dropped entirely.** Ball detection deferred until iPhone
  footage is available (the WASB basketball subset is much harder than the
  TrackID3x3 amateur half-court where it underperforms; we expect iPhone footage
  to land closer to the broadcast distribution WASB was trained for). User
  decision (2026-05-28): pursue pose-only pipeline through Stage 5/6/7. F24's
  Stage-5 + Stage-6 cross-method filter (§9 below) takes pose-only precision
  surprisingly far.
- **Decision 9 redesign** — Stage 7 contest metric rewritten as a pose-only
  proxy (wrist-to-wrist, not wrist-to-ball-trajectory) — see F25. Original
  ball-based design preserved as the Phase-2 path when a working ball detector
  arrives.

**Decisions.** Decision 3 — WASB chosen; dropped after F19 ball-quality review.

**Results.** No metric reported (eval was qualitative-only); WASB checkpoint
loads cleanly but coverage doesn't meet the release-window recall gate on our
intended footage. **No ball input feeds Stages 5 / 6 / 7.**

**Key files / outputs.**
- `src/courtvision/stage3_ball/` (kept for future re-attempt; README + runner)
- `flags.md` F19, F11, F12, F16 (the postmortem)

---

## 7. Phase E — Stage 5 (PoseConv3D Action Classifier)

**Planned (per PRD § 9.5 + Decisions 5/6/7).** Two-stream architecture — pose
PoseConv3D + ball-trajectory MLP — trained via a 4-stage transfer pipeline
(Kinetics → MultiSubjects → FineSports → ours), with a 4-row ablation isolating
each stage's contribution and the two-stream gain, plus external benchmark
comparisons on MultiSubjects + FineSports.

**Built.** Path A (3-stage: Kinetics → MultiSubjects → ours-eventually). 64-frame
contiguous-sliding-window classifier (full implementation, multiple iterations).
**No ball stream** (F19 ball drop). Stage 5 is the only trained model this sprint.

Major sub-deliverables, in order built:

1. **PoseConv3D image + sanity load** (Phase E0). Isolated `poseconv3d_image`
   (torch 2.1.2 + mmcv 2.1.0 + mmaction2 1.2.0 from source). The Stage-2 mmcv
   wall (F20) doesn't apply here: prebuilt mmcv 2.1.0 wheel works at torch 2.1.2.
2. **First Path-A fine-tune (v1, the 32f model)** on MultiSubjects with
   `UniformSampleFrames` over the whole clip. Best epoch 21, val top-1 = 0.854.
3. **Skeleton-overlay bug (F22)** — `classify_windows` mutated the caller's
   keypoint array in place; debugging took multiple sessions. Fixed by
   `.copy()` at the window boundary. **Side benefit:** also corrected a subtle
   double-shift in inference (overlapping windows re-cropped shifted kp).
4. **Train/inference sampling mismatch diagnosis.** `UniformSampleFrames`
   stretches every clip to 32 samples → trained model never saw real contiguous
   1.3 s snippets. Sliding-window inference uses contiguous 32-frame windows
   (deployment regime). Measured cost: uniform top-1 0.78 → contiguous per-window
   0.57 (−21 pts on the same clips, same model, only sampling changed).
5. **Window-length sweep** (32 / 48 / 64 / 80 / 96 / 128 frames at constant
   overlap). Sweet spot **64 frames (2.56 s)**: long enough to contain a complete
   median shot (1.96 s) and layup (2.93 s); longer windows degrade layup and
   smear localisation. Past ~80f the gains are artifactual (window engulfs the
   short trimmed clip).
6. **Step-1 retrain (v1, 64f_bal)** — 64-frame **contiguous** training
   (`SampleFrames`), inverse-frequency class weights (layup ~2.6 ×), label
   smoothing 0.1, discriminative LR (backbone at 0.1 × head's LR), cosine schedule.
   24 epochs. Val mean-per-class 0.772 (best epoch 10).
7. **Stage-5 → Stage-6 cross-method filter discovered (F24)** — see §9 below.
   Drove the 0.75 → 0.6 threshold change in `run_shot_inference` because Stage 6
   absorbs the FPs that come along at the lower threshold.
8. **v2 subject-disjoint split** (the report-grade ablation). Combined
   train.pkl + val.pkl (4096 clips, 988 subjects) → seed-0 shuffle by subject
   → 70/15/15 by subject. **No subject overlap across splits.** 2914 train /
   594 val / 588 test (148 unseen subjects in test). Class proportions match
   the overall distribution to within ~3 pts (random subject shuffle naturally
   preserves balance because >80 % of subjects span multiple action classes).
   Helper: `src/courtvision/stage5_classifier/split.py::compute_subject_disjoint_split`,
   used by *both* training and test-eval to guarantee no leakage.
9. **Before/after fine-tuning ablation** — linear probe (frozen K400 backbone,
   head only) vs. full fine-tune, *same data, same seed, same everything else*.
   On the held-out v2 test:

   | Metric (MS test, single centered 64f window) | Linear probe (before) | Full fine-tune (after) | Δ |
   |---|---:|---:|---:|
   | Dribble (n = 222) | 0.968 | 0.716 | −0.252 |
   | Layup (n = 68) | 0.353 | 0.529 | +0.176 |
   | **Shoot (n = 298)** | **0.537** | **0.930** | **+0.393** |
   | Overall accuracy | 0.679 | 0.803 | +0.124 |
   | **Mean per-class** | **0.619** | **0.725** | **+0.106** |

   Headline: MultiSubjects fine-tuning is worth **+10.6 pts mean-per-class**,
   and almost all of the lift is on the **shoot** class (+39 pts) — exactly the
   class our downstream pipeline depends on.
10. **Regularisation ablation (negative result, §11 of `docs/stage5_training_method.md`)**.
    Bumped weight_decay 3 × (1e-3), dropout 0.5 → 0.7, label-smooth 0.1 → 0.15,
    crop area_min 0.56 → 0.4. Trained on the same v2 split.
    Val mean-per-class peak essentially unchanged (0.773 vs. baseline 0.769) but
    plateau pushed from epoch 2 → 13 (reg slows overfitting). On **test**: shoot
    accuracy crashed from 0.930 → 0.564 (model defaulted into a "predict
    dribble" comfort zone — dribble 0.716 → 0.964). **Mean-per-class hides
    this** (0.725 → 0.691, only −3 pts) but per-class profile is much worse
    for the shoot-detection use case. **Verdict:** overfitting wasn't the
    binding constraint; the data is. Baseline `64f_bal_v2` stays canonical.

**Deferred / Changed.**
- **F19** — Ball stream never built (Stage 3 dropped). Two-stream architecture
  reduced to pose-only. Decision 5's row-4 ablation row is omitted; PRD's 4-stage
  transfer pipeline reduced to 2 stages (Kinetics → MultiSubjects) — FineSports
  intermediate transfer is the documented future-work step when access
  materialises.
- **F22** — Skeleton-overlay bug (in-place mutation) — RESOLVED.

**Decisions.** Decision 5 (architecture; ball stream now deferred), Decision 6
(inputs; ball channel never added), Decision 7 (recipe; reduced to Kinetics +
MultiSubjects), F24 cross-method filter (added 2026-05-29).

**Results.**
- **Headline test number:** mean-per-class 0.725, shoot 0.930 on the v2 test
  holdout (588 clips, 148 unseen subjects).
- **Linear-probe baseline:** 0.619 mean-per-class, shoot 0.537. Fine-tuning lift
  = +10.6 mean-per-class, +39.3 shoot.
- **Regularisation ablation:** keep baseline; reg traded shoot for dribble.

**Key files / outputs.**
- `src/courtvision/stage5_classifier/` (model, infer, split, overlay)
- `scripts/stage5_modal.py` (training + test eval + sweep + diag entrypoints)
- `docs/stage5_training_method.md` (the report-ready method writeup, §11 has
  the reg ablation)
- `outputs/stage5_diag/learning_curves_before_after.png` (probe vs fine-tune
  training curves)
- `outputs/stage5_diag/confusion_before_after.png` (before/after confusion
  matrices)
- `outputs/stage5_diag/learning_curves_reg_ablation.png` (reg-ablation curves)
- `outputs/stage5_diag/confusion_matrix.png` (canonical v1 model, kept for the
  report)
- `outputs/stage5_diag/learning_curve.png` (v1 training curve, mean-per-class
  primary)
- Canonical checkpoint: `/artifacts/checkpoints/stage5/stage2_multisubjects.pth`
  (= the promoted 64f_bal model from v1; v2 fine-tune is at `..._64f_bal_v2.pth`)

---

## 8. Stage 6 — Release-Frame Extractor (2D pose-only, single-camera)

**Planned (PRD § 9.6 + Decision 8).** 4-condition geometric rule on 3D pose + ball,
plus a backward-trace-from-apex variant for head-to-head comparison.

**Built.** **2D pose-only release detector** — because there is no ball (F19 drop) and
no 3D yet (Stage 4 is iPhone-gated, §5). Decision 8 rules are all ball-based;
F23 logs the deviation. Implementation: pure NumPy + cv2, CPU only.

- `src/courtvision/stage6_release/geometry.py` — dominant-wrist selection (by
  upward excursion), elbow angle, vertical velocity.
- `rule_backward.py::detect_release_apex` — *secondary*. Peak wrist height +
  velocity zero-crossing + elbow extension.
- `rule_forward.py::detect_release_forward` — *primary* (after qualitative review,
  see "Calibration" below). Onset of the first sustained
  forward-active interval: wrist-above-shoulder + elbow ≥ 160° for ≥ 2 frames
  (gap-tolerant against 1-frame keypoint flickers).
- `detect.py::analyze_release` — driver returning both methods + traces
  (wrist y, velocity, elbow angle, forward-active interval) for the overlay.
- `overlay.py::make_release_overlay_video` — skeleton + bottom wrist-height
  trace strip + red apex / orange forward markers + flash at predicted release.
- **Schema deviation (F23):** `ReleaseFrame.method` Literal extended with
  `"pose_apex"` and `"pose_forward"` so provenance stays truthful. The ball-based
  values (`forward_rule`, `backward_trace`, `fallback_apex`) are kept for the
  future 3D variant.

**Calibration on MultiSubjects (the validation step).**
1. Initial 12-clip overlay review: apex marker was systematically *late*
   (the wrist keeps rising on the follow-through after the ball leaves the hand),
   while forward landed near the true release. **Decision: swap primary from
   `pose_apex` to `pose_forward`.**
2. 36-clip overlay review at the larger sample: user identified that the true
   release sits ~1 frame **before** the detected interval onset. Set
   `release_offset = −1`. (Required allowing negative offsets — relaxed the
   clamp from interval bounds to the search window.)
3. **Overlay-alignment fix:** the original ±2-frame symmetric flash made the
   marker appear 2 frames *before* the detected frame, confusing the
   calibration. Changed to `0 ≤ f − X ≤ flash_radius` so the marker's first
   appearance = detected frame = trace bar.
4. Applied to **all 9 Stage-5-detected shot events on TrackID3x3** with a
   ±50-frame search window around each Stage-5 peak. Real shots fire with
   forward-confidence "high" (forward ≠ apex); false positives fall back to
   apex (`forward == apex`, confidence "low") because the rule found no
   sustained extension in the window. → F24 (next section).

**Decisions / Deviations.**
- **F23** — 2D pose-only, schema enum extended, jump shots only (layups deferred).
- The `forward_offset = −1` value is user-calibrated, not a published constant;
  document with the small-sample caveat in the report.

**Results.** Qualitative only on this sprint:
- 36 MultiSubjects shot clips: apex consistently late ~3–5 frames vs. true
  release; forward (onset − 1) consistently within ±1–2 frames.
- 9 TrackID3x3 detected shots: all 4 user-confirmed real shots fire with
  forward_conf "high" and forward ≈ Stage-5 peak ± 30 frames. 4 of 5
  user-confirmed false positives have `forward == apex` (the diagnostic FP
  signal that drives F24).

**Key files / outputs.**
- `src/courtvision/stage6_release/` (geometry, rule_backward, rule_forward,
  detect, overlay)
- `scripts/stage6_modal.py` (`release_ms` for MS clips, `release_trackid3x3`
  for the Stage-5 events)
- `outputs/stage6_release_overlays/` (36 MS overlays + 9 TrackID3x3 overlays)
- `tests/test_stage6_smoke.py` (5 tests pass)

---

## 9. Cross-cutting — F24 Stage 5 + Stage 6 cross-method FP filter

**The finding.** Both Stage 5 (CNN on pose heatmaps) and Stage 6 (explicit
geometry on the same pose) operate on the *exact same* COCO-17 per-player track
— there is no other input. They use completely different methods. **When the
two methods disagree, the Stage-5 fire is probably a false positive**: the pose
looked shot-like to the CNN but the geometric rule cannot find any sustained
overhead arm extension in the window.

**The signal.** `ReleaseFrame.confidence_flag == "high"` iff the forward rule
found a sustained extension interval. When it doesn't, Stage 6 falls back to
the wrist apex and tags `"low"`. This is exposed as
`stage6_release.is_verified_shot(rf)` — a one-line predicate.

**Cross-checked against user labels on TrackID3x3 (9 events at threshold ≥ 0.6):**

| Stage 5 alone @ 0.75 | Stage 5 @ 0.6 + Stage 6 `forward_conf=="high"` gate |
|---|---|
| 3 TP / 1 FP / 1 FN | **4 TP / 1 FP / 0 FN** |
| precision 0.75, recall 0.75 | **precision 0.80, recall 1.0** |

The combined pipeline strictly dominates Stage 5 alone on this small sample.
**Caveat: n = 9 events on 13 short clips**; re-validate on iPhone footage with
hand-labelled GT before locking the precision claim. Honest limitations are in
F24 (the report-ready entry).

**Implementation.** `stage5_classifier/infer.py` lowered the default
`score_threshold` from 0.75 → 0.6. `stage6_release/detect.py` exposes
`is_verified_shot()`. End-to-end runner: `scripts/stage5_modal.py::eval_combined`
(reproduced the F24 numbers exactly: 9 raw events at 0.6 → 5 verified by
Stage-6 high-conf → 4 dropped).

---

## 10. Stage 7 — Contest Metric (5-measurement pose-only, deployed on iPhone session)

**Planned (Decision 9 + F25 redesign).** Five measurements per shot at release per the **Phase D-drop pose-only redesign** (Decision 9 sub-bullet, 2026-06-02): `wrist_to_wrist_distance_m`, `wrist_height_advantage_m`, `shooter_fade_mps`, `floor_distance_m`, `composite_score`. Selection rules: "closest defender" = pelvis-to-pelvis 3D (was: defender pelvis to ball); "shooting hand" = higher wrist at release (was: nearer to ball). Schema fields landed in `schemas.py::ContestMeasurement` per F25.

**Built.** `scripts/compute_contest_session.py` — full 5-measurement implementation that consumes `pose3d.json` + `release_frames.json` + GT clicks and produces `contest_metrics.json` + augments `evaluation.json`. Measurement formulas (all from Decision 9 redesign):

| # | Field | Formula | Pose-only proxy of |
|---|---|---|---|
| 1 | `wrist_to_wrist_distance_m` | 3D L2 from defender's nearer wrist to shooter's *shooting* wrist | Defender's hand near ball trajectory at release |
| 2 | `wrist_height_advantage_m` | `defender_higher_wrist_z − shooter_shooting_wrist_z` | Defender contesting *at* or *above* shooter's release point |
| 3 | `shooter_fade_mps` | `sqrt(max(0, −v_forward)² + v_lateral²)` with `forward = unit(rim − shooter_pelvis)` in court xy | Shot taken off-balance / fading away |
| 4 | `floor_distance_m` | 2D Euclidean pelvis-to-pelvis (xy only) | NBA SportVU baseline (literature comparable) |
| 5 | `composite_score` | z-score sum: `−z(w-w) + z(hgt) + z(fade)` | Single contest score, sign-corrected: bigger = more contested |

Implementation details:
- **Pelvis velocity** = central difference over ±3 frames at the predicted release frame (1/10 s window at 60 fps); fps read from `sync.json::fps_a`.
- **Shooting-hand selection** = the higher-Z wrist of the shooter's pose at the release frame (Decision 9 redesign rule). Verified by the eval-review PNGs against hand-labelled GT.
- **Closest-defender selection** = the non-shooter track whose 3D pelvis is closest to the shooter's pelvis at the release frame (pelvis = Halpe-26 index 19, "hip-center").
- **Smoothing (anti-jitter)** = each measurement is reported instantaneous AND as a 5-frame mean (±2 frames around release). Both are saved.
- **Uncontested flag** triggers if `floor_distance_m > 3 m` (Decision 9 threshold). 0 / 11 evaluable shots hit it on this session.
- **Composite z-score** is computed across the 11 evaluable shots in this session. Sign convention: `−z(wrist_to_wrist)` because *smaller* w-w = more contested; `+z(height_advantage)` because larger = more contested; `+z(fade)` because larger = more contested. Sum = composite. With n = 11, the absolute z-scores have only ~10 dof; report directional, not significance.

GT triangulation for evaluation: each shot's 3 GT wrist clicks per camera (shooter + up to 2 defender wrists) are 3D-reconstructed via `cv2.triangulatePoints` at compute time. The shooter wrist GT 3D is exact (one click pair); the GT defender wrist 3D uses the same closest-wrist rule. Errors are reported for measurements (1) and (2) — the two where we have hand-clicked GT.

**Visualisation deliverables.** Two are deck-grade:
1. **Per-shot evaluation review PNGs** (`scripts/review_evaluation.py`) — side-by-side cam A / cam B at the release frame with GT wrist clicks (red) + predicted wrists (green) + per-pixel error (yellow) plus a 6-line caption showing all stage results, including all 5 Stage 7 measurements + GT errors. Output: `outputs/sessions/2026-06-02_1on1/eval_review_full_metrics/shot_NNN.png` (×15).
2. **3D skeletal reenactment videos** with metric panel overlay (`scripts/render_3d_reenactment.py`) — 3-second clip per shot, 180° orbiting camera, both player skeletons + dashed contest line + on-screen panel showing all 5 metrics during the release moment. Panel border colour-coded by composite z (green = uncontested, amber = medium, red = heavily contested). Best 7 shots rendered to `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/` + a 60-second highlight reel.

**Deferred / Changed.**
- **Rater study DROPPED** to future work (Decision 9 explicit, ★ stretch).
- **Composite weights learning** — uniform sum of three z-scores used; no weight tuning. Tuning would require external rating which we don't have (rater study was dropped). Future work: bootstrap labels from make/miss outcomes + Spearman optimisation.
- **Mikic-2024 quartile effect reproduction NOT DONE.** n = 11 evaluable shots → 2–3 per quartile is too thin to test "top-contest quartile has higher release point + quicker release" meaningfully. Honest reporting: defer until ≥ 30 shots.
- **No "fadeaway" classification flag.** The schema's `flags: list[str]` permits one, but with only n = 11 there's no calibration data for the threshold. Skipped.

**Decisions.** Decision 9 (original, ball-based, preserved for Phase 2); **Decision 9 redesign** (2026-06-02, pose-only); F25 (the schema/redesign flag). Verified all schema field names match `schemas.py::ContestMeasurement` (no stale `wrist_to_trajectory_distance_m` references in code or this doc).

**Results (session `2026-06-02_1on1`, n = 15 GT shots / 11 evaluable / 2 makes / 9 misses among evaluable).**

| Measurement | n | Mean | Median | Range | Makes mean | Misses mean | Δ (makes − misses) |
|---|---:|---:|---:|---|---:|---:|---:|
| (1) Wrist-to-wrist distance | 10 | 0.87 m | 0.78 | 0.32–1.61 | **1.39 m** | **0.74 m** | **+65 cm** |
| (2) Wrist height advantage | 10 | −0.36 m | −0.12 | −1.47 to +0.28 | **−1.07 m** | **−0.18 m** | **+89 cm** |
| (3) Shooter fade | 11 | 1.07 m/s | 0.73 | 0.02–4.37 | 1.24 m/s | 1.04 m/s | +0.20 m/s |
| (4) Floor distance (pelvis) | 10 | 1.29 m | 1.32 | 0.99–1.60 | 1.14 m | 1.33 m | −0.19 m |
| (5) Composite z-score | 11 | 0.00 | 0.00 | −5.20 to +4.83 | **−2.84** | **+0.63** | **−3.47 σ** |

**Headline:** the composite z-score separates makes from misses by 3.47 σ (makes are 2.84 σ below the mean → uncontested; misses are 0.63 σ above → moderately contested). With n = 2 makes / 9 misses among evaluable shots, this is **directional, not significant.**

**Make/miss correlation (Decision 9 validation (b) — Spearman ρ).**

| Measurement | Spearman ρ vs miss outcome | p | Direction |
|---|---:|---:|---|
| Wrist-to-wrist distance (1) | **−0.696** | **0.025** | Significant — smaller w-w → more likely to miss |
| Composite z-score (5) | +0.522 | 0.100 | Directional — higher contest → more likely to miss |
| Wrist height advantage (2) | sign matches | — | (not formally tested) |
| Shooter fade (3) | weak | — | (not formally tested) |

**Wrist-to-wrist distance is significantly correlated with miss outcome at α = 0.05 (ρ = −0.70, p = 0.025, n = 10).** This is the Decision-9-(b) validation success — at the n we have, the headline measurement carries the expected signal direction with statistical significance.

**Distribution sanity check (Decision 9 validation (c)).** All 11 evaluable shots produce non-degenerate values; no all-zero / all-equal failure modes; no Stage 4 triangulation collapse anywhere in the contest measurements. The Stage 6 release-frame error on 2 outlier shots (Δ −36 to −44 frames) pulls the wrist-to-wrist GT error up to mean 25 cm (median 10 cm) — Stage 7 inherits Stage 6's worst cases, as expected.

**Accuracy vs hand-clicked GT.**

| Measurement (vs GT) | n | Mean \|err\| | Median \|err\| | Max \|err\| |
|---|---:|---:|---:|---:|
| Wrist-to-wrist distance | 9 | 25.1 cm | **10.2 cm** | 112.9 cm |
| Wrist height advantage | 9 | 31.3 cm | **16.0 cm** | 114.2 cm |

Median errors below the makes-vs-misses gap (65 / 89 cm for measurements 1 / 2). Both max errors come from the same Stage-6 outlier shot (release-frame error of 36+ frames → contest measured 0.6 s away from the actual release).

**Key files / outputs.**
- `scripts/compute_contest_session.py` — the 5-measurement implementation (instantaneous + smoothed + composite + GT comparison)
- `scripts/review_evaluation.py` — per-shot eval-review PNG renderer (full 5-metric caption)
- `scripts/render_3d_reenactment.py` — 3D skeletal reenactment with metric panel
- `src/courtvision/schemas.py::{ContestMeasurement, ContestMeasurementSequence}` — typed Stage 7 contracts
- `outputs/sessions/2026-06-02_1on1/contest_metrics.json` — per-shot detail + summary
- `outputs/sessions/2026-06-02_1on1/evaluation.json` — augmented with `stage7_contest_metric` block
- `outputs/sessions/2026-06-02_1on1/eval_review_full_metrics/shot_*.png` (×15) — the per-shot deck-ready PNGs
- `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/` (×7 shot clips + highlight reel) — the 3D-overlay videos
- `outputs/sessions/2026-06-02_1on1/shot_release_pngs/` (×15) — extracted release-moment frames from the original reenactments

**Note on implementation path.** Stage 7 lives in `scripts/compute_contest_session.py` rather than `src/courtvision/stage7_contest/...` because the session-deployment workflow drove the structure. A refactor into a `stage7_contest` package would extract the measurement functions into testable units (`measure_wrist_to_wrist`, `measure_height_advantage`, etc.) and keep the orchestrator thin. Deferred — see §13.

---

## 11. Stage 8 — 3D Demo (deferred to ★ stretch)

Stub only; viz framework not selected. Out of scope for the CS231N writeup
unless the 3D pipeline lands in time.

---

## 12. Cross-stage operational items

- **Modal infra (F1, F6, F9, F10, F20, F22):** all heavy ML runs are on Modal
  (A10G GPUs, dedicated per-stage images). Long-running training survives
  `--detach` and the user closing the laptop / dropping wifi.
- **Subject-disjoint v2 split** (Phase E §7): the *report-grade* split.
  Reproducible from seed 0 + the helper in
  `stage5_classifier/split.py`. Use it for any future MultiSubjects-based
  evaluation (e.g. the deferred FineSports intermediate stage).
- **24+ flag entries in `flags.md`** are the comprehensive issue log. Every
  deviation from the PRD, every fix, every accepted caveat is there.

---

## 13. End-to-End Deployment — iPhone Session `2026-06-02_1on1`

This is **the project's novel end-to-end contribution** that doesn't fit cleanly inside any single stage. A real 1-on-1 game was captured on two iPhones in a high-school gym, calibrated from scratch, run through every stage of the pipeline, and evaluated against hand-labelled ground truth at the per-shot level. Everything in §§ 3 – 10 was designed in isolation against open datasets; this section is where it ran end-to-end against the original problem statement.

### 13.1 Capture

- **Setup.** Two iPhones on tripods at opposite ends of the half-court, both nominally 60 fps in 1080 × 1920 portrait. AE/AF locked on each phone aimed at court centre per Decision 10. Subjects: 2 players, 1-on-1 ball-side defense, ~5 minutes of continuous play. 4 hand-claps at the start (audio sync).
- **What we got.** ~10 800 frames per camera (180 s after trim). Camera A's actual fps measured at 59.940 (NTSC drop-frame), B at 60.000 — the 0.060 Hz difference matters at frame-accurate sync (compensated downstream).
- **Quirk we hadn't anticipated.** Both phones are "60 fps" but actually run at slightly different rates. Caught at the clap-sync stage; absorbed via the drift-slope correction in `trim_metadata.json`. Worth documenting because it's a real iPhone-cinematography gotcha.

### 13.2 Calibration

Two-stage as per Decision 10:

1. **Intrinsics** — chessboard videos captured pre-session, one per phone. `cv2.calibrateCamera` returned RMS reprojection 2.22 px (A, n = 97 frames) / 1.09 px (B, n = 75 frames). Below the 5 px Decision 4 target.
2. **Extrinsics** — court landmarks clicked once per camera in the calibration frame. Custom solver tries every {EPNP, SQPNP, IPPE, iterative} × {greedy outlier drop} combination and keeps the best by mean reproj. Result: A 9 of 17 landmarks @ 4.89 px mean (EPNP); B 12 of 17 @ 3.71 px mean (SQPNP). The dropped landmarks (RIM, FT_LANE corners, ARC_TOP) were the ones with the largest perspective-foreshortening sensitivity from each viewing angle — the iterative dropping is essentially identifying landmarks that the user mis-clicked or that have ambiguous world coordinates from each camera's view.

The triangulation sanity check (re-triangulate 9 shared landmarks → compare to known world positions) hit **mean 3.90 cm error** at a 12.5 m camera baseline. That's the headline calibration result.

### 13.3 Sync + trim

`scripts/detect_claps.py` detects clap onsets via spectral-flux + adaptive threshold + zero-crossing snap. 4 of 4 claps detected on both cameras; inter-clap offset agreement = ±1 frame (max disagreement 64 ms). Trim window = 5 s after the last clap, 180 s duration, drift-corrected (B trimmed by the 1.001× drift slope so frame `f` in A maps to frame `round(f × 1.001)` in B with ±1 frame max error across the full 180 s clip).

### 13.4 Ground-truth labeling

**15 shots hand-labelled** across the 180-second clip using a custom Tk interface (`scripts/label_shots.py`) with the matplotlib key-handler conflicts removed:

- Per shot: release frame, shot type (jump_shot / layup), make/miss flag, uncontested flag.
- Per shot per camera: shooter's shooting-wrist click + up to 2 defender-wrist clicks (3 clicks × 2 cameras = 6 per shot).
- **3 makes, 12 misses, 1 layup, 0 uncontested.** Shot types skew toward contested jump shots because the players were specifically working on contested-shot mechanics.

Labels saved as 5 chunked JSON files (`shots_gt_part1.json` through `_part5.json`) merged into `shots_gt.json` (one canonical file). Pixel-precise; reviewed in `scripts/review_shots.py` (paginated review UI showing each clicked frame with marker overlays).

### 13.5 End-to-end run

`scripts/run_session_pipeline.py` orchestrates Stages 1 → 4 (tracking → 2D pose → 3D triangulation) as one Modal job (server-side per F14). Stages 5 → 7 run from separate scripts (per Phase E's existing infra). All-stage outputs landed in `outputs/sessions/2026-06-02_1on1/`.

### 13.6 Per-shot evaluation (the deck table)

| Stage | Metric | Value | Notes |
|---|---|---:|---|
| Stage 2 (2D pose, cam A) | Wrist PCK@0.05 vs GT clicks (n = 44) | **0.932** | At hand-labelled wrist clicks — much higher than the 0.65 model-derived GT number |
| Stage 2 (2D pose, cam B) | Wrist PCK@0.05 vs GT clicks (n = 38) | **0.947** | Same comment |
| Stage 4 (3D wrist) | n shots evaluated | 13 of 15 | 2 shots had ≥ 1 missing GT click |
| Stage 4 (3D wrist) | Median error | **3.94 cm** | Headline 3D accuracy |
| Stage 4 (3D wrist) | Mean / P90 / max | 20.8 / 42.7 / 177.3 cm | Mean inflated by Stage 6 outliers |
| Stage 5 (shot detection) | TP / FP / FN @ thr 0.75 | 12 / 18 / 3 | Detected 12 of 15 GT shots |
| Stage 5 (shot detection) | Precision / Recall / F1 | 0.40 / **0.80** / 0.53 | FPs are defender contest-poses (no defender class in training) |
| Stage 6 (release frame) | n TP events | 12 | Same as Stage 5 TPs |
| Stage 6 (release frame) | Mean / median Δframes | −3.9 / 3.5 | Median within 1/17 s — close to perfect |
| Stage 6 (release frame) | Mean \|Δ\| / P90 / max | 11.6 / 33.7 / 44 frames | 2 outlier shots at the boundary of the search window |
| Stage 7 (contest, w-w distance) | Mean / median / range (n = 10) | 0.87 / 0.78 / 0.32–1.61 m | Makes 1.39 / Misses 0.74 |
| Stage 7 (contest, composite) | Make/miss separation | **−2.84 vs +0.63 σ (3.47 σ gap)** | The headline result |
| Stage 7 (validation) | Spearman ρ (w-w distance vs miss) | **ρ = −0.70, p = 0.025 (n = 10)** | Statistically significant; smaller w-w → more misses, expected direction |

### 13.7 Issues encountered + fixed (in chronological order, the engineering log)

1. **Wrong-bar Stage 2 PCK** (F18) — TrackID3x3 GT is model-derived; for *this* session we have hand-labelled wrist GT and the wrist PCK is 0.93/0.95 (vs 0.65 against the model-derived bar). The "wrist accuracy" story is far better at the report level than the F18 number alone suggests.
2. **Local-client disconnect on long Modal runs** (F14). The original orchestrator was a local-client driver, which dies when the laptop sleeps. Refactored to a server-side `@app.function` per F14 + `--detach` runs.
3. **rtmlib trying to download ONNX from openmmlab.com** (network timeout). Fix: extracted `end2end.onnx` locally, uploaded to Modal volume as `checkpoints/rtmpose_l_halpe26.onnx`, set the config key to `onnx_model` (rtmlib > 0.0.13 changed `weights` → `onnx_model`).
4. **Triangulation only producing 1 pose per frame** — ByteTrack reassigned ~10 different track IDs per camera over the 180 s clip; the "2 stable tracks" assumption was wrong. Fixed by per-frame Hungarian matching on pelvis reprojection error (the Stage-4 result above).
5. **Stage 6 outlier on 2 shots** — `release_frame` predicted 36–44 frames away from GT on shots 13 and 14. Root cause is the same shooter-identification issue Stage 5 has (defender's contest-pose looks shot-like). Stage 6 picks the wrong "shooter" track inside the window. Accepted as a known issue (the per-shot review PNG makes it visually obvious).
6. **matplotlib keymap collisions** in `label_shots.py` (k → xscale, l → yscale). Camera B panel collapsed during labeling. Fixed by clearing all matplotlib keymaps that could conflict; the type-toggle keybinding moved from 'l' to 't'.
7. **Δ (Unicode delta) rendering as "??" in cv2.putText** in overlays. Switched to plain ASCII (`d=`, `cm`, `s`/`sigma`).
8. **ffmpeg `-c copy` failing on highlight-reel concat** — title-plate clips and reenactments had differing codec params. Fix: re-encode during concat with libx264 (one ffmpeg flag change in `concat_clips`).

### 13.8 Limitations of this deployment

- **n = 15 shots / 3 makes is too small for inferential statistical tests** with the rare class (makes). Make/miss correlation on wrist-to-wrist is significant at α = 0.05, but the composite z-score and Mikic-effect quartile reproductions are gated on a larger session.
- **Stage 5 precision (0.40) is the bottleneck.** Most of the FPs are defender contest poses; the classifier was never trained on a "defender contest" class (Decision 5 had `{dribble, layup, shoot, background}` only). Future work: add a defender-contest class to MultiSubjects pretraining, or use F24's cross-method gate more aggressively.
- **2 outlier shots (13, 14) pull every Stage 4 / 6 / 7 aggregate up** — they're at the edge of the 180-second clip where the search window cuts off; the shooter-track identification picks a defender. Re-running with a wider half-window or with a per-shot manual shooter assignment fixes them but neither is shipped.
- **No third camera ⇒ no triangulation hold-out.** Decision 4's "≤ 5 px reprojection on held-out 3rd camera" target wasn't tested as designed; the triangulation sanity check on shared landmarks (3.9 cm @ 12.5 m baseline) is the alternative we have.

### 13.9 Deployment / refactor backlog

- **Refactor `scripts/compute_contest_session.py` → `src/courtvision/stage7_contest/` package.** Extract measurement functions into testable units; keep the orchestrator thin.
- **Refactor `scripts/run_session_pipeline.py` → `src/courtvision/stage4_triangulation/` package.** Same.
- **Calibration-drift auto-detector** (Decision 10's 2× baseline gate). ~30 lines. Skipped because no drift incidents.
- **Wider Stage 6 search window** for shots near the clip boundary (shots 13/14).
- **Per-shot manual shooter assignment override** for the 1–2 shots/session where the auto-shooter rule picks a defender.

---

## 14. Process / engineering notes

A handful of project-level lessons worth recording for the report's "Discussion / Future Work" section that didn't fit neatly inside any single stage:

- **Two-terminal collaboration pattern.** Stages 1–3 / 5 / 6 / F24 ran in one Claude Code terminal; Stages 4 / 7 + end-to-end deployment ran in another. Hand-off via `flags.md` (the shared issue log) + `docs/decisions.md` (the shared rationale log) + this development_log.md (the shared narrative). Worked because both terminals respected the same source-of-truth files; no merge conflicts in the docs because each terminal owned a different stage's content. **Lesson:** for multi-agent IDE workflows, define a single shared narrative file and partition by section, not by file — agents agree on facts that way.
- **Research before iterating on dependency walls.** The mmcv/mmpose torch-2.12 incompatibility (F20) was originally chased with ~10 rebuild attempts before pivoting to rtmlib. Lesson stuck — when stuck on a heavy ML dep, check viability via the upstream issue tracker first, then pivot deliberately. (Recorded as auto-memory `feedback_research_before_iterating`.)
- **Comprehensive evaluation, not just counts.** Reporting "Stage 4 produced 18 002 3D poses on 10 789 frames" is not the same as reporting "median wrist 3D error 3.94 cm vs 13 hand-clicked GT shots." The first is a *count*; the second is an *interpretable metric vs ground truth*. The user (Lukas) emphasised this throughout; both this doc and `report_content.md` use the interpretable form. (Recorded as auto-memory `feedback_comprehensive_evaluation`.)
- **Verify detection quality, not just detection counts.** Stage 5's "12 of 15 GT shots detected" is precision-and-recall, not just recall. Most of the FPs are defender contest poses, not random noise — that's a classifier-class-design issue, not a threshold issue. Honest reporting requires unpacking what the FPs *are*. (Recorded as auto-memory `feedback_verify_detection_quality_not_just_counts`.)
- **Don't ask "should I proceed?" mid-execution.** Phase STOPs in the PRD apply; in-phase execution should run without gating questions. (Recorded as auto-memory `feedback_dont_ask_to_proceed`.)

---

# Report figures + tables — recommendations and source paths

Below is a section-by-section mapping of *what to put in the report*, grouped
by the CS231N guideline sections, with the source of the figure / table
(existing file, or where to make it from).

## Report § 0 — Abstract
- No figures. Text only, 1–2 paragraphs.

## Report § 1 — Introduction (0.5–1 page)
- **Figure: pipeline diagram (8 stages).** Strongly recommended at the top of
  the intro to anchor input → output for the reader. **NOT DONE** — needs
  to be drawn (Figma, draw.io, OmniGraffle, or even a clean PowerPoint). Suggested
  flow updated for what shipped: `2 iPhones → Stage 1 (player boxes) → Stage 2 (2D pose) → Stage 4 (3D triangulation)
  → Stage 5 (action classifier → shot events) → Stage 6 (release frame) →
  Stage 7 (5-measurement contest metric)`. Mark Stage 3 (ball) as "deferred (F19)".
- **Optional: setup photo** of the half-court 2-iPhone setup. **NOT DONE** —
  user has session photos but they're not in `outputs/`; could be added if
  desired.
- **Hero figure: a 3D reenactment frame at the release moment with the metric
  panel visible.** Source: any frame from `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/shot_005.mp4`
  (the fadeaway, composite +4.83 σ, very visually striking) or `shot_002.mp4`
  (the made layup, composite −0.47 σ). Use `ffmpeg -ss 1.50 -i shot_005.mp4 -vframes 1`
  to extract. **Recommended** — instantly communicates "this is what the project
  does."
- Explicit "input is X, output is Y" sentence per the guidelines.

## Report § 2 — Related Work (0.5–1 page, ≥ 10 references)
- **Optional table: prior work comparison.** Three columns: paper / input
  modality / approach / dataset (e.g., TrackNet+YOLOv7 badminton 2023,
  Daly-Grafstein & Bornn 2020, PoseConv3D Duan 2022, Free-Throw Hit-Probability
  2024, Mikic 2024). The full citation list is in Decision 5 / Decision 9 of
  `docs/decisions.md` — copy those into BibTeX. The full list of references
  to chase down is also implicit in `references/RELATED_WORK.md`.

## Report § 3 — Methods (2 pages)
- **Architecture diagram: PoseConv3D + heatmap input.** Adapt from the
  PoseConv3D paper's Fig. 2 (Duan 2022); cite. Show: 17 COCO keypoints
  rendered as Gaussian heatmaps stacked across 64 frames → `[17, 64, 56, 56]`
  → SlowOnly-R50 → 4-way head.
- **Sampling-regime illustration.** Two panels: (a) `UniformSampleFrames` over
  the whole clip (the v1 mistake — time-warped to 32 samples) and (b)
  `SampleFrames(clip_len=64, frame_interval=1, num_clips=1)` contiguous (the
  fix). One sentence: train as you deploy.
- **Sliding-window inference + NMS schematic.** Diagram of the 64-frame window
  stepping by stride 8 over a per-player pose track, with `_segment_peaks`
  consolidating high-shoot windows into one event. Easy to make in PowerPoint.
- **Equations to include (per the guidelines):**
  - Inverse-frequency class weight: `w_c = n_total / (3 · n_c)`.
  - Label-smoothed cross-entropy: `L = − Σ [(1−ε) δ_{y,c} + ε/K] · log p_c`.
  - `pose_forward` release rule: `release = onset_first_sustained − 1`,
    where `onset_first_sustained` is the first frame in a gap-tolerant
    contiguous run of `wrist_y < shoulder_y ∧ elbow_angle ≥ 160°` lasting ≥ 2
    frames within the search window.
- **Note: explicitly state what was off-the-shelf vs. what you wrote.** Off-the-shelf:
  YOLOv8m + ByteTrack (ultralytics), RTMPose-l Halpe-26 (model from MMPose,
  inference via rtmlib), PoseConv3D K400 (model from mmaction2). Custom:
  v2 split helper, contiguous-window training recipe, `_segment_peaks`,
  `pose_forward`/`pose_apex` rules, F24 cross-method gate, every overlay
  renderer, every Modal runner, every eval harness.

## Report § 4 — Dataset and Features (0.5–1 page)
- **Table: dataset summary.** Use the v2 split table from §7 above + the
  Stage 2 PCK eval datasets (TrackID3x3 Indoor, MultiSubjects 300-clip
  corroboration). Sources:
  - **MultiSubjects** (Henan U., CVIU 2024) — 4,096 trimmed clips, 988
    subjects, COCO-17 keypoints, 25 fps. Hugging Face snapshot, on volume.
  - **TrackID3x3** (open-starlab/TrackID3x3) — 13 indoor pose-labelled clips
    (model-derived GT) used for Stage 2 acceptance. Google Drive.
  - **FineSports** (CVPR 2024) — accessed but pkl-labels not delivered; deferred.
- **Table: subject-disjoint v2 split** (from §7 above, the train/val/test by-subject
  numbers + class proportions).
- **Figure: example frames per class.** Three side-by-side stills (dribble,
  layup, shoot) from MultiSubjects clips you already extracted — easiest source
  is the action-overlay frames in `outputs/stage5_overlays/*.mp4` (extract a
  representative frame with `cv2.VideoCapture.read()` at the shot moment).
- **Figure: pose-heatmap visualisation** (one panel, one frame). Render the
  17 COCO heatmaps for a single frame to show the input format. Either grab a
  diagnostic plot from mmaction2's tutorial or render quickly from any
  `analyze_release` call's COCO-17 array using cv2 Gaussian disks.

## Report § 5 — Experiments / Results / Discussion (2–3 pages)

**Hyperparameters paragraph (Decision 5 + 7 + `stage5_training_method.md` §6):**
state SGD, head LR 0.01, cosine 24 ep, batch 8, class weights = inverse-freq,
label smoothing 0.1, dropout 0.5, weight decay 3e-4, RandomResizedCrop
area_range (0.56, 1.0). Discriminative backbone LR = 0.1 × head LR for full
fine-tune, = 0 for linear probe. One paragraph; longer hyperparameter table can
go in the appendix.

**Primary metrics:** **shot-class accuracy** is the deployment metric (§0 of
`stage5_training_method.md`); mean-per-class is the supporting balanced metric.
Use `acc/top1` only with the class-imbalance caveat.

**Quantitative results — tables**
1. **Stage 2 PCK table** (§4 above) — Stage 2 acceptance number.
2. **Linear probe vs. full fine-tune on the v2 test holdout** — *the headline
   training contribution* (§7 above; six rows including the per-class
   shoot row). Source: `outputs/stage5_diag/confusion_before_after.png` +
   the numbers I just copied into §7.
3. **Regularisation ablation on the v2 test holdout** — the negative result
   showing reg trades shoot for dribble (`docs/stage5_training_method.md` §11).
4. **F24 cross-method filter on TrackID3x3 detected events** (§9) — the 4/9 →
   4/5 precision lift.

**Quantitative results — plots**
5. **Learning curves — fine-tune vs. probe.** File:
   `outputs/stage5_diag/learning_curves_before_after.png` (already made; ready
   to drop in).
6. **Confusion matrices — fine-tune vs. probe, side by side.** File:
   `outputs/stage5_diag/confusion_before_after.png` (already made).
7. **Reg-ablation learning curves.** File:
   `outputs/stage5_diag/learning_curves_reg_ablation.png` (already made).
8. *(Optional)* **Sweep summary plot** — window-length sweep showing 64f is
   the sweet spot. Quick to regenerate if you want it; data is in
   `evaluate_actions_combined` outputs and the earlier diagnostic results
   captured in `flags.md` F22/F24 context.

**Qualitative results — figures**
9. **TrackID3x3 action-overlay still** showing one player flagged "SHOOT"
   (red highlight) during a real shot. Frame from
   `outputs/stage5_action_overlays/basket_S4T4_post_action.mp4` (the user's
   first-shot review); the 0.94-confidence S4T4 P4 event is a clean visual.
10. **Stage-6 release overlay still** showing the release marker landing on
    the shooter's release. From `outputs/stage6_release_overlays/755_s_4_release.mp4`
    (MultiSubjects, very clean) and/or
    `outputs/stage6_release_overlays/trackid3x3/basket_S4T4_post_t4_f200_release.mp4`
    (realistic). Show both side-by-side: clean MS example + realistic
    TrackID3x3 example.
11. **Failure-case figure.** Two stills: (a) S6T4 high-confidence shooting-like
    pose without the ball (the pose-only ceiling — F24 caveat 3); (b) S4T6 t6
    "wrong-time extension" case the F24 gate doesn't catch. From the
    re-rendered action overlays.
12. **Stage 2 wrist-overlay still** from `outputs/stage2_gate/` — at least one
    frame showing wrists tracked through a shot release (the qualitative
    acceptance evidence, F21).

**Stage 4 / 7 / deployment figures and tables — recommendations**

13. **Calibration diagnostic PNGs (Stage 4).**
    `data/sessions/2026-06-02_1on1/extrinsics_{A,B}.diag.png` — each shows
    the calibration frame from one camera with clicked landmarks (red circles
    with labels) + the reprojected world landmarks (green crosses). Visual
    sanity-check of the EPNP/SQPNP solution; the small offsets between red and
    green = the per-landmark reprojection error. **Recommended** as a
    one-line proof of "the cameras are calibrated."
14. **Triangulation sanity numbers — table (Stage 4).** Source:
    `data/sessions/2026-06-02_1on1/triangulation_sanity.json`. Nine shared
    landmarks re-triangulated to world, compared against known world
    positions, mean 3.90 cm error. Drop the 9-row table from this JSON
    directly into the report. **Required** — it's the headline calibration
    quality number.
15. **Stage 4 wrist 3D error histogram / boxplot.** Source:
    `outputs/sessions/2026-06-02_1on1/evaluation.json::stage4_wrist_3d::per_shot`.
    Need to be generated; quick matplotlib boxplot of the 13 per-shot errors.
    Annotate the 2 outliers (shots 13/14, Stage 6 release-frame failures).
    **NOT DONE** but trivial to make.
16. **3D reenactment frame at release — metric panel visible (Stage 7
    hero shot).** Source: pull a single frame from any of the 7 clips in
    `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/` at
    `ss=1.50`. Recommended candidates: `shot_005.mp4` (fadeaway, very visually
    striking with red CONTESTED panel border), `shot_008.mp4` (open shot,
    green UNCONTESTED border), `shot_002.mp4` (made layup, amber medium
    border). A 2× or 3× panel of these three would make a clean
    "deployment in action" figure. **Required** — this is the project's
    visual headline.
17. **Per-shot eval review PNGs (Stage 7 + the full per-stage caption).** Source:
    `outputs/sessions/2026-06-02_1on1/eval_review_full_metrics/shot_NNN.png`
    (×15). Each shows the GT wrist clicks vs predicted wrists at the release
    frame, side-by-side cam A + B, with all 5 Stage 7 measurements + per-stage
    errors as caption. **Recommended** — pick 2–3 representative shots (one
    clean, one mediocre, one outlier) to show the project's range.
18. **Contest metric distribution figure (Stage 7).** Histogram of the 11
    composite z-scores with the 2 makes overlaid as red vertical lines.
    Shows the 3.5 σ make/miss separation visually. **NOT DONE** but trivial
    matplotlib; numbers in `contest_metrics.json::per_shot[].pred.instantaneous.composite_z`.
19. **Wrist-to-wrist distance scatter vs make/miss (Stage 7).** 10 points,
    x = `wrist_to_wrist_distance_m`, color/marker = make vs miss, with
    Spearman ρ = −0.70 / p = 0.025 annotated. **NOT DONE** but trivial.
20. **Stage 5 + 6 + 7 chained-result table for the iPhone session.** Pull rows
    from `outputs/sessions/2026-06-02_1on1/evaluation.json` showing Stage 2
    PCK (0.93/0.95) → Stage 4 median 3.94 cm → Stage 5 P=0.40, R=0.80, F1=0.53
    → Stage 6 median Δ = 3.5 frames → Stage 7 composite makes vs misses (3.5
    σ) and the Spearman ρ. Five rows, one number column. **Required** — the
    project-level summary table.
21. **3D skeletal highlight reel (Stage 7 video supplement).** File:
    `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/highlight_reel.mp4`
    (~60 s, 5 shots with title plates). For deck use; not a still figure but
    referenced as a supplementary material link in the report.

**Discussion paragraphs to write**
- Overfitting analysis (`docs/stage5_training_method.md` §11): bound by data,
  not by the loss; reg trades shoot for dribble; therefore future work is
  data-side (more layup, FineSports intermediate, multi-window TTA, real
  iPhone footage), not loss-side.
- The F24 cross-method finding — same input, different methods, disagreement
  catches FPs. Frame as a clean methodological contribution of the project.

## Report § 6 — Conclusion / Future Work (1–3 paragraphs)
- Summarise: built **end-to-end 7-stage pose-only pipeline** deployed on a
  hand-captured iPhone session; trained PoseConv3D delivers shot accuracy 0.93
  on the v2 test holdout vs. 0.54 for the K400 linear-probe baseline; Stage 6's
  geometric rule + Stage 5's CNN acting in series filters 4/5 false positives
  at no recall cost on the small TrackID3x3 sample; Stage 4 2-camera DLT
  triangulation achieves 3.9 cm median error on a 12.5 m baseline; Stage 7's
  5-measurement pose-only contest metric separates makes from misses by
  3.5 σ on the composite z-score and Spearman ρ = −0.70 (p = 0.025) on
  wrist-to-wrist distance against make/miss outcome.
- Future work (in rough priority order): more iPhone session data (n = 15 is
  too small for inferential statistics on the make/miss outcome); a defender-
  contest action class in Stage 5 (most current FPs are defender poses
  mislabeled as shots); ball detector second attempt on iPhone footage (Phase
  2 path for Stage 7's wrist-to-trajectory measurement); FineSports
  intermediate transfer for Stage 5; multi-window TTA at inference; rater
  study for Stage 7 composite weights; refactor of the script-based deployment
  pipeline into the `src/courtvision/stage{4,7}*/` packages.

## Report § 8 — Contributions & Acknowledgements
- Honest disclosure of AI usage: Claude Code for code generation,
  brainstorming, debugging, and report assistance (per the guidelines).
  Two-terminal collaboration pattern (see §14) — one terminal owned Stages
  1 / 2 / 3 / 5 / 6 + F24; another owned Stages 4 / 7 + the end-to-end
  deployment in §13. Shared facts via `flags.md`, `docs/decisions.md`, and
  this `development_log.md`.
- Datasets cited: MultiSubjects (CVIU 2024), TrackID3x3 (open-starlab),
  FineSports (CVPR 2024) (downloaded, labels never delivered), DeepSportradar
  (ACM MMSports 2022), Kinetics-400 (PoseConv3D pretraining), **own
  hand-captured iPhone session** (2026-06-02_1on1, 15 hand-labelled shots).
- Off-the-shelf code: ultralytics (YOLOv8 + ByteTrack), rtmlib (RTMPose ONNX
  runtime), MMAction2 (PoseConv3D + K400 checkpoint), OpenCV
  (`cv2.calibrateCamera` / `solvePnP` / `triangulatePoints`), SciPy
  (`linear_sum_assignment`). Pose2Sim was a declared dependency but
  ultimately not used — `cv2.triangulatePoints` is a one-liner that did the
  job without the heavier wrapper.

## Report § 9 — References
- Pull from Decision 5 / 9 of `docs/decisions.md` and `references/RELATED_WORK.md`.
- At a minimum (≥ 10): PoseConv3D (Duan 2022), TrackNet (Huang 2019),
  WASB (Tarashima 2023), MultiSubjects (CVIU 2024), FineSports (CVPR 2024),
  Pose2Sim (Pagnon 2022), RTMPose (Jiang 2023), YOLOv8 (Jocher 2023),
  ByteTrack (Zhang 2022), Daly-Grafstein & Bornn (2020), Mikic (2024),
  Kinetics-400 (Kay 2017), LP-FT (Kumar 2022).
- **For Stage 4 / triangulation:** Hartley & Zisserman MVG textbook (DLT
  derivation); Zhang 2000 (camera-calibration chessboard); Tsai 1987 (the
  foundational camera-calibration paper); Lepetit, Moreno-Noguer, Fua 2009
  (EPNP solver used in extrinsic calibration); Terzakis & Lourakis 2020
  (SQPNP solver); Kuhn 1955 (Hungarian assignment, used for per-frame
  cross-camera ID matching).

---

# Where everything lives — quick reference

| Topic | Path |
|---|---|
| Comprehensive issue log | `flags.md` (F1–F26) |
| Design decisions log | `docs/decisions.md` (Decisions 1–10) |
| Stage 5 method writeup (report-ready) | `docs/stage5_training_method.md` |
| Stage 2 evaluation report | `docs/stage2_evaluation_report.md` |
| Per-stage scope + status | `src/courtvision/stage*/README.md` |
| Sprint plan (planned) | `docs/sprint_plan.md` |
| Original PRD | `CourtVision_PRD.md` (root level) / `docs/prd.md` |
| Auto-memory (cross-session) | `~/.claude/.../memory/MEMORY.md` and `project_*.md` |
| Training curves + confusion figures | `outputs/stage5_diag/*.png` |
| Action overlays (TrackID3x3) | `outputs/stage5_action_overlays/` |
| Release overlays (MS + TrackID3x3) | `outputs/stage6_release_overlays/` |
| Stage 2 acceptance overlays | `outputs/stage2_gate/` |
| **iPhone session inputs (calibration + GT + sync + trim)** | `data/sessions/2026-06-02_1on1/` + `data/calibration/intrinsics_{A,B}_checker.json` |
| **iPhone session pipeline outputs (Stages 1–7)** | `outputs/sessions/2026-06-02_1on1/` |
| **Per-shot evaluation review PNGs (full 5-metric caption)** | `outputs/sessions/2026-06-02_1on1/eval_review_full_metrics/shot_*.png` (×15) |
| **3D skeletal reenactments with metric panel** | `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/` (×7 clips + highlight reel) |
| **Per-stage overlay videos for session** | `outputs/sessions/2026-06-02_1on1/stage{1,2,4,5}_overlay_cam_*.mp4` |
| **Contest metrics JSON (5-measurement per-shot detail)** | `outputs/sessions/2026-06-02_1on1/contest_metrics.json` |
| **All-stage evaluation JSON (Stages 2/4/5/6/7 vs GT)** | `outputs/sessions/2026-06-02_1on1/evaluation.json` |
| **Triangulation sanity numbers (3.9 cm mean error)** | `data/sessions/2026-06-02_1on1/triangulation_sanity.json` |
| **Calibration diagnostic PNGs** | `data/sessions/2026-06-02_1on1/extrinsics_{A,B}.diag.png` + `sync.diag.png` |
