# CourtVision — report_content.md (factual dump)

**Rule of this document.** Exact numbers only, pulled from code, configs, training
logs, and saved result files. **NOT DONE** appears wherever something was never run
or measured. Nothing in this document is estimated or invented.

---

## 1. Pipeline inventory

| Component | Exact model + version | Status | Code I wrote | Code from a starter repo |
|---|---|---|---|---|
| **Player tracking** | YOLOv8m + ByteTrack | Off-the-shelf, no training | Modal wrappers in `scripts/stage1_detection_modal.py`; overlay rendering in `make_stage1_overlay`; smoke test in `tests/test_stage1_smoke.py` | `ultralytics` PyPI package — https://github.com/ultralytics/ultralytics (YOLOv8m weights auto-downloaded; ByteTrack config from `ultralytics/cfg/trackers/bytetrack.yaml`) |
| **2D pose** | RTMPose-l Halpe-26 (`rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192`, checkpoint `2abb7558`, **ONNX export**) | Off-the-shelf, no training (Phase 2 ★ RTMPose fine-tune kept as documented escalation, not run) | All of `src/courtvision/stage2_pose/rtmpose_inference.py`, `halpe_to_coco17.py`, `overlay.py`, batch eval `scripts/stage2_batch.py` | Model from MMPose model zoo — https://github.com/open-mmlab/mmpose ; inference runtime via **rtmlib** (https://github.com/Tau-J/rtmlib) using `onnxruntime` (PRD §5.2 deviation logged in `flags.md` F20) |
| **Ball tracking** | WASB-SBDT (HRNet backbone, basketball model, `wasb_basketball.pth`, 6.1 MB) | **DROPPED 2026-05-28** — qualitative review on TrackID3x3 showed coverage too partial for downstream use (`flags.md` F19) | Modal wrappers + checkpoint sanity-load + overlay in `scripts/stage3_ball_modal.py` | https://github.com/nttcom/WASB-SBDT (BMVC 2023) — checkpoint via gdown from project's `MODEL_ZOO` |
| **Shot classifier** | PoseConv3D SlowOnly-R50, K400-pretrained, **fine-tuned** on MultiSubjects | Trained — canonical model is `stage2_multisubjects.pth` (promoted v1 64f_bal); v2 ablation models are `stage2_multisubjects_64f_bal_v2.pth` (full fine-tune), `stage2_multisubjects_64f_probe_v2.pth` (linear probe), `stage2_multisubjects_64f_bal_v2_reg.pth` (stronger reg) | All training/eval infra in `scripts/stage5_modal.py`; model wrapper `src/courtvision/stage5_classifier/model/poseconv3d.py`; sliding-window inference + `_segment_peaks` + `_nms_peaks` + class gating in `infer.py`; subject-disjoint split in `split.py`; action overlay `overlay.py::make_action_overlay_video`; cross-method gate `is_verified_shot()` (F24) | Model code from **mmaction2** v1.2.0 (https://github.com/open-mmlab/mmaction2) — cloned from source per `flags.md` F20 (pip wheel missed `localizers.drn`). K400 checkpoint `poseconv3d_k400.pth` (13 MB) from MMAction2 zoo: `slowonly_r50_8xb16-u48-240e_k400-keypoint` (base config glob: `configs/skeleton/posec3d/slowonly_r50_*u48*k400-keypoint.py`) |
| **Release detector** | Custom 2D pose-only rule (`pose_forward` primary, `pose_apex` secondary; PRD §9.6 deviation logged in F23) | Custom (pure NumPy + cv2, CPU) | All of `src/courtvision/stage6_release/`: `geometry.py`, `rule_backward.py::detect_release_apex`, `rule_forward.py::detect_release_forward` + `forward_active_mask` + `first_sustained_interval`, `detect.py::analyze_release` + `is_verified_shot`, `overlay.py::make_release_overlay_video`. Modal runner `scripts/stage6_modal.py`. Tests `tests/test_stage6_smoke.py` | None — no starter code for this component |
| **3D reconstruction** | Custom **2-camera DLT** via `cv2.triangulatePoints` (linear DLT, undistorted inputs via `cv2.undistortPoints`) + per-frame **Hungarian** cross-camera ID matching via `scipy.optimize.linear_sum_assignment`. Chessboard intrinsics via `cv2.calibrateCamera`; court-landmark extrinsics via `cv2.solvePnP` with iterative landmark-dropping over {EPNP, SQPNP, IPPE, iterative} solvers. Audio-clap inter-camera sync with `fps_b/fps_a` drift correction. | **Done.** 18 002 3D poses produced for the 2026-06-02_1on1 session (10 789 frames, 180 s, 2 cameras). Triangulation sanity error 3.9 cm on 9 shared landmarks at a 12.5 m camera baseline. Median Stage 4 wrist 3D error vs hand-clicked GT = 3.94 cm (n = 13 of 15 shots). | All custom: `scripts/calibrate_intrinsics.py` (chessboard), `scripts/calibrate_extrinsics.py` (landmark click + iterative drop), `scripts/detect_claps.py` (audio sync), `scripts/run_session_pipeline.py` (orchestrator: trim + tracking + 2D pose + triangulation + sanity-check). Pose2Sim **declared but never used** — `cv2.triangulatePoints` is one line and sidesteps Pose2Sim's CLI/config wrapper. Schemas (`Pose3D`, `Pose3DSequence`, `Keypoint3D`) in `src/courtvision/schemas.py`. | OpenCV (`cv2.calibrateCamera`, `cv2.solvePnP`, `cv2.findChessboardCorners`, `cv2.undistortPoints`, `cv2.triangulatePoints`); SciPy (`linear_sum_assignment`). No starter repo for the multi-camera assignment / orchestrator code. |
| **Contest metric (Stage 7)** | Pose-only 5-measurement implementation per Decision 9 redesign + `flags.md` F25 (`wrist_to_wrist_distance_m`, `wrist_height_advantage_m`, `shooter_fade_mps`, `floor_distance_m`, `composite_score`). Closest-defender by pelvis-to-pelvis 3D; shooting-hand by higher-Z wrist; pelvis velocity by central difference ±3 frames; composite = sign-corrected sum of z-scores. | **Done.** Implementation in `scripts/compute_contest_session.py`; 11 of 15 shots evaluable on the 2026-06-02_1on1 session. Headline result: composite z-score separates makes (n = 2) from misses (n = 9) by 3.47 σ. Wrist-to-wrist distance vs miss outcome Spearman ρ = −0.70 (p = 0.025, n = 10). | All custom: `scripts/compute_contest_session.py` (5-measurement + composite + GT comparison), `scripts/review_evaluation.py` (per-shot eval-review PNGs), `scripts/render_3d_reenactment.py` (3D skeletal reenactment with metric panel). Schemas (`ContestMeasurement`, `ContestMeasurementSequence`) in `src/courtvision/schemas.py`. | None — no starter code. |

---

## 2. Datasets

### 2.1 MultiSubjects (CVIU 2024)
- **Source.** Henan University; Hugging Face snapshot. Loader: `scripts/download_datasets.py::download_multisubjects`.
- **Size.** **4 096 trimmed single-action clips**, **988 unique subjects**, 4 synchronised cameras, 25 fps, portrait **1080 × 2160** video.
- **Per-clip annotation.** COCO-17 keypoints + per-keypoint score + `total_frames` + `label ∈ {0=dribble, 1=layup, 2=shoot}`, packaged as `train.pkl` + `val.pkl` in `MultiSubjects_SA.zip` on the Modal `courtvision-data` volume.
- **Native split.** train.pkl = 3 072 clips; val.pkl = 1 024 clips. **No native test split.**
- **Combined class distribution.** dribble 1 564 (38 %) · layup 512 (13 %) · shoot 2 020 (49 %).
- **Subject-disjoint v2 split (our split, seeded, used for the before/after ablation).** Algorithm: `src/courtvision/stage5_classifier/split.py::compute_subject_disjoint_split(seed=0, ratios=(0.70, 0.15, 0.15))`. All clips from a given subject go to the same split.

  | Split | Subjects | Clips | dribble | layup | shoot |
  |---|---:|---:|---:|---:|---:|
  | train | 692 | 2 914 | 1 099 (38 %) | 369 (13 %) | 1 446 (50 %) |
  | val | 148 | 594 | 243 (41 %) | 75 (13 %) | 276 (46 %) |
  | test | 148 | 588 | 222 (38 %) | 68 (12 %) | 298 (51 %) |

  Subject-disjoint verified at split time (zero subject overlap across splits).

- **Preprocessing pipeline applied during training (mmaction2 transforms, in order):**
  ```
  SampleFrames(clip_len=64, frame_interval=1, num_clips=1)        # contiguous 64-frame window
  PoseDecode
  PoseCompact(hw_ratio=1.0, allow_imgpad=True)                    # crop to padded pose bbox
  Resize(scale=(-1, 64))                                          # short-side resize to 64
  RandomResizedCrop(area_range=(0.56, 1.0))                       # random crop 0.56-1.0 area
  Resize(scale=(56, 56), keep_ratio=False)                        # final 56x56
  Flip(flip_ratio=0.5, left_kp=[1,3,5,7,9,11,13,15],
       right_kp=[2,4,6,8,10,12,14,16])                            # horizontal flip with L/R kp swap
  GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True,
                     with_limb=False)                             # render 17 Gaussian heatmaps
  FormatShape(input_format="NCTHW_Heatmap")                       # → [B, 17, 64, 56, 56]
  PackActionInputs
  ```
- **Preprocessing pipeline applied at val/test (deterministic, single centered window):**
  ```
  SampleFrames(clip_len=64, frame_interval=1, num_clips=1, test_mode=True)
  PoseDecode
  PoseCompact(hw_ratio=1.0, allow_imgpad=True)
  Resize(scale=(56, 56), keep_ratio=False)
  GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False)
  FormatShape(input_format="NCTHW_Heatmap")
  PackActionInputs
  ```
- **Augmentation summary.** RandomResizedCrop area_range=(0.56, 1.0), 50 % horizontal flip with COCO-17 L/R keypoint swap. No temporal augmentation beyond the random start chosen by `SampleFrames`. No keypoint jitter.
- **Citation.** *MultiSubjects: a multi-subject video dataset for single-person basketball action recognition from basketball gym (CVIU 2024)*.

### 2.2 TrackID3x3 (open-starlab)
- **Source.** https://github.com/open-starlab/TrackID3x3 ; clips fetched via `gdown` from the project's public Google Drive (`scripts/download_datasets.py::download_trackid3x3`).
- **Used for.** Stage 2 pose acceptance evaluation (Phase C gate); Stage 5 action-recognition qualitative review; Stage 6 release-detection realistic application.
- **Stage 2 PCK eval subset:** all **13 Indoor pose-GT clips** (Outdoor = 1 clip + Drone = ~500 aerial segments excluded as off-domain / unverified schema, per `flags.md` F5).
- **GT keypoint schema (10 keypoints, inferred via `flags.md` F5):** 0 head/nose, 1 R-shoulder, 2 L-shoulder, 3 R-elbow, 4 L-elbow, 5 R-wrist, 6 L-wrist, 7 hip/pelvis center, 8 R-ankle, 9 L-ankle. GT is **per-frame, jersey-color-keyed**, not track-ID-keyed; matched to Stage 1 detections by spatial proximity. **GT is model-derived (not hand-labelled).**
- **No training on this data.** Used as test data only.

### 2.3 Kinetics-400 (PoseConv3D pretraining checkpoint only)
- **Source.** MMAction2 model zoo: `slowonly_r50_8xb16-u48-240e_k400-keypoint` (file `poseconv3d_k400.pth`, **13 MB**).
- **Used as.** Backbone initialisation for Stage 5; the 400-class K400 head is dropped and replaced with our 4-class head. Not retrained on K400.
- **Citation.** Kay et al., *The Kinetics Human Action Video Dataset*, 2017.

### 2.4 FineSports (CVPR 2024)
- **Status.** NOT DONE for training. HF frames-zip downloaded to the volume; **`FineSports-GT.pkl` label file pending email request — never delivered**. Path C 4-stage transfer pipeline (PRD §9.5 / Decision 7) reduced to Path A (Kinetics → MultiSubjects) as a result.

### 2.5 DeepSportradar Basketball Instants + Roboflow `ownprojects/basketball-w2xcw`
- **Status.** Substituted for the verified-404 "U Arizona Roboflow basketball" dataset (`flags.md` F12). NOT DONE — the ball detector (Stage 3) was abandoned before a mAP measurement was taken on these.

### 2.6 Own iPhone footage — session `2026-06-02_1on1`

- **Session.** 1 hand-captured 1-on-1 half-court basketball session, **2 iPhones** on tripods at opposite ends of the half-court, AE/AF locked per Decision 10. 4 hand-claps at the start for inter-camera audio sync. ~5 minutes of continuous play.
- **Per camera (trimmed clip):**
  - Camera A: **1080 × 1920** portrait, 59.940 fps (NTSC drop-frame), 10 789 frames = 180.0 s. Source: `Datasets/iPhone/A_play.MOV` (original) → `data/sessions/2026-06-02_1on1/trimmed/A_trimmed.mp4`.
  - Camera B: **1080 × 1920** portrait, 60.000 fps, 10 795 frames = 180.0 s. Source: `Datasets/iPhone/B_play.MOV` → `data/sessions/2026-06-02_1on1/trimmed/B_trimmed.mp4`. **Drift slope:** 1.0010 (60.000 / 59.940), applied in the trim step so cam B retains the correct frame-correspondence to A across the full 180 s.
- **Ground-truth labeling.**
  - **15 shots** hand-labelled across the 180-second clip via custom matplotlib UI (`scripts/label_shots.py` + `scripts/review_shots.py`).
  - Per shot: `release_frame_a`, `release_frame_b`, `shot_type ∈ {jump_shot, layup}`, `make ∈ {True, False}`, `uncontested ∈ {True, False}`, plus pixel-coordinate clicks for `shooter_wrist_a/b`, `defender_wrist1_a/b`, `defender_wrist2_a/b` (3 wrist clicks × 2 cameras = 6 per shot).
  - **Distribution:** 3 makes / 12 misses; 14 jump shots / 1 layup; 0 uncontested. (Shots skew toward contested jump shots because the players were specifically working on contested-shot mechanics.)
  - Labels chunked into `data/sessions/2026-06-02_1on1/shots_gt_part{1..5}.json` (one chunk per labeling session), merged to canonical `data/sessions/2026-06-02_1on1/shots_gt.json`.
- **Audio-clap sync.** 4 of 4 claps detected on both cameras (`data/sessions/2026-06-02_1on1/sync.json`). Per-clap offset standard deviation = 25.8 ms (≈ 1.5 frames). Max disagreement across the 4 claps = 64.3 ms (1 frame at 60 fps).
- **No training on this data.** Used as the evaluation dataset for Stages 1–7 end-to-end, with hand-clicked wrist GT as the per-shot ground truth.

### 2.7 Calibration captures

- **Intrinsic calibration — chessboards per phone (one-time, off-court).**
  - Chessboard pattern: 9 × 6 inner corners, 30 mm squares (printed on letter paper).
  - Camera A: 97 chessboard frames captured + processed, **RMS reprojection error = 2.22 px** (`data/calibration/intrinsics_A_checker.json`).
  - Camera B: 75 chessboard frames captured + processed, **RMS reprojection error = 1.09 px** (`data/calibration/intrinsics_B_checker.json`).
  - Solved via `cv2.calibrateCamera` after `cv2.findChessboardCorners`. K + 5-parameter distortion vector saved.

- **Extrinsic calibration — court-landmark clicks per session per camera.**
  - 17 court landmarks defined in world coordinates from high-school NCAA spec: backboard corners (TL/TR/BL/BR), free-throw lane corners (TL/TR/BL/BR), inner-square corners (TL/TR/BL/BR), free-throw line center, rim center, free-throw arc top, 3-point arc top. World frame is rim-centered (basket at origin xy, floor at z = 0, rim at z = 3.05 m).
  - User clicks each visible landmark in a calibration frame; solver iterates over {EPNP, SQPNP, IPPE, iterative} × greedy outlier dropping; keeps the best by mean reprojection error.
  - Camera A: **EPNP**, 9 of 17 landmarks accepted, mean reproj 4.89 px, max 9.53 px (`data/sessions/2026-06-02_1on1/extrinsics_A.json`). Dropped: RIM, FT_LANE corners (×3), SEMI_TOP, ARC_TOP.
  - Camera B: **SQPNP**, 12 of 17 landmarks accepted, mean reproj 3.71 px, max 7.77 px (`data/sessions/2026-06-02_1on1/extrinsics_B.json`). Dropped: ARC_TOP, RIM.
  - Diagnostic overlay PNGs saved at `data/sessions/2026-06-02_1on1/extrinsics_{A,B}.diag.png`.

- **Triangulation sanity check.** 9 court landmarks visible from both cameras (BB corners ×4, FT center, inner-square corners ×4) re-triangulated via `cv2.triangulatePoints`. World positions are known. **Mean 3D error 3.90 cm, median 3.44 cm, max 6.79 cm** at a 12.51 m camera baseline. Saved at `data/sessions/2026-06-02_1on1/triangulation_sanity.json`.

---

## 3. Methods

### 3.1 PoseConv3D shot classifier (Stage 5)

**Architecture.** SlowOnly-R50 backbone + I3DHead.
- **Input tensor shape:** `[B=8, C=17, T=64, H=56, W=56]` — per-window heatmap volume.
- **17 channels** = COCO-17 keypoints (nose, L/R eye, L/R ear, L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle).
- **Heatmap rendering.** Each keypoint rendered as 2D Gaussian, σ = 0.6 px (at 56×56 resolution), with intensity modulated by the keypoint's confidence score (`use_score=True`), `with_kp=True`, `with_limb=False`.
- **Output:** `[B, 4]` logits → softmax → P(dribble), P(layup), P(shoot), P(background).
- **Background class** is the 4th logit; it exists architecturally but **has 0 training samples** in our recipe (MultiSubjects clips are all action). The slot is preserved for the future Path C / iPhone-LP-FT step where mined negatives will supervise it.

**Keypoint schemas.**
- **Stage 2 produces Halpe-26.** Order: 0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear, 5 L-shoulder, 6 R-shoulder, 7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip, 13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle, 17 head, 18 neck, 19 hip-centre, 20–23 toes, 24–25 heels.
- **Stage 5 consumes the COCO-17 subset** — the first 17 indices of Halpe-26 (which are COCO-17 by construction). Conversion: `src/courtvision/stage2_pose/halpe_to_coco17.py::halpe26_to_coco17`.

**Loss.** mmaction2 `CrossEntropyLoss` with explicit per-class weights + label smoothing.
- Inverse-frequency class weights: `w_c = n_total_real / (3 · n_c)`. **For the v1 model** (train counts dribble 1174 / layup 391 / shoot 1507; n_total = 3072): `w_dribble = 0.872`, `w_layup = 2.619`, `w_shoot = 0.679`, `w_background = 1.000`. **For the v2 ablation models** (train counts dribble 1099 / layup 369 / shoot 1446; n_total = 2914): `w_dribble = 0.884`, `w_layup = 2.633`, `w_shoot = 0.672`, `w_background = 1.000`.
- **Label smoothing ε = 0.1** applied by `I3DHead.label_smooth_eps`.

**Sliding-window inference (deployment).** `clip_len = 64`, `stride = 8`, `score_threshold = 0.6` (lowered from 0.75 to feed Stage 6's verification — see F24), `merge_gap = 24`.
- Detection signal: `p_shoot` only (changed from `p_shoot + p_layup` to focus on jump shots per user priority).
- Peak picking: `_segment_peaks(scores, centers, threshold=0.6, merge_gap=24)` — merges contiguous above-threshold windows into one event at the segment peak. `_nms_peaks` kept exported for compatibility tests but no longer the primary picker.

### 3.2 Release detector (Stage 6 — 2D pose-only)

Two methods, identical signature, both returning a `ReleaseFrame`. `pose_forward` is canonical primary, `pose_apex` is secondary cross-check.

**`pose_apex` (secondary):** within the search window `[lo, hi)`, dominant wrist = side reaching the highest point (min image-y). Release = `argmin_y(wrist)` among confident frames (wrist score ≥ 0.2). Confidence flag `"high"` iff wrist score ≥ 0.3 AND elbow angle ≥ 150° at the chosen frame.

**`pose_forward` (primary):** dominant-wrist forward-active mask = `(wrist_y < shoulder_y) ∧ (elbow_angle ≥ 160°) ∧ (wrist_score ≥ 0.2)`, computed per frame. `first_sustained_interval(mask, lo, hi, min_run=2, gap_tol=2)` returns `[onset, end)` of the first run of True with total active ≥ `min_run`, allowing gaps ≤ `gap_tol` between True frames (so a 1-frame keypoint flicker doesn't push the onset late). **Release = `interval.onset + release_offset`**, clamped to the search window. `release_offset = -1` (user-calibrated on 36 MultiSubjects shot-clip overlays).

**Elbow angle computation.** `angle_at_b(a, b, c) = arccos(((a-b) · (c-b)) / (|a-b|·|c-b| + 1e-6))` in degrees. For each side: a = shoulder, b = elbow, c = wrist.

**Dominant wrist selection.** The side with the smaller min(wrist_y) over the search window (among confident frames) — i.e., the wrist that reaches highest.

**Cross-method gate (F24).** `is_verified_shot(rf: ReleaseFrame) -> bool` returns `rf.confidence_flag == "high"`. Used to filter Stage 5 events at threshold 0.6; events where the `pose_forward` rule found no sustained extension fall back to apex and tag `"low"` — interpreted as likely false positives.

### 3.3 3D triangulation (Stage 4 — custom 2-camera DLT)

- **Per-camera intrinsic calibration.** `cv2.calibrateCamera` on chessboard frames (one-time per phone). Output: K (3×3) + dist (5 params).
- **Per-session extrinsic calibration.** User clicks N ≤ 17 court landmarks in a calibration frame per camera; world coordinates known from NCAA spec. Solver tries every combination of {EPNP, SQPNP, IPPE, iterative} × {greedy outlier drop in [0, 5]}; keeps the one with lowest mean reprojection error. Output per camera: R (3×3), t (3×1), camera position in world.
- **Audio-clap inter-camera sync.** Spectral-flux clap-onset detection per camera; offset_b_minus_a = mean(per-clap offset). Drift slope = `fps_b / fps_a` (60.000 / 59.940 = 1.001). Frame correspondence: `B_frame(A_frame) = round(A_frame * fps_b / fps_a)`.
- **Cross-camera ID matching (per frame).** Stages 1–2 are run independently per camera with no cross-camera awareness; ByteTrack reissues ~10 track IDs per camera over a 180-s clip, so a "two stable tracks" assumption fails. At each frame f: enumerate all alive Pose2D tracks on cam A and cam B. Cost matrix C[i, j] = 3D L2 reprojection error of triangulating (A_track_i, B_track_j) pelvis. Solve via `scipy.optimize.linear_sum_assignment` (Hungarian). Accept pairs with cost below a threshold; reject the rest.
- **Triangulation per accepted pair.** For each of the 26 Halpe-26 keypoints: 2D undistort via `cv2.undistortPoints(pt, K, dist, P=K)`, then `cv2.triangulatePoints(P_a, P_b, ua, ub)` returns the 3D world point (homogeneous → 3D divide). `Pose3D.keypoints[i].n_cameras = 2` (triangulated) or 0 (not visible to both); the latter is filtered out at consumer time.
- **No bundle adjustment.** Considered but skipped — linear DLT is already 3.9 cm mean error on shared landmarks at 12.5 m baseline; refinement is sub-cm-class noise vs. our downstream metric resolution.

### 3.4 Contest metric (Stage 7 — pose-only 5-measurement)

Per the Decision 9 redesign (F25, 2026-06-02). Implementation in `scripts/compute_contest_session.py`.

- **Selection rules.**
  - **Shooter** = the track whose dominant wrist reaches highest world-Z in the ±32-frame window around Stage 5's detected event (Stage 6 inherits this). If multiple candidates, the global max-Z winner.
  - **Closest defender** = the non-shooter track whose 3D pelvis (Halpe-26 index 19 = "hip-center") is closest to the shooter's pelvis at the release frame.
  - **Shooting hand** = the higher-Z of the shooter's two wrists at the release frame. (Halpe-26 indices: 9 = L wrist, 10 = R wrist.)

- **The five measurements at the release frame.**
  1. `wrist_to_wrist_distance_m` = `min over defender wrists w of ||w - shooter_shooting_wrist||₂` in 3D.
  2. `wrist_height_advantage_m` = `max(defender_L_wrist_z, defender_R_wrist_z) - shooter_shooting_wrist_z`. Positive = defender's hand at or above shooter's release point.
  3. `shooter_fade_mps` = `sqrt(max(0, -v_forward)² + v_lateral²)`, where `forward = unit(rim_xy - shooter_pelvis_xy)`, `lateral = R₉₀(forward)`, and `v` is the central-difference pelvis velocity over ±3 frames. Rim center at world (0, 0.38, 3.05) m.
  4. `floor_distance_m` = `||defender_pelvis_xy - shooter_pelvis_xy||₂` (2D in court plane).
  5. `composite_score` = `-z(w-w) + z(hgt) + z(fade)`, where z(·) is the z-score across the n evaluable shots in the session. Sign convention: bigger = more contested.

- **Smoothing.** Each measurement reported both **instantaneous** (single release frame) and **5-frame mean** (±2 frames around release). Implementation: average each relevant keypoint over the window, then apply the same formula.

- **Uncontested flag** triggers if `floor_distance_m > 3.0 m` at the release frame (Decision 9 threshold).

- **GT triangulation for evaluation.** GT wrist clicks per camera are 3D-triangulated at compute time via the same `cv2.triangulatePoints` path. GT wrist-to-wrist = `min_w ||w_gt_def - w_gt_shooter||₂`; GT height advantage = `max_w z(w_gt_def) - z(w_gt_shooter)`.

---

## 4. Training details

Only Stage 5 (PoseConv3D) was trained. Every other component is off-the-shelf or untrained.

**Common training recipe (all Stage 5 runs):**
- **Hardware.** NVIDIA A10G (24 GB) on Modal. One GPU per run.
- **Optimizer.** SGD (mmaction2 K400 base config default), momentum 0.9.
- **Head learning rate.** 0.01.
- **Backbone learning rate multiplier.** Set via `optim_wrapper.paramwise_cfg = {"custom_keys": {"backbone": {"lr_mult": <m>}}}`. m = 0.1 (full fine-tune) or 0.0 (linear probe, plus `requires_grad=False` on all backbone parameters).
- **LR schedule.** Cosine annealing, `T_max = 24` epochs, `eta_min = 0`.
- **Batch size.** 8.
- **Epochs.** 24.
- **Random seed.** `cfg.randomness.seed = 0` (set explicitly on v2 ablation runs for reproducibility).
- **Weight decay.** 3e-4 (baseline) or 1e-3 (stronger-reg ablation).
- **Head dropout** (`I3DHead.dropout_ratio`). 0.5 (baseline) or 0.7 (stronger-reg ablation).
- **Label smoothing ε** (`I3DHead.label_smooth_eps`). 0.1 (baseline) or 0.15 (stronger-reg ablation).
- **RandomResizedCrop area_range minimum.** 0.56 (baseline) or 0.40 (stronger-reg ablation).
- **Early-stopping criterion.** `CheckpointHook(interval=1, max_keep_ckpts=2, save_best="acc/top1", rule="greater")` selects best epoch by val top-1 accuracy. **NB:** this is the *noisy* metric under class imbalance; mean-per-class is more honest but was not the selector. For the v1 model (canonical), best epoch by both top1 and mean1 happened to coincide (epoch 10). For the v2 full fine-tune, best by top1 happened at epoch 2 (which was also best by mean1).
- **Resume logic.** Per-epoch checkpointing to Modal Volume; on preemption, resumes from `last_checkpoint` if it has a `meta` field; otherwise wipes the work_dir and starts fresh from the K400 backbone (`flags.md` F22-region notes).

**Runs trained (all on the v2 subject-disjoint split unless noted):**

| Tag | clip_len | freeze_backbone | reg knobs | Best epoch | Saved checkpoint |
|---|---:|---|---|---:|---|
| `64f_bal` (v1, canonical) | 64 | False | baseline | 10 | `/artifacts/checkpoints/stage5/stage2_multisubjects.pth` (promoted from `..._64f_bal.pth`) |
| `64f_bal_v2` (full fine-tune, v2 split) | 64 | False | baseline | 2 | `/artifacts/checkpoints/stage5/stage2_multisubjects_64f_bal_v2.pth` |
| `64f_probe_v2` (linear probe, v2 split) | 64 | True | baseline | 13 | `/artifacts/checkpoints/stage5/stage2_multisubjects_64f_probe_v2.pth` |
| `64f_bal_v2_reg` (stronger reg, v2 split) | 64 | False | wd 1e-3, dropout 0.7, eps 0.15, crop 0.40 | 16 | `/artifacts/checkpoints/stage5/stage2_multisubjects_64f_bal_v2_reg.pth` |

The previously-trained 32-frame `UniformSampleFrames` model was diagnosed (78 % uniform vs 57 % contiguous; `flags.md` F22 / sampling-mismatch diagnostic) and superseded by the contiguous-window models above. The 32f checkpoint is archived as `stage2_multisubjects_32f.pth`.

---

## 5. Results

### 5.1 Stage 5 shot classifier — held-out v2 test (588 clips, 148 unseen subjects)

**Evaluation protocol.** `eval_on_test` in `scripts/stage5_modal.py`. For each test clip: one centered contiguous 64-frame window per the val pipeline (§2.1) → forward pass → `argmax` over 4 logits. Per-class accuracy, overall accuracy, mean-per-class accuracy, and row-normalised 3×3 confusion matrix reported.

| Metric | Linear probe `64f_probe_v2` | Full fine-tune `64f_bal_v2` | Stronger reg `64f_bal_v2_reg` |
|---|---:|---:|---:|
| Dribble accuracy (n = 222) | **0.968** | 0.716 | 0.964 |
| Layup accuracy (n = 68) | 0.353 | 0.529 | 0.544 |
| **Shoot accuracy (n = 298)** | 0.537 | **0.930** | 0.564 |
| Overall accuracy | 0.679 | **0.803** | 0.713 |
| **Mean per-class** | 0.619 | **0.725** | 0.691 |

**Δ (full fine-tune − linear probe), the headline training contribution:**
- Dribble: −0.252
- Layup: +0.176
- **Shoot: +0.393**
- Overall: +0.124
- **Mean-per-class: +0.106**

**Confusion matrices on the v2 test holdout (rows = true, cols = predicted, row-normalised):**

**Linear probe `64f_probe_v2`:**
| true \ pred | dribble | layup | shoot | bg |
|---|---:|---:|---:|---:|
| dribble | 0.968 | 0.023 | 0.009 | 0.000 |
| layup | 0.588 | 0.353 | 0.059 | 0.000 |
| shoot | 0.433 | 0.030 | 0.537 | 0.000 |

**Full fine-tune `64f_bal_v2`:**
| true \ pred | dribble | layup | shoot | bg |
|---|---:|---:|---:|---:|
| dribble | 0.716 | 0.027 | 0.257 | 0.000 |
| layup | 0.088 | 0.529 | 0.382 | 0.000 |
| shoot | 0.030 | 0.040 | 0.930 | 0.000 |

**Stronger reg `64f_bal_v2_reg`:**
| true \ pred | dribble | layup | shoot | bg |
|---|---:|---:|---:|---:|
| dribble | 0.964 | 0.032 | 0.005 | 0.000 |
| layup | 0.441 | 0.544 | 0.015 | 0.000 |
| shoot | 0.393 | 0.044 | 0.564 | 0.000 |

### 5.2 Stage 5 — v1 canonical model contiguous-window diagnostic

Diagnostic on 120 balanced clips (40 per class) drawn from MultiSubjects val (v1 native, *not* the v2 holdout). Contiguous sliding windows at clip_len = 64, stride = 8. Source: `diagnose_sampling` in `scripts/stage5_modal.py`.

| Metric | Value |
|---|---:|
| Uniform top-1 accuracy (training regime) | 0.775 |
| Contiguous per-window accuracy (deployment regime) | **0.706** |
| Contiguous majority-vote per-clip accuracy | 0.758 |
| False-shoot rate on non-shot windows | 0.231 |

Per-class contiguous per-window accuracy: dribble 0.797, layup 0.485, shoot **0.952**.

Contiguous-window confusion (true rows, pred cols, row-normalised):
| true \ pred | dribble | layup | shoot | bg |
|---|---:|---:|---:|---:|
| dribble | 0.797 | 0.000 | 0.203 | 0.000 |
| layup | 0.239 | 0.485 | 0.276 | 0.000 |
| shoot | 0.048 | 0.000 | 0.952 | 0.000 |

### 5.3 Stage 2 pose — PCK@0.05

**TrackID3x3 Indoor (n = 13 clips, model-derived GT, ~12 px tolerance):**
- **Wrist PCK@0.05 = 0.655** ± 0.043 (mean 0.648 across 13 clips). Below the 0.85 gate; accepted via documented deviation per PRD §9.2 qualitative acceptance (`flags.md` F21).
- All-keypoint PCK@0.05 = 0.781.
- Per-landmark: shoulder 0.941 · elbow 0.727 · ankle 0.845 · head 0.675 · hip 0.802 · wrist 0.655.

**MultiSubjects val (n = 300 stratified-random clips: 100 dribble + 100 layup + 100 shoot; cross-dataset corroboration per `flags.md` F26):**
- Total keypoints scored: **362 695**.
- **Wrist PCK@0.05 (mean L + R) = 0.680.**
- Overall (all 17 COCO keypoints) PCK@0.05 = **0.841**.
- Per-landmark: nose 0.91 · shoulder 0.89 / 0.89 · elbow 0.79 / 0.77 · wrist 0.69 / 0.67 · hip 0.86 / 0.86 · knee 0.88 / 0.87 · ankle 0.87 / 0.86.
- Per-action overall PCK (per-clip mean): dribble 0.844 ± 0.157 · layup 0.799 ± 0.116 · shoot 0.798 ± 0.187.
- 3 of 300 clips scored 0 (Stage 1 missed the player; negligible on aggregate).

Eval JSONs: `outputs/stage2_gate/eval_stage2_trackid3x3_all13.json` and `outputs/stage2_gate/eval_stage2_multisubjects_300.json`.

### 5.4 Stage 1 player tracking — mAP / IDF1

**NOT DONE.** Stage 1 was consumed off-the-shelf per Decision 1; no held-out mAP / IDF1 metric was reported this sprint. Acceptance was qualitative on TrackID3x3 indoor overlays. Track-persistence backlog (6 deferred tuning layers, C2 highest-leverage) logged in `flags.md` F15.

### 5.5 Stage 3 ball detection — recall

**NOT DONE.** Dropped via `flags.md` F19 before a recall number was measured on a curated test set. Qualitative review on TrackID3x3 indoor clips showed coverage too partial for downstream use.

### 5.6 Stage 4 — calibration + triangulation quality on session 2026-06-02_1on1

**Decision 4 target was < 5 px mean reprojection error against a held-out 3rd camera.** The 3rd camera never materialised; the surrogate is the **triangulation sanity check** below (re-triangulate landmarks visible to both cameras → compare to known world positions).

**Per-camera intrinsics (chessboard, off-court, one-time):**

| Camera | Chessboard frames | RMS reproj (px) | Source JSON |
|---|---:|---:|---|
| A | 97 | **2.22** | `data/calibration/intrinsics_A_checker.json` |
| B | 75 | **1.09** | `data/calibration/intrinsics_B_checker.json` |

Both below the 5 px Decision 4 target.

**Per-session extrinsics (court-landmark click, per session):**

| Camera | Solver | Landmarks used (of 17) | Mean reproj (px) | Max reproj (px) | Camera position (m) |
|---|---|---:|---:|---:|---|
| A | EPNP | 9 | **4.89** | 9.53 | (−4.69, 7.07, 2.39) |
| B | SQPNP | 12 | **3.71** | 7.77 | ( 7.32, 10.54, 2.13) |

Both below the 5 px Decision 4 target on the mean metric. Camera baseline (3D distance between camera centers) = **12.51 m**.

**Triangulation sanity check** (re-triangulate landmarks visible to both cameras → compare to known world positions). Nine shared landmarks: BB corners ×4, FT center, inner-square corners ×4. Source: `data/sessions/2026-06-02_1on1/triangulation_sanity.json`.

| Landmark | 3D error (cm) |
|---|---:|
| BB_BL | 3.44 |
| BB_BR | 4.33 |
| BB_TL | 4.88 |
| BB_TR | 3.14 |
| FT_CENTER | 2.69 |
| ISQ_BL | 3.33 |
| ISQ_BR | 0.94 |
| ISQ_TL | 6.79 |
| ISQ_TR | 5.54 |
| **Mean** | **3.90** |
| Median | 3.44 |
| Max | 6.79 |

**End-to-end Stage 4 wrist 3D error vs hand-labelled GT** (session 2026-06-02_1on1, n = 13 of 15 shots; 2 shots had ≥ 1 missing GT click). Stage 4 triangulates the shooter's shooting-wrist at the Stage-6-predicted release frame; GT is the 3D-triangulated hand-click pair at the GT release frame.

| Metric | cm |
|---|---:|
| n evaluated | 13 |
| n skipped (missing GT) | 2 |
| **Median error** | **3.94** |
| Mean error | 20.84 |
| P90 error | 42.66 |
| Max error | 177.30 |

The median 3.94 cm is the headline 3D accuracy result. The mean is inflated by 2 outlier shots where Stage 6 picks a release frame 36–44 frames away from GT (the shooter-identification picks a defender for those shots), so Stage 4 is evaluated at the wrong moment — Stage 4 itself is healthy, Stage 6 is the cause.

### 5.7 Stage 6 release-frame accuracy

**No Δframes-vs-GT metric was computed on MultiSubjects or TrackID3x3** (MultiSubjects does not ship per-frame release labels; TrackID3x3 has no shot/release annotations). All MS / TrackID3x3 evaluation was qualitative video review.
- **MultiSubjects, 36 shot clips with overlay rendering.** User review concluded `pose_apex` is systematically late (wrist keeps rising on follow-through), `pose_forward` is closer to the true release frame, and **release ≈ interval onset − 1 frame** → `release_offset = -1` baked in.
- **TrackID3x3, 9 Stage-5-detected shot events** (all events with `p_shoot ≥ 0.6`): release detector ran with `window_half = 50` frames around each Stage-5 peak. Results per event are in §5.8 below.

**Quantitative Δframes-vs-GT on the iPhone session (2026-06-02_1on1, n = 12 TP events).** This is the first run with hand-labelled per-shot GT release frames; the Stage 6 implementation was extended to 3D pose (using triangulated keypoints rather than 2D) for this deployment.

| Metric | frames | seconds @ 60 fps |
|---|---:|---:|
| n TP events | 12 | — |
| Mean Δframes (signed) | −3.92 | −0.065 |
| Median Δframes (signed) | +3.50 | +0.058 |
| Mean \|Δframes\| | **11.58** | **0.193** |
| P90 \|Δframes\| | 33.70 | 0.562 |
| Max \|Δframes\| | 44 | 0.733 |

The median 3.5 frames (0.06 s) is essentially perfect for downstream Stage 7 use; the mean / P90 are pulled up by 2 outlier shots (13 and 14) where the search window picks a defender as the shooter.

### 5.8 F24 cross-method filter on TrackID3x3 detected shots (Stage 5 + Stage 6)

**Sample:** all 9 events at threshold ≥ 0.6 from `evaluate_actions` on the 13 TrackID3x3 indoor clips, with the user manually labelling real-vs-FP after watching the action-overlay videos.

| Stage 5 conf | clip · track · peak | Stage 6 `forward_conf` | fwd frame | apex frame | User label |
|---|---|---|---:|---:|---|
| 0.94 | S4T4 · t4 · f200 | high (186 ≠ 213) | 186 | 213 | real shot ✓ |
| 0.92 | S5T7 · t3 · f112 | high (122 ≠ 126) | 122 | 126 | real shot ✓ |
| 0.90 | S5T5 · t4 · f112 | high (123 ≠ 127) | 123 | 127 | real shot ✓ |
| 0.82 | S6T4 · t4 · f88 | **low** (65 == 65) | 65 | 65 | FP |
| 0.73 | S6T4 · t9 · f237 | **low** (241 == 241) | 241 | 241 | FP |
| 0.67 | S6T1 · t6 · f96 | **low** (142 == 142) | 142 | 142 | FP |
| 0.66 | S4T6 · t3 · f192 | high (194 ≠ 198) | 194 | 198 | real shot ✓ |
| 0.65 | S4T6 · t6 · f176 | **high** (135 ≠ 214) | 135 | 214 | FP (filter miss) |
| 0.63 | S4T6 · t8 · f148 | **low** (104 == 104) | 104 | 104 | FP (bg player) |

**Pipeline comparison on this sample:**

| | Stage 5 alone @ 0.75 | Stage 5 @ 0.6 + Stage 6 `forward_conf=="high"` gate |
|---|---|---|
| TP | 3 (S4T4, S5T5, S5T7) | 4 (adds S4T6 t3) |
| FP | 1 (S6T4 0.82) | 1 (S4T6 t6, the filter miss) |
| FN | 1 (S4T6 t3 0.66 dropped by threshold) | 0 |
| **Precision** | **0.750** | **0.800** |
| **Recall** | **0.750** | **1.000** |

Reproduces via `scripts/stage5_modal.py::eval_combined`. Sample-size caveat: 9 events, 13 short clips, single human labeller.

### 5.9 Window-length sweep (deployment-regime sweet spot determination)

Single 32f-trained checkpoint evaluated at multiple inference window lengths on the same 120 balanced val clips. Contiguous per-window accuracy:

| Window | Window duration | Shoot | Layup | Dribble | Overall |
|---:|---:|---:|---:|---:|---:|
| 32 frames | 1.28 s | 0.86 | 0.23 | 0.80 | 0.57 |
| 48 frames | 1.92 s | 0.88 | 0.29 | 0.80 | 0.61 |
| **64 frames** | **2.56 s** | **0.95** | **0.28** | **0.80** | **0.64** |
| 80 frames | 3.20 s | 0.97 | 0.26 | 0.81 | 0.65 |
| 96 frames | 3.84 s | 0.97 | 0.19 | 0.78 | 0.61 |
| 128 frames | 5.12 s | 1.00 | 0.14 | 0.77 | 0.62 |

Past ~80f the apparent shoot gains are artifactual (window engulfs the entire short trimmed clip, equivalent to whole-clip evaluation). Layup degrades monotonically beyond 64f. **64f selected as the deployment window.**

### 5.10 Stage 5 + Stage 7 — iPhone session results (2026-06-02_1on1)

**Sample.** 15 hand-labelled shots (3 makes, 12 misses, 14 jump shots, 1 layup, 0 uncontested). Source for all numbers: `outputs/sessions/2026-06-02_1on1/evaluation.json` + `outputs/sessions/2026-06-02_1on1/contest_metrics.json`.

**Stage 5 shot detection vs the 15 GT shots.**

| Metric | Value |
|---|---:|
| Shoot threshold | 0.75 |
| Tolerance (frames around GT) | 30 |
| n GT shots | 15 |
| n predicted events (after cam-A/cam-B fusion) | 30 |
| True positives | 12 |
| False positives | 18 |
| False negatives | 3 |
| **Precision** | **0.400** |
| **Recall** | **0.800** |
| F1 | 0.533 |

**Note on Stage 5 FPs.** Most of the 18 FPs are *defender contest poses* on real shots — the classifier was trained on `{dribble, layup, shoot, background}` with no defender-pose class, so a player making a contesting arm-extension looks shot-like to the CNN. The cross-method gate (F24) would help; not applied to this session's eval as written.

**Stage 2 wrist accuracy at the GT release frames (vs hand-clicked GT, n = 15 shots).** Pixel error of the closest predicted wrist (Stage 2) to the hand-clicked GT wrist. **This is hand-labelled GT, not the model-derived TrackID3x3 GT in §5.3 — the comparison is more honest.**

| Camera | n clicks | Wrist PCK@0.05 | Mean err (px) | Median err (px) | Max err (px) |
|---|---:|---:|---:|---:|---:|
| A | 44 | **0.932** | 6.92 | 2.79 | 78.85 |
| B | 38 | **0.947** | 12.32 | 5.41 | 249.16 |

At hand-labelled GT, the Stage 2 wrist PCK is **0.93–0.95** — far above the 0.65 number against model-derived GT on TrackID3x3 (§5.3 / F18). Confirms the F18 caveat: model-derived GT *under-reports* RTMPose accuracy when both estimators converge on the same systematic error.

**Stage 7 — five-measurement contest metric (per-shot details in `contest_metrics.json`).**

| Measurement | n | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|---:|
| `wrist_to_wrist_distance_m` | 10 | 0.870 | 0.782 | 0.320 | 1.610 | 0.335 |
| `wrist_height_advantage_m` | 10 | −0.357 | −0.123 | −1.469 | 0.280 | 0.501 |
| `shooter_fade_mps` | 11 | 1.072 | 0.731 | 0.015 | 4.365 | 1.229 |
| `floor_distance_m` | 10 | 1.290 | 1.321 | 0.991 | 1.600 | 0.173 |
| `composite_z` | 11 | 0.000 | 0.002 | −5.199 | 4.827 | 2.371 |
| Uncontested shots (floor dist > 3 m) | 0 | — | — | — | — | — |

Reasons for n < 15: shots 1, 3, 9, 10 had no Stage 6 release prediction (Stage 5 missed those shots OR Stage 6 search window cut off); shot 12 had a release prediction but the defender pose was missing at that frame (only `shooter_fade_mps` is computable without a defender, which is why n = 11 for that field and n = 10 for the others).

**Make / miss separation — the headline result.**

| Measurement | Makes mean (n = 2) | Misses mean (n = 8 or 9) | Δ (makes − misses) |
|---|---:|---:|---:|
| `wrist_to_wrist_distance_m` (1) | 1.389 m | 0.740 m | **+65 cm** (makes more open) |
| `wrist_height_advantage_m` (2) | −1.071 m | −0.179 m | **+89 cm** (makes had defender's hand 89 cm further below shooter) |
| `shooter_fade_mps` (3) | 1.237 m/s | 1.035 m/s | +0.20 m/s |
| `floor_distance_m` (4) | 1.143 m | 1.326 m | −0.18 m (counter-intuitive — see Discussion) |
| **`composite_z` (5)** | **−2.836** | **+0.630** | **−3.47 σ separation** |

**Statistical correlation with make/miss outcome (Spearman ρ).**

| Measurement | n | Spearman ρ vs miss outcome | p | Interpretation |
|---|---:|---:|---:|---|
| `wrist_to_wrist_distance_m` | 10 | **−0.696** | **0.025** | **Significant** at α = 0.05; smaller w-w → more misses |
| `composite_z` | 11 | +0.522 | 0.100 | Directional; higher contest → more misses; marginal at n = 11 |

Computed via `scipy.stats.spearmanr` on the per-shot pairs. With n = 2 makes / 8–9 misses, the absolute statistical power is limited; the wrist-to-wrist result is the most informative because the underlying measurement has the most well-defined make/miss-separating signal.

**Stage 7 accuracy vs hand-clicked GT** (only measurements 1 and 2 have GT — we triangulate the GT wrist clicks; floor distance / fade / composite have no GT to compare against because we didn't hand-click pelvis or velocity).

| Measurement (vs GT) | n | Mean \|err\| | Median \|err\| | Max \|err\| |
|---|---:|---:|---:|---:|
| `wrist_to_wrist_distance_m` | 9 | 25.1 cm | **10.2 cm** | 112.9 cm |
| `wrist_height_advantage_m` | 9 | 31.3 cm | **16.0 cm** | 114.2 cm |

Median errors well below the makes-vs-misses gap. Both max errors come from the 2 Stage-6 outlier shots (release-frame error of 36–44 frames).

### 5.11 Action duration statistics (informed the 64f decision)

Computed over all 4 096 MultiSubjects clips:

| Class | n | Mean (s) | Median (s) | p10–p90 (s) | Std (s) |
|---|---:|---:|---:|---|---:|
| dribble | 1 564 | 3.95 | 3.40 | 1.92–6.44 | 2.17 |
| layup | 512 | 2.93 | 2.52 | 1.44–5.00 | 1.66 |
| shoot | 2 020 | **1.96** | **1.80** | **1.12–2.96** | **0.84** |

The 64f (2.56 s) window covers 142 % of a median shot and 102 % of a median layup.

---

## 6. Figures (file path · one-line caption)

All figures live under `outputs/`. Sizes verified on disk 2026-06-05.

**Training curves**
- `outputs/stage5_diag/learning_curve.png` — *v1 canonical model: train loss + val mean-per-class over 24 epochs, best epoch 10 at mean1 = 0.772.*
- `outputs/stage5_diag/learning_curves_before_after.png` — *v2 split: full fine-tune (best ep 2, val mean1 = 0.769) vs linear probe (best ep 23, val mean1 = 0.721), side-by-side.*
- `outputs/stage5_diag/learning_curves_reg_ablation.png` — *v2 baseline vs stronger reg: train loss + val mean-per-class. Reg lands at higher train-loss floor (0.81 vs 0.70) and pushes best epoch from 2 to 13; val ceiling essentially unchanged.*

**Confusion matrices**
- `outputs/stage5_diag/confusion_matrix.png` — *v1 canonical model, contiguous-window confusion on 120 balanced val clips (dribble 0.80 / layup 0.49 / shoot 0.95).*
- `outputs/stage5_diag/confusion_before_after.png` — *Linear probe (shoot 0.54, mean1 0.62) vs full fine-tune (shoot 0.93, mean1 0.73) on the v2 test holdout, side-by-side.*

**Stage 5 action overlays (TrackID3x3, multi-player)**
- `outputs/stage5_action_overlays/basket_S4T4_post_action.mp4` — *P4 (track 4) shooting on the right side of the court; SHOOT flash + skeleton at the predicted release.* **Canonical "real shot" example.**
- 12 more clips in the same directory (one per TrackID3x3 indoor clip evaluated).

**Stage 5 shot overlays (MultiSubjects, single-player close-up)**
- `outputs/stage5_overlays/442_s_3_shot.mp4` — *MS clip, single shooter, full overlay with per-frame class probability bars + SHOOT flash at peak.*
- `outputs/stage5_overlays/446_p_2_shot.mp4` — *MS layup clip.*
- `outputs/stage5_overlays/964_d_1_shot.mp4` — *MS dribble clip (negative-class example).*

**Stage 6 release overlays (MultiSubjects, 36 clips)**
- `outputs/stage6_release_overlays/755_s_4_release.mp4` — *Clean MS shot, dominant wrist apex (red) clearly late on follow-through, forward marker (orange) at the visible release.* **Canonical clean-release example.**
- `outputs/stage6_release_overlays/{1_s_2, 133_s_1, 143_s_3, …}_release.mp4` — *35 additional MS shot clips with the same overlay format.*

**Stage 6 release overlays (TrackID3x3 detected shots, 9 events)**
- `outputs/stage6_release_overlays/trackid3x3/basket_S4T4_post_t4_f200_release.mp4` — *Realistic gameplay: P4's predicted release frame on a real jump shot.* **Canonical realistic example.**
- 8 more files in `outputs/stage6_release_overlays/trackid3x3/`, one per Stage-5 event at ≥ 0.6 (4 real shots + 5 FPs).

**Stage 2 wrist-tracking overlays (Phase C qualitative acceptance)**
- `outputs/stage2_gate/pose2d_overlay_basket_S4T4_post.mp4` — *RTMPose-l Halpe-26 wrists tracked through a full TrackID3x3 indoor clip; the qualitative-acceptance evidence for F21.*
- `outputs/stage2_gate/stage1_detect_overlay_basket_S4T4_post.mp4` — *Stage 1 YOLO+ByteTrack boxes on the same clip.*

**Pipeline diagram / setup photo / pose-heatmap viz**
- **NOT DONE.** No pipeline architecture diagram, no setup photo, no rendered pose-heatmap visualisation exists in `outputs/` yet. These would need to be made (recommended in `docs/development_log.md` "Report figures" section).

**Stage 4 — calibration + triangulation figures (iPhone session)**
- `data/sessions/2026-06-02_1on1/extrinsics_A.diag.png` — *Cam A calibration frame with clicked landmarks (red) + reprojected world landmarks (green); 9 of 17 landmarks accepted; mean reproj 4.89 px (EPNP).*
- `data/sessions/2026-06-02_1on1/extrinsics_B.diag.png` — *Cam B calibration frame; 12 of 17 landmarks accepted; mean reproj 3.71 px (SQPNP).*
- `data/sessions/2026-06-02_1on1/sync.diag.png` — *Per-clap offset agreement plot; 4 of 4 claps detected on both cameras; max disagreement 64 ms.*
- `outputs/sessions/2026-06-02_1on1/stage4_overlay.mp4` — *Reprojection-overlay video: 3D keypoints projected back to cam A image, showing the triangulated skeleton tracks reproject onto the players cleanly.*

**Stage 7 — contest-metric figures (iPhone session)**
- `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/shot_005.mp4` — *Fadeaway example (shot 5, jump-shot miss). 3.6 s clip with 180° orbiting camera, both skeletons + yellow dashed contest line + on-screen metric panel showing w-w 32 cm / fade 4.37 m/s / composite +4.83 σ. Red panel border = heavily contested.* **Canonical "heavily contested" example.**
- `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/shot_008.mp4` — *Open shot example (shot 8, fadeaway miss). Composite −2.12 σ, defender's hands 83 cm below shooter's release. Green panel border = uncontested.* **Canonical "uncontested" example.**
- `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/shot_002.mp4` — *Open layup MAKE (shot 2). Composite −0.47 σ, w-w 117 cm, height adv −67 cm.* Canonical layup-make example.
- `outputs/sessions/2026-06-02_1on1/3d_reenactments_full_metrics/highlight_reel.mp4` — *60-second concatenated highlight reel of the 5 best-rendered shots with title plates.* Deck-grade supplementary material.
- `outputs/sessions/2026-06-02_1on1/eval_review_full_metrics/shot_NNN.png` (×15) — *Per-shot evaluation review: side-by-side cam A and cam B at the release frame, GT wrist clicks (red circles) + predicted wrists (green circles) + per-pixel error (yellow line), with 6-line caption showing all stage results.* The deck-grade per-shot inspection panels.
- `outputs/sessions/2026-06-02_1on1/shot_release_pngs/shot_NNN_release.png` (×15) — *Single-frame extracts at the predicted release moment from the original 3D reenactments (simpler — only the wrist-to-wrist distance shown, no full metric panel). Useful for compact in-report figures.*

**Stage 7 — to be made (deck plots, not yet in `outputs/`)**
- **Composite z-score histogram** with the 2 makes overlaid as vertical red lines. Visualises the 3.5 σ make/miss separation. **NOT DONE** but trivial; numbers in `contest_metrics.json::per_shot[*].pred.instantaneous.composite_z`.
- **Wrist-to-wrist scatter vs make/miss** with Spearman ρ = −0.70 / p = 0.025 annotated. **NOT DONE** but trivial.
- **Stage 4 wrist 3D error boxplot** (the 13 per-shot errors) with the 2 outliers annotated. **NOT DONE** but trivial; numbers in `evaluation.json::stage4_wrist_3d::per_shot`.

---

## 7. Failure cases

Three concrete examples drawn from the user's manual review of the 9 TrackID3x3 Stage-5 events (§5.8):

1. **S6T4 track 4, frame 88, p_shoot = 0.82 — "shooting-like pose without the ball."**
   The player is in an arms-raised pose that visually resembles a shot but no shot attempt occurs in the clip. The pose CNN is fooled because the input is pose-only — there is no ball cue to disambiguate. Stage 6 fallback catches it (`forward_conf == "low"`, fwd == apex == 65) because the rule cannot find a sustained overhead arm extension. **Class of failure: pose-only ceiling. The fundamental fix is a ball cue, which is gated on iPhone footage + a working ball detector.**

2. **S4T6 track 6, frame 176, p_shoot = 0.65 — "wrong-time extension."**
   Stage 5 fires a borderline-confidence event on a player who never shoots in the clip. Stage 6 *does* find a sustained `wrist-above-shoulder + elbow ≥ 160°` interval — but elsewhere in the search window (centred at frame 135, far from the supposed Stage-5 peak at 176), and tags `forward_conf == "high"`. The F24 cross-method filter therefore misses this FP. **Class of failure: F24 filter blind spot — the rule confirms the geometric signature exists somewhere in the window, but cannot enforce that it occurs near the supposed shot moment.** A tighter spatial / temporal gate (e.g., require the extension to land within ±10 frames of the Stage-5 peak) would catch this.

3. **S6T3 — occluded layup at the very end of the clip.**
   A layup attempt is visible in the clip but no Stage 1 ByteTrack track exists at the shot frame (the shooter is occluded behind a defender). With no pose track, neither Stage 5 nor Stage 6 can fire. **Class of failure: upstream Stage 1 / Stage 2 detection gap, not a classifier failure.** Mitigations are detector-side (Decision 1 fallbacks: BoT-SORT, OC-SORT, larger YOLOv8 variant).

Additional related observation (S6T5): a reverse layup was *partially* picked up by the classifier at p_shoot ≈ 0.5, below the 0.6 detector threshold. Unusual motion + non-standard release. Not counted in the 9 Stage-5 events but worth noting as a class of motion the model under-represents.

**Session 2026-06-02_1on1 failure cases (Stage 4 / 5 / 6 / 7 specific):**

4. **Defender contest poses misclassified as shots (Stage 5).** 18 of 30 fused Stage-5 events on this session are defender contest-poses on real shots — the defender's contesting arm extension looks shot-like to a classifier that was never trained on a defender-pose class. The 0.40 precision number is dominated by this. **Class of failure: training-data class design.** Fix is to add a defender-contest class to MultiSubjects pretraining, or to apply F24's cross-method gate more aggressively (the present session eval did not apply it).

5. **Stage 6 release-frame outliers on shots 13 + 14.** Both have Δframes = −36 to −44, i.e., Stage 6 picked a release frame ~0.6 s before the actual release. Root cause: at the boundary of the 180-second clip, the ±32-frame search window catches the defender's contest extension instead of the shooter's. Stage 6's `identify_shooter` picks the player with the highest peak wrist-Z in the window — which can be the defender if their contest arm extends higher than the shooter's. **Class of failure: shooter-identification heuristic.** Fix is to either (a) require shot-track continuity with Stage 5's track ID, or (b) manually override shooter assignment per shot. Neither is currently shipped.

6. **Stage 4 wrist 3D errors inflated by Stage 6 outliers.** Stage 4 wrist median error vs GT = 3.94 cm; mean = 20.8 cm; P90 = 42.7 cm. The mean is pulled up entirely by shots 13 and 14 (#5 above). Stage 4 itself is healthy at every shot's *actual* release frame; the inflation is downstream from a Stage 6 mistake. **Class of failure: dependent-stage error compounding.** Lesson for report: when reporting cascaded-pipeline accuracy, separate per-stage error from the compounded end-to-end error, and identify which stage's mistake explains which outlier.

7. **No third iPhone ⇒ no triangulation hold-out.** Decision 4's "≤ 5 px reprojection on held-out 3rd camera" test was the canonical accuracy gate. We have no held-out camera, so the alternative — re-triangulating known landmarks visible to both cameras (3.9 cm sanity) — is what we report. **Class of failure: scope reduction.** The 3.9 cm number is honest but cannot detect a systematic bias that would only show on a 3rd camera; for that we'd need either the 3rd camera or per-frame visibility cross-correlation against an external reference.

---

## 8. References

Repos used directly:
- **ultralytics** (YOLOv8 + ByteTrack) — https://github.com/ultralytics/ultralytics
- **rtmlib** — https://github.com/Tau-J/rtmlib
- **MMPose** (RTMPose-l Halpe-26 model + weights only) — https://github.com/open-mmlab/mmpose
- **MMAction2 v1.2.0** (PoseConv3D code, K400 checkpoint) — https://github.com/open-mmlab/mmaction2
- **MMEngine / MMCV** — https://github.com/open-mmlab/mmengine and https://github.com/open-mmlab/mmcv
- **WASB-SBDT** (downloaded, ran, ultimately dropped) — https://github.com/nttcom/WASB-SBDT
- **Pose2Sim** (declared dep; Stage 4 build is in the other terminal) — https://github.com/perfanalytics/pose2sim

Papers / datasets:
```bibtex
@inproceedings{duan2022posec3d,
  title={Revisiting Skeleton-based Action Recognition},
  author={Duan, Haodong and Zhao, Yue and Chen, Kai and Lin, Dahua and Dai, Bo},
  booktitle={CVPR}, year={2022}
}
@inproceedings{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and others},
  booktitle={ECCV}, year={2022}
}
@article{jiang2023rtmpose,
  title={RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  author={Jiang, Tao and Lu, Peng and Zhang, Li and others},
  journal={arXiv preprint arXiv:2303.07399}, year={2023}
}
@misc{jocher2023yolov8,
  title={YOLOv8 by Ultralytics},
  author={Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year={2023}, howpublished={\url{https://github.com/ultralytics/ultralytics}}
}
@inproceedings{tarashima2023wasb,
  title={Widely Applicable Strong Baseline for Sports Ball Detection and Tracking},
  author={Tarashima, Shuhei and Haq, Mohammad and Wang, Yushan and Tagawa, Norio},
  booktitle={BMVC}, year={2023}
}
@inproceedings{xu2024finesports,
  title={FineSports: A Multi-person Hierarchical Sports Video Dataset for Fine-grained Action Understanding},
  author={Xu, Jinglin and Sun, Sibo and Zhang, Guohao and others},
  booktitle={CVPR}, year={2024}
}
@article{multisubjects2024,
  title={A multi-subject video dataset for single-person basketball action recognition},
  journal={CVIU}, year={2024}
}
@inproceedings{kay2017kinetics,
  title={The Kinetics Human Action Video Dataset},
  author={Kay, Will and Carreira, João and Simonyan, Karen and others},
  booktitle={arXiv:1705.06950}, year={2017}
}
@inproceedings{kumar2022lpft,
  title={Fine-tuning can distort pretrained features and underperform out-of-distribution},
  author={Kumar, Ananya and Raghunathan, Aditi and Jones, Robbie and Ma, Tengyu and Liang, Percy},
  booktitle={ICLR}, year={2022}
}
@misc{pagnon2022pose2sim,
  title={Pose2Sim: An open-source Python package for multiview markerless kinematics},
  author={Pagnon, David and Domalain, Mathieu and Reveret, Lionel},
  year={2022}, howpublished={\url{https://github.com/perfanalytics/pose2sim}}
}
@article{dalygrafstein2020,
  title={A Bayesian hierarchical model for shot success in the NBA},
  author={Daly-Grafstein, Daniel and Bornn, Luke},
  year={2020}
}
@article{mikic2024,
  title={Biomechanical effects of defensive pressure on the basketball jump shot},
  author={Mikic, M. and others}, year={2024}
}
@inproceedings{huang2019tracknet,
  title={TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications},
  author={Huang, Yu-Chuan and others}, booktitle={DSAA}, year={2019}
}
@misc{trackid3x3,
  title={TrackID3x3 dataset},
  howpublished={\url{https://github.com/open-starlab/TrackID3x3}}
}
@misc{deepsportradar2022,
  title={DeepSportradar Basketball Instants},
  howpublished={ACM MMSports 2022 Kaggle competition},
  year={2022}
}
@book{hartley2003mvg,
  title={Multiple View Geometry in Computer Vision},
  author={Hartley, Richard and Zisserman, Andrew},
  year={2003}, publisher={Cambridge University Press}
}
@article{zhang2000calibration,
  title={A flexible new technique for camera calibration},
  author={Zhang, Zhengyou},
  journal={IEEE TPAMI}, volume={22}, number={11}, pages={1330--1334}, year={2000}
}
@article{tsai1987camera,
  title={A versatile camera calibration technique for high-accuracy 3D machine vision metrology using off-the-shelf TV cameras and lenses},
  author={Tsai, Roger Y.},
  journal={IEEE Journal of Robotics and Automation}, volume={3}, number={4}, pages={323--344}, year={1987}
}
@article{lepetit2009epnp,
  title={EPnP: An Accurate O(n) Solution to the PnP Problem},
  author={Lepetit, Vincent and Moreno-Noguer, Francesc and Fua, Pascal},
  journal={IJCV}, volume={81}, number={2}, pages={155--166}, year={2009}
}
@inproceedings{terzakis2020sqpnp,
  title={A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem},
  author={Terzakis, George and Lourakis, Manolis},
  booktitle={ECCV}, year={2020}
}
@article{kuhn1955hungarian,
  title={The Hungarian method for the assignment problem},
  author={Kuhn, Harold W.},
  journal={Naval Research Logistics Quarterly}, volume={2}, number={1-2}, pages={83--97}, year={1955}
}
@misc{opencv,
  title={Open Source Computer Vision Library},
  howpublished={\url{https://opencv.org}},
  note={Used for cv2.calibrateCamera, cv2.solvePnP, cv2.findChessboardCorners, cv2.undistortPoints, cv2.triangulatePoints}
}
```

---

## 9. What's NOT in this document (explicit list)

Updated 2026-06-05 after the Stage 4 / 7 + end-to-end deployment work landed. Items still genuinely NOT DONE:

**Genuinely NOT DONE (would require additional work):**

- **Stage 1 mAP / IDF1 on a curated test set — NOT DONE.** Acceptance was qualitative only (Decision 1); no benchmark number reported.
- **Stage 3 ball recall on a curated test set — NOT DONE.** Dropped (F19) before a recall number was measured.
- **Stage 4 calibration-drift auto-detector — NOT DONE.** Decision 10's "2× baseline reprojection-error" gate not implemented (no drift incidents observed; ~30 lines if needed).
- **Stage 5 FineSports intermediate transfer (PRD Path C step 3) — NOT DONE.** FineSports pkl labels never delivered.
- **Stage 5 ball-trajectory MLP (two-stream head, Decision 5 row 4) — NOT DONE.** Ball drop (F19) eliminated the input.
- **Stage 5 defender-contest action class — NOT DONE.** The bottleneck on session precision (most FPs are defender contest-poses).
- **Stage 6 release-frame eval on MultiSubjects / TrackID3x3 vs GT — NOT DONE** (no per-frame release labels exist in those datasets). Δframes-vs-GT now done on the iPhone session (§5.7).
- **iPhone footage LP-FT on own data — NOT DONE.** Session 2026-06-02_1on1 used as evaluation only, no training.
- **Multi-window TTA at inference — NOT DONE.**
- **Rater study for Stage 7 composite weights — NOT DONE** (★ stretch, moved to future work per Decision 9).
- **Mikic-2024 quartile effect reproduction for Stage 7 — NOT DONE.** n = 11 evaluable shots per quartile is too thin for the test to be meaningful; defer until ≥ 30 shots.
- **3rd iPhone for triangulation hold-out — NOT DONE.** Scope reduced from 3 to 2 cameras.
- **Refactor `scripts/compute_contest_session.py` / `scripts/run_session_pipeline.py` into `src/courtvision/stage{4,7}*/` packages — NOT DONE.** Deferred; see "Deployment / refactor backlog" in `development_log.md` §13.9.
- **Pipeline architecture diagram, court setup photo, rendered pose-heatmap visualisation — NOT DONE.** Trivial to draw; the 3D reenactment frames serve as a partial substitute for the third.
- **Composite z-score histogram + wrist-to-wrist scatter plots — NOT DONE.** Trivial matplotlib once the numbers are pulled from `contest_metrics.json`.

**Previously NOT DONE — now done as of 2026-06-05:**

- ~~Stage 4 reprojection error — NOT DONE in this terminal~~ → done; intrinsics 2.22 / 1.09 px, extrinsics 4.89 / 3.71 px, triangulation sanity 3.9 cm mean (§5.6).
- ~~Stage 4 visibility metric — NOT DONE~~ → implicit via `Pose3D.keypoints[i].n_cameras` ∈ {0, 2}.
- ~~Stage 7 contest metric implementation — NOT DONE in this terminal~~ → done; 5-measurement implementation in `scripts/compute_contest_session.py` (§3.4, §5.10).
- ~~Stage 7 distribution sanity check — NOT DONE~~ → done; all 11 evaluable shots produce non-degenerate values, no triangulation-collapse modes (§5.10).
- ~~Make / miss correlation (Stage 7 Decision 9 validation (b)) — NOT DONE~~ → done; Spearman ρ = −0.70, p = 0.025, n = 10 (§5.10).
- ~~iPhone footage collected — NOT DONE~~ → done; session 2026-06-02_1on1, 15 hand-labelled shots (§2.6).
