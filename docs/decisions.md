# CourtVision — Technical Approach Working Doc

## Milestone 2 (due May 22) — requirements

Each decision below should be framed to support the milestone deliverable:

- **Method overview** — what we're doing, with diagrams
- **Design intuition** — *why* this choice, not just what
- **Baselines** — what we're comparing against, and why we expect to beat / match them
- **Experimental setup** — how the dataset is used (train/val/test splits, pretraining stages, held-out subsets)
- **Evaluation** — concrete metrics + thresholds per component
- **Implementation plan / progress** — current state and what's next

For every decision, capture: (a) the choice, (b) the intuition, (c) what we're comparing against, (d) how we'll know it worked.

---

## Locked decisions

1. **Player detection + tracking — YOLOv8m + ByteTrack, person class only.**
   - Rationale: off-the-shelf, no baseline comparison needed (we're consuming output, not contributing here). Detecting ≤4 people on a fixed half-court is well within COCO-trained capability; ByteTrack handles brief contest occlusions via low-confidence association.
   - RTMDet-m considered for MMPose ecosystem consistency but rejected — small integration win only matters if we do the ★ RTMPose fine-tune; YOLOv8m is more popular, better documented.
   - Fallback if poor results: try BoT-SORT (camera motion compensation), OC-SORT, or larger YOLOv8l/x. Treat as future work, not core scope.

2. **2D pose estimation — RTMPose-l, Halpe-26 keypoints, top-down on YOLOv8 player crops, off-the-shelf.**
   - Rationale: Halpe-26 gives wrist precision the contest metric needs, maps cleanly to Pose2Sim/OpenSim for ★ IK, and is the format Pose2Sim natively supports. Top-down works because we have ≤4 people. Speed irrelevant (offline processing), so RTMPose-l over -m/-s for accuracy.
   - **Scope update (per Decision 6):** Halpe-26 stays the locked output format for RTMPose-l and is used by the **geometric pipeline** — triangulation (Decision 4), release-frame rule (Decision 8), contest metric (Decision 9), ★ OpenSim IK. The **shot classifier (Decisions 5, 6)** consumes a COCO-17 subset of these same Halpe-26 keypoints — no second pose-extraction pass, just a subset selection at classifier input construction. Reason: PoseConv3D's Kinetics-400 pretrained checkpoints use COCO-17, and COCO-17 is the natural common subset between Halpe-26 (ours) and BlazePose-33 (MultiSubjects pretraining format).
   - Phase 1 = off-the-shelf, no fine-tune.
   - **★ Phase 2 fallback (if wrist accuracy is poor):** fine-tune on TrackID3x3 Indoor (2,601 pose-labeled frames at 720p, Sony camcorder) + ~200 self-labeled frames of our own footage. Use partial supervision: backprop loss only on the 10 TrackID3x3 keypoints for those frames, leave other Halpe-26 outputs to pretraining. Fine-tune head + last RTMPose stage at lr=1e-5 with early stopping.
   - **Phase 2 concerns to flag:**
     - TrackID3x3 only labels 10 keypoints (not 26) — need partial-supervision setup
     - Exact 10-keypoint schema not yet verified — load annotation files and confirm wrists are included before committing
     - Domain shift: their Indoor footage is different gym, 720p Sony camcorder, full-court fixed camera; ours is 1080p iPhone half-court multi-view
     - 2,601 frames is small — fine-tune last layers only, not full retrain
   - Fallback if even Phase 2 underperforms: try HRNet-W48 or ViTPose-L off-the-shelf as alternative.
   - **Phase C engine swap (2026-05-28): inference via `rtmlib` (ONNX), not MMPose — PRD §5.2 deviation, model unchanged.** We keep the locked *model* (RTMPose-l, Halpe-26, `body7-halpe26 256x192`, the same weights — exact ONNX export of the `2abb7558` checkpoint) but run it through **rtmlib** (`onnxruntime`) instead of the MMPose/`mmcv` PyTorch stack.
     - **Why (researched, not guessed):** the Modal image runs bleeding-edge **torch 2.12 + CUDA 13.0**, and the OpenMMLab stack is effectively unmaintained against it — mmcv's last release (2.2.0, Apr 2024) ships prebuilt wheels only up to torch ~2.1, and there is an *open, unanswered* mmcv bug (#3310) that `mmcv._ext` fails to compile on torch 2.9 + CUDA 13. So mmpose could not import/run, and the long error chain (numpy-2 ABI → xtcocotools → mmdet caps → mmcv compile) were all symptoms of one root cause: too-new an environment for an unmaintained toolchain. Even forcing the imports through would likely leave `mmcv._ext` (CUDA ops) broken at runtime. Full postmortem in `flags.md` F20.
     - **Why rtmlib is safe:** it depends only on `numpy`/`opencv`/`onnxruntime` — *no torch, mmcv, mmpose, or mmdet* — so it is fully decoupled from the torch/CUDA/numpy version web. Same model + weights → numerically equivalent keypoints. Its low-level `RTMPose(onnx_model, model_input_size, backend, device)(image, bboxes)` takes our Stage 1 bboxes directly (top-down). It also runs CPU/MPS, so Stage 2 can run locally, and the Modal image builds in ~1 min. Bonus: Pose2Sim (our Stage 4 tool) already uses rtmlib.
     - **Scope:** swap affects Stage 2 inference only. The `Pose2DSequence` schema, wrist-PCK eval, COCO-17 subsetting, and overlay are unchanged. **Phase E (PoseConv3D / mmaction2) still needs the OpenMMLab stack** and will face the same torch wall — to be isolated then in a dedicated pinned-torch image (the pattern Phase D used for WASB's `wasb_image`), not the shared image.
     - **Fallback within this deviation:** if RTMPose-l ONNX wrist accuracy is poor, rtmlib also hosts -m/-s/-x halpe26; or trigger the existing Phase 2 ★ fine-tune (which would then need an ONNX re-export).
   - **Phase C result + gate disposition (2026-05-28): CLOSED via documented acceptable deviation.** Evaluated all **13 TrackID3x3 Indoor pose-GT clips** (the full in-domain set; Outdoor=1 clip + Drone=~500 aerial segments excluded as off-domain / unverified schema).
     - **Wrist PCK@0.05 = 0.655** (mean 0.648, std 0.043 across the 13 clips) — **below the 0.85 gate**. But the per-landmark diagnostic shows this is *not* a pose/eval failure:

       | landmark | PCK | | landmark | PCK |
       |---|---|---|---|---|
       | shoulder | 0.941 | | elbow | 0.727 |
       | ankle | 0.845 | | head | 0.675 |
       | hip | 0.802 | | **wrist (gate)** | **0.655** |

       all-keypoint PCK = 0.781.
     - **Interpretation:** (1) shoulders at **0.94** prove the eval pipeline (player matching, normalization, coordinates) is correct — a broken eval can't score any joint that high; (2) the numbers are a textbook **proximal→distal falloff** (shoulder 0.94 → elbow 0.73 → wrist 0.66) — wrists are the hardest joint, not uniquely broken; (3) the GT is **model-derived, not hand-truth (F18)**, so PCK measures *agreement with another estimator* at a strict ~12 px tolerance on small 3v3 players — which understates true accuracy. The 0.85 bar is therefore inappropriately strict for this distal joint against this GT.
     - **`head=0.675` caveat:** likely a mapping offset — GT keypoint 0 ("head") was compared against Halpe **nose**; if their label is top-of-head, that's a systematic shift, not a localization miss. Doesn't affect the wrist gate.
     - **Qualitative (the decisive check, PRD §9.2):** user reviewed 13 wrist-overlay videos (`outputs/stage2_gate/`); **wrist location looks accurate, especially on shot-release frames — the frames the Stage 7 contest metric depends on.** This is the acceptance signal the PRD allows in lieu of the quantitative bar.
     - **Disposition:** Phase C **accepted off-the-shelf** (no ★ fine-tune this sprint). The Phase 2 ★ RTMPose fine-tune remains the documented escalation if wrist accuracy proves insufficient downstream on own footage (Stage 7 contest metric).

3. **Ball detector — WASB (BMVC 2023, `nttcom/WASB-SBDT`), off-the-shelf, no additional tracking wrapper.**
   - Rationale: WASB is purpose-built for sports ball detection across multiple sports including basketball. Heatmap-based (HRNet backbone) handles motion blur natively; this is what TrackNet/WASB exist for. Built-in temporal consistency in inference replaces any need for ByteTrack+Kalman wrapper. Code, pretrained weights, and basketball dataset all public. Peer-reviewed.
   - Initially planned YOLOv8 (ayushpai 3K-image checkpoint) + ByteTrack + Kalman wrapper; switched to WASB after research surfaced it as the SOTA SBDT baseline.
   - **Reclassification:** Phase 1 = off-the-shelf, NOT a model we train. Only the shot classifier (Decision 5) is currently in the "models we train" category. Ball detector only moves into "trained" if Phase 2 fires (★ stretch).
   - **Baseline comparison for milestone (still meaningful even though both are off-the-shelf):**
     - Baseline: YOLOv8 (ayushpai) — generic detector, simpler
     - Our choice: WASB — sports-specialized, heatmap-based, temporally aware
     - Compare on identical held-out test data
   - **Eval plan:**
     - **mAP@0.5 / mAP@0.5-0.95** on held-out slice of U Arizona Roboflow basketball dataset (~9.8K images) → literature-comparable number
     - *Post-script (2026-05-26):* "U Arizona Roboflow basketball" dataset could not be verified at any public URL. Substituted with **DeepSportradar Basketball Instants** (Kaggle; ACM MMSports 2022; primary, headline mAP) + **Roboflow `ownprojects/basketball-w2xcw`** (~9.7K images; secondary). See `flags.md` F12 and PRD §8.3 / §9.3 for current state. Original wording preserved here for Milestone-2 record-keeping.
     - **Qualitative check** on ~5-10 clips each from TrackID3x3 and MultiSubjects → catches catastrophic domain-shift failures
     - **Release-window recall** on WASB's basketball subset (NBA broadcast videos with per-frame ball annotations): hand-pick ~30 obvious shot release moments (~10 min), compute detection rate in ±15-frame windows using their existing ball labels. No labeling on our own footage for now; defer to final report if needed.
   - **Phase 1 → Phase 2 (★) gate:** release-window recall ≥ 90% on WASB external eval. If gate fails, trigger the 5-day Phase 2 custom-training plan from scope doc §5.2. (Note: WASB basketball is NBA broadcast, harder ball pixel size than our amateur half-court — passing this gate likely means our footage will be at least as good, but not guaranteed.)
   - Fallback if WASB underperforms or has integration issues: TrackNetV5, YOLOv8 (ayushpai), or Phase 2 custom training.

4. **Triangulation — Pose2Sim DLT for player pose, separate `cv2.triangulatePoints` for ball, chessboard calibration.**
   - **Player pose:** Pose2Sim DLT triangulation across 3 calibrated cameras with default config — Butterworth 4th-order low-pass filtering (6 Hz cutoff) for smoothing.
   - **Ball:** Separate `cv2.triangulatePoints` (or equivalent direct DLT) on WASB detections, with per-camera RANSAC across the 3 cameras. Rationale: clean separation; ball physics (ballistic when airborne) differs from body kinematics; isolates failure modes between pose and ball.
   - **Calibration:** chessboard, per scope doc default. AVCameraCalibrationData deferred to ★ stretch.
   - **Per-frame visibility tracking:** log how many cameras contributed to each keypoint per frame (3 / 2 / 0). Enables auditing of "strong" vs. "weak" 3D points downstream.
   - **2-camera fallback:** when occluded in 1 of 3 cameras, document and continue with degraded accuracy. When occluded in 2 of 3, keypoint is missing for that frame.
   - **Reprojection error — two things, both tracked:**
     - **Threshold (input):** Pose2Sim default ~30 px for RANSAC view rejection; `likelihood_threshold` ~0.3 for RTMPose confidence cutoff before triangulation. Defaults for v1; tune empirically based on visibility logs.
     - **Target (output):** mean Euclidean pixel error < 5 px when projecting triangulated 3D pose back into a held-out 3rd camera and comparing to that camera's 2D detections. Zheng 2025 hits ~2 px with 8 cameras; 5 px is appropriate for our 3-camera setup. Report aggregated AND on release-window subset.
   - **Tuning path:** if visibility logs show frequent 2-camera fallback while target metric is acceptable → loosen threshold; if target metric is poor → check calibration first, not thresholds; if target metric is well below 5 px and 2-camera fallback is rare → consider tightening for cleaner 3D.
   - No baseline comparison (off-the-shelf, geometric); evaluation is self-supervised via held-out camera reprojection.

5. **Shot classifier architecture — PoseConv3D backbone + auxiliary ball-trajectory MLP at head (two-stream), 4-stage transfer pipeline with 4-row ablation + external benchmark comparisons.**
   - **Architecture:** Two-stream design. **Pose stream:** PoseConv3D (Duan 2022) — renders keypoint sequences as 3D heatmap volumes, applies 3D-CNN. SlowOnly-R50 backbone. **Ball-trajectory stream:** small MLP (2-3 layers) consuming per-window 3D ball-trajectory features. Two streams concatenated before the final 4-class classification head.
   - **Why two-stream (literature-backed):** the 2023 badminton shot-detection paper (TrackNet + YOLOv7) directly tests this. Ball-trajectory alone: 72.3% F1. Player-action + ball-trajectory combined: 90.5% F1 — **18-point F1 gain from the combination**. Ball-trajectory has high recall (sees obvious things) but low precision (false positives on non-shot ball motion); player pose adds precision. The two signals are complementary in a way that's hard to argue against for a sport with a ball-following-arc structure like basketball.
   - **Pose-stream choice (Family B) rationale:** chosen over Family A (pixel-based) because — data realism (continuous video + hand-labeled release frames is small training data for pixel fine-tuning), pipeline alignment (we already produce poses), interpretability, fast iteration, domain robustness.
   - **Why ball features at the HEAD, not as a PoseConv3D input channel:** earlier iterations considered ball-as-heatmap-channel (treats ball as an 18th input channel). Rejected because: (a) requires modifying PoseConv3D's first conv from 17 → 18 channels, breaking direct compatibility with pretrained Kinetics-400 weights; (b) ball isn't a body keypoint, doesn't belong in a pose-rendering channel; (c) head-level concatenation is the standard two-stream pattern (literature-backed) and lets pretrained pose backbone load cleanly. Ball features enter at the head as a parallel modality.
   - **Window ballistics sanity check:** typical jump shot — release at ~2.5m height, peak at ~4m, peak reached ~0.55s post-release. In a 32-frame window @ 60fps centered on release (frames T-16 to T+16), ball goes from ~2.4m → 3.6m (~8ft to ~12ft). Even without reaching peak in-window, the **ascent rate** is highly discriminative vs. passes (chest-pass height, no clean ascent) and rebounds (bouncing ball lacks clean upward trajectory). Layup edge case noted: layups peak around 10-11ft so a hard "above 12ft" threshold would miss them — using continuous trajectory features rather than hard threshold avoids this.
   - **Task framing:** event detection on continuous video, NOT clip classification. Inference = sliding window (32 frames @ 60fps, stride 4) → P(shoot) + P(layup) per window per player → NMS/peak finding → detected shot events with approximate center timestamps. Downstream release-frame rule (Decision 8) pinpoints exact release frame inside each detected window. Input format detailed in Decision 6; transfer-learning recipe detailed in Decision 7.
   - **Inference-time ball-ascent filter (Option B from research discussion):** before running the full classifier, gate on a simple ball-trajectory check — only consider windows where the 3D ball reaches above ~3m (~10ft, rim height) within ±20 frames of window center. Drastically reduces false-positive opportunities at inference; doesn't require retraining. Free win on top of the two-stream classifier. Threshold tunable; conservative enough to catch layups.
   - **Earlier muddle corrected:** initially proposed rule-based shot detector as a "baseline" — this was a category error. The rule lives downstream as the release-frame extractor, NOT as a competing classifier. Dropped from baseline list.
   - **Pipeline path:**
     - **Path C (primary):** 4-stage transfer — Kinetics → MultiSubjects → FineSports → ours
     - **Path A (fallback if FineSports access fails):** 3-stage — Kinetics → MultiSubjects → ours
   - **Internal ablation (Path C — 4 rows, isolates contribution of each transfer stage AND the ball-trajectory stream):**
     - **Row 1: Stage 1+2 only, pose-only** — Kinetics + MultiSubjects pretraining; checkpoint here
     - **Row 2: Stage 1+2+3, pose-only** — adds FineSports temporal-localization pretraining; checkpoint here
     - **Row 3: Stage 1+2+3+4, pose-only** — adds LP-FT on our continuous video; pose-only final model
     - **Row 4: Stage 1+2+3+4 + ball-trajectory MLP** — adds the two-stream ball head; full model
     - Rows 1-3 quantify per-stage transfer contribution; Row 3 vs. Row 4 quantifies the ball-trajectory stream contribution (badminton paper predicts ~17pp F1 gain)
     - Each row evaluated on the same test sets
   - **Internal ablation (Path A fallback — 3 rows):**
     - **Row 1: Stage 1+2 only, pose-only**
     - **Row 2: Stage 1+2+3, pose-only** (LP-FT on our data)
     - **Row 3: Stage 1+2+3 + ball-trajectory MLP** — full model
   - **External benchmark comparisons** (apply our method to their data, compare to their published numbers — shows pipeline isn't just iPhone-footage-tuned):
     - **MultiSubjects (CVIU 2024):** top-1 accuracy on their 3-class test split, vs. their published numbers — evaluated using the post-Stage-2 checkpoint (pose-only)
     - **FineSports (CVPR 2024):** mAP at tIoU={0.3, 0.5, 0.7} for temporal localization on NBA footage, vs. their published numbers — evaluated using the post-Stage-3 checkpoint (pose-only, Path C only)
     - **Free-Throw Hit-Probability 2024:** F1 = 0.98 reference, with explicit caveat that their setting is much simpler (free throws, no defender, single camera)
   - **Test sets:**
     - MultiSubjects held-out clips — clip-level recognition on pretraining distribution
     - FineSports held-out videos — temporal localization on out-of-distribution continuous NBA video (Path C only)
     - Our held-out sessions — event-level precision/recall/F1 with ±0.3s tolerance on in-distribution iPhone half-court footage
   - **No results required for Milestone 2** — present experimental design only.
   - Fallback if pose-stream underperforms: switch to Family A pixel-based (SlowFast or VideoMAE with same transfer chain), keeping ball-trajectory MLP intact.

6. **Shot classifier inputs — 2D per-camera with late fusion, COCO-17 pose subset + separate ball heatmap channel, single-player per-crop input.**

   - **Input modality — 2D per-camera, NOT 3D triangulated.** Literature-backed reversal of earlier intuition (3D-primary was considered). The PoseConv3D paper itself argues 2D pose representations outperform 3D coordinate inputs for action recognition because: (a) 3D triangulation inherits 2D error and adds calibration/reprojection noise on top; (b) 2D heatmaps preserve uncertainty (Gaussian spread) that point estimates lose; (c) PoseConv3D's whole thesis is that stacking 2D heatmaps over time as a 3D volume gets the temporal-3D benefit from 2D inputs directly. From the paper: *"3D skeleton is noisy... 2D skeleton has superior quality"* and *"high-quality 2D pose representations, when processed as 3D heatmap volumes, lead to better recognition performance than traditional 3D reconstruction methods or coordinate-based input formats."* 3D pose still produced and used downstream (contest metric, release rule, ★ biomech) — just not at the classifier. No 3D classifier path / fallback needed.

   - **Keypoint format — COCO-17 subset, classifier ONLY.** Geometric pipeline (triangulation, contest, release rule) continues using full Halpe-26 from RTMPose-l. Classifier consumes a 17-keypoint subset extracted from the same RTMPose output — no second pose extraction, just subset selection at input construction. Reasons:
     - PoseConv3D's pretrained Kinetics-400 checkpoints use COCO-17 → compatible out of the box
     - MultiSubjects uses BlazePose-33 (MediaPipe format). COCO-17 is the natural common subset between Halpe-26 (us) and BlazePose-33 (MultiSubjects) — both formats extend COCO-17 with different extras (Halpe adds feet, BlazePose adds face/hands)
     - Keypoints kept (shoulders, elbows, wrists, hips, knees, ankles) are exactly what matters for shot mechanics — upper-body extension and arm dynamics
     - Decouples classifier from rest of pipeline: triangulation/contest stay on full Halpe-26 for wrist+foot detail; only classifier subsets to COCO-17
   - **Cross-reference note (already added to Decision 2):** Halpe-26 stays locked as RTMPose-l's output; classifier subsetting is downstream.

   - **Ball input — auxiliary ball-trajectory MLP at the head (two-stream), NOT as PoseConv3D input channel.** Updated from earlier iterations:
     - **v1 design (now superseded):** ball as separate heatmap channel parallel to pose, in same per-player crop. Rejected because: required modifying PoseConv3D's first conv from 17→18 channels, breaking direct compatibility with pretrained Kinetics-400 weights; ball isn't a body keypoint and didn't belong in a pose-rendering channel.
     - **v1.5 design (briefly considered):** drop ball from classifier entirely; defer to release rule and contest metric downstream. Rejected because: ball-trajectory information is highly discriminative for shot detection (the badminton paper showed 18pp F1 gain from adding ball signal to player-action classifier), and "above-rim ball height" is exactly the kind of physical signal a sliding-window classifier can pick up.
     - **Locked design (current):** **two-stream architecture** — PoseConv3D processes 17-channel pose input normally; ball trajectory features (3D ball z-trajectory over window, plus summary statistics like max_z, time-to-peak, peak_v_z, ascent rate) feed a parallel small MLP (2-3 layers); both feature vectors concatenated before the final 4-class linear head. Standard two-stream pattern, literature-backed (PoseConv3D paper's multi-modal fusion claim; badminton TrackNet+YOLOv7 paper).
     - **Why head-level fusion over input-level fusion:** pretrained pose backbone loads cleanly; ball features can be injected at later transfer stages without disturbing earlier pretraining; cleaner architectural separation.
     - 3D ball positions (from Decision 4 via `cv2.triangulatePoints` on WASB outputs) are the natural input for the ball-trajectory MLP — uses 3D not 2D because trajectory features (height above court, ascent rate) are physically meaningful in 3D coordinates, not camera-frame coordinates.
   - **Inference-time ball-ascent filter** (Decision 5 cross-reference): before running the full classifier, gate on a simple ball-trajectory check — only consider windows where the 3D ball reaches above ~3m (~10ft, rim height) within ±20 frames of window center. Drastically reduces false-positive opportunities; doesn't require retraining; free win on top of the two-stream classifier. Note layup edge case: layups peak around 10-11ft so the 3m threshold catches them; we don't use the higher 12ft threshold that would miss layups.

   - **Player scope — single-player per crop.** Cleanest alignment with MultiSubjects pretraining (one-player-per-clip). Per-player crop renders ONLY the target player's pose as heatmap channels — other players' poses are NOT rendered into the target player's input even if their bodies geometrically overlap the crop region. Per-player identity comes from YOLO+ByteTrack (Decision 1). Inter-player context (defender pressure) is the geometric contest metric's job (Decision 9), not the classifier's. Clean separation of concerns.

   - **Possession — implicit, not modeled.** With pose-only inputs, the classifier learns "shot motion" from upper-body kinematics directly (arm extension, wrist flick, jump). The model produces P(shoot) + P(layup) per player; the player whose probability peaks during a shot event IS the shooter for downstream stages. No possession-detection model needed.

   - **Coordinate frame — image coords within person crop** (PoseConv3D convention). Earlier "pelvis-centered, torso-scaled" framing was wrong — that's the ST-GCN / coordinate-input convention. PoseConv3D works in image coordinates within the cropped frame; normalization comes from the cropping itself, not from re-centering coordinates. Corrected.

   - **Crop padding — tight person bbox (~1.0× to 1.2×), tunable hyperparameter.** Updated from earlier 1.5× recommendation, which was motivated by keeping the ball visible inside the crop. With ball channel dropped (above), crop padding only needs to comfortably contain the player through their full motion (jump apex, full arm extension). Tight-to-mild padding (~1.0–1.2×) suffices. Does NOT cause pose contamination (only target player's pose is rendered into their input).

   - **Input tensors:**
     - **Pose stream (per player, per camera, per window):** [T=32 frames @ 60fps ≈ 0.53s, K=17 channels (pose only, COCO-17), H, W] as Gaussian heatmaps stacked into a 3D volume.
     - **Ball-trajectory stream (per window, single 3D ball trajectory shared across players):** ball z-trajectory over 32 frames (32-length vector) + scalar summary features (max_z, time-to-peak if reached, peak_v_z, z at window start/end, ascent rate). Fed to small MLP. NOTE: ball stream is per-window not per-player — same ball trajectory feeds all players' classifications in that window.

   - **Inference aggregation — late fusion across cameras at probability level.** 4 players × 3 cameras = 12 PoseConv3D inferences per sliding window. Per-camera probability outputs averaged per player. Per-player aggregated probability stream feeds NMS/peak detection for event timestamps. Graceful degradation: if a player is occluded in 1 camera, the other 2 still produce probability estimates. Late fusion at probability level (vs. early feature fusion or max-pooling) chosen for simplicity and per-camera independence.

   - **Class output — 4-way (dribble / layup / shoot / background)** per Decision 7. Shot detection = peak in P(shoot) + P(layup) above threshold. Multi-class kept consistent across all transfer stages (required for LP-FT to function).

7. **Shot classifier transfer-learning recipe — 4-stage pipeline (Path C primary) with LP-FT in the final stage.**

   - **Halpe-26 stays as RTMPose-l output across the whole pipeline.** Classifier subsets to COCO-17 (Decision 6); geometric stages use full Halpe-26; ★ OpenSim biomech stretch goal preserved (needs foot keypoints for ankle/knee IK and push-off mechanics — Mikic 2024 reproduces joint-angle effects under defense). Free option value; same RTMPose call produces all keypoints.

   - **4-class head structure throughout** (dribble / layup / shoot / background). Required for LP-FT architectural consistency — the linear-probe step needs a meaningful pretrained head to probe. Switching to binary at the final stage would discard that benefit. Also avoids class-imbalance issues a binary shot/not-shot would face. Inference for our task: shot detected = argmax in {shoot, layup} above threshold.

   - **Pipeline path:**
     - **Path C primary:** Kinetics → MultiSubjects → FineSports → ours (4 stages)
     - **Path A fallback** (if FineSports access falls through): Kinetics → MultiSubjects → ours (3 stages)

   - **Stage ordering rationale:** standard transfer principle = generic-task → specific-task, each stage's output domain becomes the next stage's input domain. Kinetics-400 (generic human motion) → MultiSubjects (amateur gym basketball, 3-class clip recognition) → FineSports (NBA basketball, temporal localization on continuous video — task structure closest to ours) → ours (deployment distribution). Task progression matters more than visual proximity; MultiSubjects → FineSports → ours preserves monotonic task-complexity growth.

   - **Stage 1 — Kinetics-400 pretraining (download only).** Use canonical PoseConv3D checkpoint from MMAction2 (`slowonly_r50_8xb16-u48-240e_k400-keypoint`). SlowOnly-R50 backbone, COCO-17 input, no first-conv modification needed (ball channel dropped per Decision 6).

   - **Stage 2 — MultiSubjects pretraining.** Full fine-tune (sufficient data — 6,144 clips). Use MultiSubjects's native 3-class labels; extend head to 4-class by adding "background" output (initialized but not actively supervised here since MultiSubjects clips are all action). Standard SGD recipe. Validation: random 10% held-out clips from MultiSubjects.
     - **Checkpoint here** → run MultiSubjects external benchmark eval (Decision 5)
     - **Checkpoint here** also serves as the Row 1 ablation point (Decision 5)

   - **Stage 3 — FineSports pretraining (Path C only).** Synthesize training windows from FineSports's continuous NBA video using their temporal localization labels:
     - Positive windows centered on labeled action timestamps + temporal jitter (±5 frames)
     - Negative windows sampled from non-action portions = `background` class supervision
     - All 4 classes get explicit supervision here (this is where background becomes a real class)
     - Full fine-tune. Validation: held-out videos from FineSports.
     - **Checkpoint here** → run FineSports external benchmark eval (Decision 5) at tIoU={0.3, 0.5, 0.7}
     - **Checkpoint here** also serves as the Row 2 ablation point (Decision 5)

   - **Stage 4 — Our continuous video, LP-FT recipe** (Kumar et al. 2022; 1% better in-distribution, 10% better out-of-distribution than full fine-tuning):
     - Training data construction:
       - **Positive windows**: 32-frame windows centered on hand-labeled release frames with ±5 frame temporal jitter (3–5 positive examples per labeled release). Each labeled `shoot` or `layup` (small added cost to labeling).
       - **Negative windows**: 32-frame windows sampled from non-release portions (no labeled release within ±1 second), all labeled `background`
       - Acknowledged tradeoff: `dribble` class performance on our distribution will degrade since we don't explicitly label dribble windows. Acceptable — we don't evaluate dribble on our data; only shot detection matters.
     - **Stage 4a (Linear Probe):** freeze entire backbone, train only the 4-class head until convergence (~10–20 epochs, LR ~1e-3 on head only). Standard cross-entropy loss.
     - **Stage 4b (Fine-Tune):** unfreeze backbone, train end-to-end at very low LR (~1e-5), early stopping on validation F1.
     - Validation: session-level held-out from our footage (per Decision 5 — training sessions ≠ val sessions ≠ test sessions, to avoid same-possession leakage).
     - **Pose-only checkpoint here** = Row 3 ablation point in Path C (Decision 5).

   - **Stage 5 — Ball-trajectory MLP training (two-stream head, both Path A and C).** After Stage 4 LP-FT completes, freeze the entire PoseConv3D backbone AND the pose-stream head; train only the new ball-trajectory MLP + combined 4-class linear head on top.
     - **Inputs:** 3D ball trajectory features from Decision 4 (`cv2.triangulatePoints` on WASB detections), computed per window — z-trajectory time series + summary statistics (max_z, time-to-peak, peak_v_z, ascent rate, z at window endpoints).
     - **Training data:** same synthesized windows as Stage 4 (positives centered on labeled releases + temporal jitter, negatives from non-release portions).
     - **Architecture:** small MLP (2-3 layers, hidden dim ~64) on ball features → concatenate with frozen pose-stream feature vector → new combined 4-class linear head.
     - **Loss:** standard cross-entropy on the combined head.
     - **Why train ball-head last and freeze everything else:** the pose stream has already done the hard work of learning shot motion from millions of frames of generic + basketball pretraining. The ball MLP only needs to learn the relatively simple mapping from ball-trajectory features to shot likelihood, then combine with the pose stream's confidence. Keeping the pose stream frozen prevents the small ball-stream from disturbing well-learned pose features.
     - **Full-model checkpoint** = Row 4 ablation point in Path C (or Row 3 in Path A) (Decision 5).
     - Literature support: badminton TrackNet+YOLOv7 paper showed 18pp F1 gain from adding ball-trajectory features to a player-action classifier — the strongest empirical evidence we have that this stream is worth training.

   - **Why LP-FT and not partial freezing?** Initially considered "freeze stem + ResNet stage 1, train rest" — that's an older heuristic. LP-FT is the current best-practice for small-data + distribution-shift settings. The linear-probe-first phase produces a near-optimal head before the fine-tune phase touches the backbone, which protects pretrained features from being distorted by gradient updates flowing through a randomly-initialized head. Particularly important here because we have meaningful distribution shift (MultiSubjects/FineSports → our amateur half-court 2v2 iPhone footage).

   - **Inference protocol (Decision 5 cross-reference):** sliding window over continuous video, 32 frames @ 60fps, stride 4. 4-way classifier output per window per (player, camera). Late-fuse across 3 cameras (average probabilities per player). Shot detected = peak in P(shoot) + P(layup) above threshold. NMS deduplicates adjacent peaks into single events.

8. **Release-frame extraction — 4-condition geometric rule on 3D pose+ball with confidence-aware logging.**

   - **What this stage does:** given a window flagged by the shot classifier (Decision 5), find the exact frame within that window where the shooter releases the ball. Output: single frame index per detected event. Every downstream metric (contest distance, release height, ★ biomech joint angles) is computed AT this frame, so frame precision matters — over 5 frames at 60fps the defender wrist moves ~15-25cm, comparable to the contest distances we're trying to measure.

   - **Why a rule, not a model:** release is a physical event with a clean geometric signature (ball separating from hand with upward velocity). Training a frame-precise model on hand-labeled releases would need denser supervision than we have. A hand-coded rule directly encodes the physics. Scope doc §3.2 already specified this.

   - **Inputs — 3D primary, 2D-per-camera fallback** (opposite of classifier in Decision 6 — appropriate here because the rule is geometric, not statistical; 3D positions directly answer "is the ball above the shoulder in real space?"). Camera-frame answers are ambiguous (ball may appear above shoulder from one angle while held at shoulder level). 2D fallback only if triangulation fails per Decision 4's 2-camera fallback logic.

   - **Identifying the shooter:** the player whose classifier output P(shoot)+P(layup) peaks during the event window IS the shooter. The rule operates on their 3D pose + the triangulated 3D ball.

   - **The 4 conditions** (rule fires at the FIRST frame in the classifier-flagged window where all four are satisfied simultaneously):
     1. **Ball above shoulder:** `ball.z > shooter.shoulder.z` (direct geometric comparison, no threshold)
     2. **Ball-wrist separation:** `ball.z − shooter.wrist.z > Δ_sep` (ball has left the hand)
     3. **Upward ball velocity:** `v_z > v_thresh` where v_z computed as smoothed finite difference on `ball.z` (rules out pump-fakes and overhead passes)
     4. **Sustained upward trajectory:** condition 3 holds for the next N frames (rules out brief lifts)

   - **Confidence-aware output, NO frame substitution.** Per detected event, log:
     - Release frame index (the frame the rule fired on)
     - Per-keypoint pose confidence at that frame (from RTMPose-l output)
     - Camera visibility count for triangulation at that frame (3/2/0 cameras, from Decision 4 logging)
     - Low-confidence events flagged but NOT relocated to nearby frames. Considered substituting a nearby higher-confidence frame; rejected because: (a) release frame has biomechanical meaning we'd lose by shifting (defender wrist position changes over a few frames; "shot was contested" measurement loses meaning); (b) sports science literature reports metrics AT release — shifting breaks comparability; (c) shooter at release is typically well-visible (mid-jump, fully extended); the genuinely problematic poses are usually defenders in contests, not shooters at apex.

   - **Considered and rejected: ball-height-alone with backward trace.** User raised the possibility of detecting shots purely from ball trajectory (find ball-height peaks, trace backwards to release). Rejected for shot DETECTION because: too many false positives (lobs reach above shoulder, rebounds bounce up, dunks have unusual trajectories). Ball-height signal IS used implicitly via the classifier's ball-trajectory MLP (Decision 5/6) and via the 4-condition rule's conditions 1, 3, 4. Player-pose context is what disambiguates real shots from non-shooting upward ball motion.

   - **Threshold tuning protocol** (concrete values determined empirically, not specified now):
     - **Tuning set:** ~20-30 hand-labeled releases from TRAINING sessions (NOT held-out test sessions per Decision 5)
     - **Grid search** over candidate (Δ_sep, v_thresh, N) values — e.g., Δ_sep ∈ {5, 10, 15, 20} cm × v_thresh ∈ {1, 2, 3, 4} m/s × N ∈ {3, 5, 7} frames
     - **Optimization target:** mean absolute frame error vs. labeled releases on tuning set
     - **Lock thresholds** before final evaluation
     - Velocity computation note: apply small temporal smoother to `ball.z` before finite-differencing — Pose2Sim's default Butterworth (6 Hz cutoff) per Decision 4 already does this for keypoints; same approach for ball z-trajectory.

   - **Failure-mode fallbacks:**
     - No frame satisfies all conditions → fallback: pick frame where `ball.z` reaches maximum within the window, flag low-confidence
     - Multiple satisfying frames → take FIRST (scope doc spec; release is the transition moment)
     - 2-camera triangulation in release window → continue with degraded accuracy, log weak-triangulation flag
     - Layup vs. jumper differences: start with same rule for both; add layup-specific variant only if evaluation shows systematic bias (layups release closer to body, may have weaker condition-1 signal)

   - **Evaluation:**
     - **Primary metric:** mean absolute frame error on held-out hand-labeled releases (separate from tuning set)
     - **Target: ≤3 frames mean error (~50ms @ 60fps)** — defensible accuracy floor for downstream contest metric (defender wrist position error ~10-15cm, small relative to typical 30-100cm contest distances). Earlier scope doc target of <2 frames recalibrated as ambitious; ≤3 frames is realistic.
     - **Acceptable for project usefulness: ≤5 frames (~83ms)** — beyond this, contest measurements get too noisy to distinguish tight vs. loose contests reliably
     - **Secondary metrics:** median frame error (robust to outliers), 95th percentile error (worst case), error histogram (catches systematic bias like always-late firing on layups)
     - **Baseline comparison:** naive ball-peak-height-in-window rule (just pick frame where `ball.z` is maximum in window). Shows the 4-condition rule earns its complexity. Compare mean frame error head-to-head.

9. **Contest-metric formulation — 5 measurements at release with novel 3D wrist-to-trajectory distance as headline metric; multi-dimensional output + optional composite.**

   - **What this stage does:** quantifies defender pressure on each detected shot release. This is the project's HEADLINE measurement — every other pipeline component exists to enable this. If the metric isn't meaningful, the project doesn't have a useful output.

   - **Literature positioning (Daly-Grafstein & Bornn 2020 — scope doc's closest prior work):** they used 3 features paired with shot trajectories — defender height, distance, and contest angle. Their distance metric is **floor distance to nearest defender** (NBA SportVU convention, 2D Euclidean). They had SportVU 3D ball + 2D players at 25Hz but chose floor distance because they didn't have defender hand positions. We do (via RTMPose-l + triangulation). This is the project's genuine novelty: **closest defender hand position relative to the actual ball trajectory** — mechanistically what causes a contested shot (defender hand intruding on ball's path forces shooter to alter release), measured directly rather than inferred from outcomes.

   - **Five measurements at release (multi-dimensional output, Option C):**

     **(1) Wrist-to-trajectory distance (headline novel metric):** minimum 3D perpendicular distance from any defender wrist to the ball's actual outgoing flight path through the contest volume. Signed (above/below trajectory) since under-the-ball doesn't impact shot as much as above-the-ball intrusion.
       - **Trajectory computation — empirical polyline (NOT parametric fit):** form polyline through measured 3D ball positions in frames T through T+12 after release (~0.2s @ 60fps). Empirical wins over parametric because: (a) no model assumptions; (b) no need to estimate release velocity (which has its own noise); (c) ball is in pure projectile motion post-release so the measured points already describe the truth; (d) air drag on basketball over 0.2s is < 2%, negligible; (e) the post-release flight is the EASIEST part of the game for WASB to detect (ball high, well-lit, away from shooter's body — fewest occlusions). Parametric fit is the backup if WASB loses the ball mid-flight.
       - **Time window — 0.2s post-release (~12 frames @ 60fps):** ball reaches ~1m above release in this window; defender's contesting hand is typically near release height; window captures the segment where wrist proximity is mechanically meaningful. Too short (<0.1s) misses physical intrusion; too long (>0.5s) is dominated by points far from contest. Tunable; can switch to "first 1m of trajectory above release height" if fixed time window proves arbitrary.
       - **For each defender:** compute min perpendicular distance from each wrist to polyline; take min across two wrists (closest contesting hand). Then take this value for the **closest defender** (per "which defender" below).
       - **Implementation note:** ~30 lines of Python with numpy. Geometrically simple; the hard work is upstream (triangulation, ball detection).
       - **If block actually happens** in the window: defender wrist literally on trajectory, distance ≈ 0 — correct measurement, not a failure mode.

     **(2) Wrist height advantage at release:** z-difference between defender's highest wrist and ball position at release frame. Positive = defender hand at-or-above ball. Captures vertical pressure independently of horizontal distance. Simple, interpretable, complementary to (1).

     **(3) Shooter velocity at release — direction-decomposed (user insight, supported by fadeaway literature):** pelvis velocity at release frame, decomposed:
       - **Forward/backward component:** projection of pelvis velocity onto the horizontal shooter-to-basket vector. Negative = fading away from basket = harder shot.
       - **Lateral component:** perpendicular projection. Captures sideways drift.
       - **Vertical component:** z-velocity (jump dynamics; known shooting factor).
       - **Why this matters:** captures "forced biomechanical adjustment" — defender presence causing shooter to fade, drift, or rush. Fadeaway literature confirms: backward lean elevates the release point but shifts center of mass rearward, varying release angle (avg 59° vs. ideal 45-55°). Defenders force these adjustments; measuring shooter response directly captures contest effect even when the geometric defender position misses something.
       - **Doesn't measure defender directly** — measures shooter's response, which is what ultimately matters for shot difficulty.

     **(4) Floor distance to closest defender (NBA-comparability baseline):** simple 2D Euclidean distance, shooter pelvis to closest defender pelvis at release. SportVU/Daly-Grafstein convention. Computed for direct comparison against literature — shows our 3D wrist-to-trajectory metric (1) provides information that this 2D number misses.

     **(5) Composite score:** unweighted z-scored combination of (1), (2), (3-forward component). Sign-corrected so "bigger = more contested" — distance components inverted, height-advantage as-is, shooter-fade-velocity such that more-fade = more-contested. **Weights are explicitly flagged as future work** — would need rater-correlation data or large make/miss dataset to learn weights; we don't attempt to validate composite weighting in this project. Provided as baseline only.

   - **Dropped from scope doc's original 3 measurements: body angle.** A side-stance defender can still contest with a single extended arm — body orientation may be misleading. The signal we care about (defender between shooter and basket) is captured implicitly by (1) wrist-to-trajectory and (4) floor distance. Removed for parsimony.

   - **Closest defender selection:** by 3D Euclidean distance from defender pelvis to BALL position at release. Cleaner than "to shooter wrist" because doesn't depend on identifying shooting hand reliably; ball position is unambiguous.

   - **Shooting hand identification:** closer of shooter's two wrists to ball position at release. Per user note, jump shots are single-handed so ball-proximity test is reliable; only layups can be ambidextrous — those use the same test (whichever hand is on the ball).

   - **Defender's contesting hand:** closer of defender's two wrists to ball trajectory. No need to identify dominant hand.

   - **Smoothing — both reported:**
     - **Instantaneous** at release frame (matches biomechanical-event definition; matches Decision 8's "no frame substitution" principle)
     - **Smoothed** over ±2 frames centered on release (5-frame mean — robust to triangulation noise)
     - Report both; lead with instantaneous. Substantial disagreement = flag shot as noisy.

   - **Edge cases:**
     - **No defender within 3m of shooter at release:** log "uncontested" rather than meaningless numbers. 3m chosen because beyond that even an extended arm can't physically affect the trajectory.
     - **Defender ID swap mid-shot (ByteTrack confusion):** detect via ID continuity check across shot window; flag affected shots.
     - **Weak triangulation in release window:** flag using Decision 4's per-keypoint visibility log.

   - **Validation (rater study DROPPED to future work per user direction):**
     - **(a) Mikic 2024 effect reproduction:** split our shots by contest metric (top vs. bottom quartile), check whether higher-contested shots have higher release point + quicker release. Same effect direction = construct validity. This is the strongest validation we can do without external rating.
     - **(b) Make/miss correlation:** higher contest metric should correlate with lower make rate. Spearman ρ between contest metric and make/miss binary outcome. Easy to compute from our manual make/miss labels.
     - **(c) Distribution sanity check:** ensure most shots have meaningful contest values; no degenerate failure modes (e.g., always-0 distance from triangulation collapse).
     - **★ Rater study moved to "future work" section** of milestone deck — not a project deliverable.

   - **Baseline for milestone:** NBA SportVU's floor-distance convention (measurement #4 above) IS the literature baseline. We compute it alongside the novel 3D wrist-to-trajectory metric (#1) and compare. Story: "floor distance is what NBA tracking measures; we show that 3D wrist-to-trajectory captures contest information that floor distance misses (e.g., defender 5ft away on floor with hand 2ft from trajectory is more contested than defender 3ft away with hand at hip)."

   - **Failure modes for wrist-to-trajectory specifically:**
     - WASB loses ball mid-flight: switch to parametric kinematic fit (`r(t) = r₀ + v₀·t + ½gt²`, gravity acts on z only) as backup
     - Triangulation degraded (2-camera fallback): increases polyline noise; flag with Decision 4 marker, still usable
     - Ball trajectory exits frame quickly: use whatever points we have; contest volume is first few frames anyway

10. **Calibration logistics — chessboard intrinsics (one-time per phone), court-landmark extrinsics (per session), AE/AF lock, reprojection-error drift monitoring.**

    - **What this covers:** the mechanics of camera calibration that feed Pose2Sim's triangulation (Decision 4). Calibration error propagates directly into every 3D measurement; ~1° extrinsic rotation error → ~5cm position error at 3m distance, which matters for the contest metric (Decision 9).

    - **Intrinsic calibration — chessboard, OpenCV `cv2.calibrateCamera`, one-time per phone.** Scope-doc default. Procedure:
      - Print chessboard (e.g., 9×6 inner corners, 30mm squares)
      - Record video on phone, moving board through ~20 positions/orientations across frame
      - Extract corners via `cv2.findChessboardCorners`; solve intrinsics via `cv2.calibrateCamera`
      - Save as `intrinsics_phone_<id>.json` in project repo
      - **Why chessboard not ChArUco:** ChArUco marginally better corner robustness but adds tooling dependency for negligible gain at our setup scale (3 phones, one-time)
      - **AVCameraCalibrationData stays ★ stretch** (requires custom Swift app + lookup-table conversion); chessboard is Phase 1.

    - **AE/AF lock per session, NO per-session intrinsic re-calibration.** iPhone autofocus and OIS effectively vary focal length frame-to-frame, which would invalidate fixed intrinsics. Operational fix: lock AE/AF on each iPhone aimed at court center before recording starts. Standard iPhone Camera app supports this. Removes autofocus drift; remaining OIS effects are sub-pixel and negligible for our triangulation accuracy targets. **Why this over per-session intrinsic re-calibration:** adds 30 seconds of operational discipline vs. ~30 minutes of per-session calibration work; gives equivalent stability.

    - **Extrinsic calibration — court landmarks via manual clicking, per-session, offline.** Scope-doc default. Procedure:
      - Per session, in the first frame of each camera's footage, manually click ~6-8 known court landmarks (free-throw line corners, key boundaries, 3-point arc intersections, half-court line endpoints) whose world coordinates come from standard court dimensions (NCAA/FIBA).
      - `cv2.solvePnP` solves for camera pose given landmark image positions + known world positions
      - Save as `extrinsics_session_<date>_<id>.json` per session per camera
      - **Why court landmarks over a chessboard on the court each session:** faster (~10 min total, all offline post-recording — doesn't add on-site time); world origin naturally centered on court (critical for biomechanics — "release height" is meaningful relative to floor, not relative to a chessboard at random court location)
      - **Why manual clicking over automated court-line detection:** manual ~30 sec per camera per session; automated line detection adds a CV dependency that could fail on partially-occluded courts or unusual lighting and would need its own validation. Not worth the complexity for ~6 landmark clicks.
      - **Considered: bundle adjustment with shared player keypoints** (Pose2Sim supports this) — useful as automatic refinement step on top of court-landmark initial calibration. Adopt if Pose2Sim's tooling makes it one-line; skip if it complicates things.

    - **Drift detection — reprojection-error monitoring (NO extra hardware).** Decision 4 already logs reprojection error per-frame. Sudden jumps above ~2× session baseline indicate calibration drift (bumped tripod, etc.). Procedure:
      - Compute baseline reprojection error on first ~30 seconds of session
      - Flag any 30-second window where mean reprojection error exceeds 2× baseline
      - For flagged segments: either re-extrinsic-calibrate from a landmark frame within the new segment, OR exclude the segment from analysis
      - **Why this over periodic static-marker re-check:** no extra physical setup; uses signal we already produce; catches drift even without operator awareness of a bump

    - **Per-session operational sequence:**
      1. Position tripods, lock AE/AF on each iPhone aimed at court center, set 60fps recording
      2. Start all 3 recordings
      3. Audio clap → inter-camera time sync (Decision 4)
      4. Record play
      5. Stop recordings
      6. **Offline:** manually click court landmarks in first frame per camera → extrinsics; run drift monitoring on triangulation outputs
      - Total on-site setup: ~5 min. Offline calibration: ~10 min/session. No 30-minute chessboard setup per session.

    - **No baseline comparison needed.** Calibration is infrastructure, not a contribution. Evaluation is implicit via Decision 4's reprojection error target (< 5px on held-out camera).

    - **Storage and reproducibility:** all calibration JSONs versioned in project repo. Pose2Sim consumes them via its standard config-file interface. Naming convention specified above.

## All decisions locked. Working session complete.

**Status:** All 10 technical-approach decisions for Milestone 2 logged with rationale, supporting literature, alternatives considered, fallback plans, and evaluation criteria.

**Next step:** translate logged decisions into Milestone 2 slide deck. Suggested slide structure follows the milestone requirements (method overview / design intuition / baselines / experimental setup / implementation plan).

## Status

Working through one at a time. User asks questions, I present options, user decides.
