# CourtVision Session Journal — Phase C Through MultiSubjects Evaluation

**Span:** 2026-05-27 → 2026-06-08

## About this document

This is a **session journal**, not a verbatim transcript. I (Claude) do not have API-level access to the raw chat history, and our session was long enough that earlier portions were likely compressed by the runtime — so reconstructing exact wording would mean guessing. Instead, this is a faithful **chronological record of the questions asked, decisions made, work performed, and results obtained** in this Claude Code session, with concrete pointers to the authoritative records in the repo.

This is a *companion* to the parallel session's `Claude/transcript.md` (which covers the Phase D pivot, Stage 5 detector-fix work, and Stage 6 release-frame detector). The two sessions ran concurrently on the same repo and coordinated through shared docs.

**Where the authoritative content lives:**
- `docs/decisions.md` — design decisions (Decisions 1–10)
- `flags.md` — flag log (F1–F26)
- `docs/sprint_plan.md` — phase status
- `docs/stage2_evaluation_report.md` — report-ready Stage 2 eval consolidation
- `~/.claude/projects/.../memory/` — cross-session memories

---

## Scope of this session

This session focused on:
1. Phase B closure (Stage 1 detection + tracking)
2. Phase C end-to-end (Stage 2 2D pose) — planning, dependency hell, rtmlib pivot, evaluation, closure
3. Phase D drop strategic implication → Stage 7 contest metric redesign
4. MultiSubjects pose evaluation (secondary corroboration for the report)
5. TrackID3x3 Outdoor/Drone exploration
6. Report-ready consolidation

The parallel session handled Phase D pivot (ball detection deferred), Stage 5 classifier retrain, and Stage 6 release-frame detector. Cross-session coordination was through shared files (`modal_app.py`, `flags.md`, `decisions.md`) with explicit commit hygiene to avoid entangling workstreams.

---

## Timeline

### Phase B Closure (2026-05-27)

User opened the session by asking what was last worked on. I traced recent commits and identified the Stage 1 track post-processing pipeline (track stitching + non-player speed filter, calibrated against TrackID3x3 Indoor).

Deferred persistence-layer options were logged to `flags.md` F15 and memory (`project_stage1_deferred_backlog`): A2 court mask, A3 team-colour, B2 BoT-SORT Re-ID, C2 cross-camera (highest leverage), D1 jersey OCR, D2 custom Re-ID. C2 was flagged as the most valuable revisit when own footage exists.

**Phase B capstone:** commit `6701277` "Phase B: Stage 1 detection + tracking closed — gate met (test IDF1 0.785)".

### Phase C Planning (2026-05-27)

User requested a comprehensive Phase C plan. I re-read the PRD (§9.2 Stage 2, §10 Phase C, §15 execution protocol) and proposed a file-level plan covering inference, evaluation, configs, scripts, and tests.

**Key planning decisions:**
- TrackID3x3 chosen as primary eval dataset (deployment-domain fixed-cam multi-player, despite model-derived GT noise) — see decisions.md Decision 2.
- Verified TrackID3x3 Indoor pose GT structure locally on samples: jersey-color-keyed, dense (all 6 players × all frames), 10 keypoints with the F5 anatomy (wrists at idx 5/6).
- Verified kinematically that idx 5 is anatomically nearer the R-elbow (idx 3) and idx 6 nearer the L-elbow (idx 4) — confirms F5's mapping. ~80–90% kinematic consistency, with the remaining ~10–20% reflecting hands-meeting-at-body cases (which motivated the side-agnostic 2×2 wrist assignment in the eval).
- Provenance check: coordinates carry heavy subpixel float precision, consistent with model output, not hand annotation.

**Initial Phase C commit (MMPose path):** `6512e08` "feat(stage2): RTMPose-l Halpe-26 inference + wrist-PCK eval (Phase C)".

### Dependency Hell (2026-05-27 → 2026-05-28)

The OpenMMLab installation saga. I attempted ~10 iterative Modal image rebuilds chasing failures:

1. **numpy ABI mismatch** at `xtcocotools._mask` import (`ValueError: numpy.dtype size changed`). xtcocotools wheel built against numpy 2.x; image had numpy 1.26 (per pyproject pin). F20.
2. Pinned `xtcocotools<1.14` to get numpy-1.x wheel — no cp311 wheel, pip tried source build, **`maskApi.c` missing from sdist** (known xtcocotools 1.13.0 packaging bug).
3. Forced numpy 2.x in the image — same `dtype size 96 vs 88` error even on numpy 2.4.6. Diagnostic revealed: xtcocotools wheel built against a *different* numpy 2.x ABI than the one installed.
4. **Source-compiled xtcocotools** against the image's numpy with `--no-binary` + `--no-build-isolation` + build-essential — got past the numpy ABI wall. ✓
5. Hit next blocker: **mmpose hard-imports `mmdet`** at package init (RTMO head). Added mmdet.
6. Hit next blocker: **mmdet 3.x asserts `mmcv<2.2.0`**, but only mmcv 2.2.0 builds against the image's torch. Attempted sed-patch the cap.
7. Sed hit a chicken-and-egg: my command tried to `import mmdet` to find its path, but `import mmdet` is exactly what the assert blocks.
8. Fixed to fixed-path sed — but **mmdet versions were resolving non-deterministically** across builds (line numbers in `__init__.py` differed), so the sed sometimes patched a different mmdet than the run got. Pinned `mmdet==3.3.0`.
9. Forced a clean rebuild (`MODAL_FORCE_BUILD=1`) — **mmcv source build failed on `pkg_resources`** (modern setuptools dropped it; this is the F1 issue manifesting on a fresh build).
10. Pinned `setuptools<80` + `--no-build-isolation` for the mm-stack install — mmcv built — but the mmdet cap reasserted, with non-deterministic resolution again (sed patched one mmdet, run got another).

**At this point user stopped me:**
> "Before you do anything, please conduct some deep research and determine if this is the right approach. you should look at academic journals and trusted forums to see how others might have approached the issues you are facing. I don't want you to keep trying things and wasting my time if you can pre-determine that an approach isn't going to work."

This became one of the most important methodological pivots of the session, and the basis for the `feedback_research_before_iterating` memory.

### Research Findings (2026-05-28)

WebSearch + WebFetch of:
- mmcv compatibility matrix
- GitHub issue **open-mmlab/mmcv#3310** ("mmcv._ext not getting built with PyTorch 2.9.0 and Cuda 13.0") — open since Oct 2025, **no maintainer response**
- rtmlib (Tau-J/rtmlib) — RTMPose without mmcv/mmpose/mmdet

**Conclusion:** The OpenMMLab stack is effectively unmaintained against the image's bleeding-edge torch 2.12 + CUDA 13.0. mmcv's last release (2.2.0, April 2024) only ships prebuilt wheels up to torch ~2.1; #3310 confirms `mmcv._ext` doesn't compile on torch 2.9+. Even forcing imports through would likely leave the CUDA ops broken at runtime. The remaining `mmcv==2.1.0` attempt was **predetermined not to reliably work**.

`rtmlib` is purpose-built for exactly this escape: depends only on `numpy/opencv/onnxruntime`, no torch/mmcv/mmpose/mmdet, supports RTMPose-l Halpe-26 ONNX with bbox-conditioned top-down inference. I verified the API signature in the rtmlib source: `RTMPose(onnx_model, model_input_size, backend, device)(image, bboxes=[...])`.

### Rtmlib Pivot (2026-05-28)

User approved the PRD §5.2 deviation: same RTMPose-l Halpe-26 model + weights, different runtime.

Work performed:
- Rewrote `src/courtvision/stage2_pose/rtmpose_inference.py` to use rtmlib's `RTMPose` class with the ONNX URL `rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip` (verified HTTP 200).
- Removed the mm-stack from the shared Modal image (it was dead weight for Stages 1/3 and unbuildable for Stage 2/E anyway).
- Added a lightweight `rtmpose_image` (debian_slim + ffmpeg/opencv libs + `rtmlib, onnxruntime, opencv-python-headless, numpy, pydantic, rich`). Builds in ~1 min.
- Updated config + scripts to use ONNX URL + backend/device params.
- Logged the pivot in `decisions.md` Decision 2 ("Phase C engine swap"), `flags.md` F20 (postmortem), and memory (`project_openmmlab_torch_incompat`).

**First Modal run on the new path succeeded end-to-end:** rtmlib loaded the ONNX, ran inference on `basket_S4T4_post`, emitted 1,257 poses across 236 frames. **Dependency hell resolved.**

Initial single-clip wrist PCK: 0.627.

### Phase C Evaluation (2026-05-28)

User pushed back on partial evaluation (had done 10 of 13 available Indoor pose-GT clips):
> "why haven't you run a more comprehensive evaluation on the trackID3x3 data? why not evaluate on all of them?"

And separately:
> "we only care about wrists, shouldn't we make a PCKw metric that looks only at wrists?"

I clarified that my `evaluate_trackid3x3` already implemented wrist-only PCK as the headline gate (so PCKw was already what was reported), but agreed I should:
1. Extend to ALL 13 Indoor pose-GT clips.
2. Add a per-landmark + all-keypoint diagnostic alongside (so we'd know if wrists were uniquely weak or just tracking the GT-disagreement floor).

Implemented:
- `_GT_LANDMARK_GROUPS` constant mapping TrackID3x3's 10 GT keypoints → Halpe-26 indices, grouped by landmark type (head, shoulder, elbow, wrist, hip, ankle).
- `_group_distances` helper — side-agnostic 2×2 for L/R pairs, direct for central.
- Extended `evaluate_trackid3x3` to return per-landmark + all-keypoint PCK alongside the wrist gate.
- New tests including a per-landmark perfect-prediction sanity test.

**Comprehensive eval on all 13 Indoor clips:**

| Landmark | PCK@0.05 |
|---|---:|
| shoulder | 0.941 |
| ankle | 0.845 |
| hip | 0.802 |
| elbow | 0.727 |
| head | 0.675 |
| **wrist (gate)** | **0.655** |
| all-keypoint | 0.781 |

Per-clip wrist mean **0.648 ± 0.043** across 13 clips. Stable.

**The per-landmark diagnostic was the decisive interpretive tool:**
- Shoulder at 0.94 proves the matching, normalization, and coordinate handling are correct — a broken eval can't score any joint that high.
- The numbers are a textbook proximal→distal falloff (shoulder 0.94 → elbow 0.73 → wrist 0.66) — wrists are the hardest joint, not uniquely broken.
- TrackID3x3 GT is model-derived (F18), so PCK measures *agreement with another estimator* at a strict ~12 px tolerance.
- The 0.85 bar is therefore inappropriately strict for distal joints against this GT.

User watched the 13 wrist-overlay videos and confirmed:
> "Qualitatively, wrist location looks pretty accurate, especially on shot frames which is the key thing."

**Phase C accepted via documented deviation.**

### Phase C Closure Logging (2026-05-28)

- `docs/decisions.md` Decision 2 extended with the Phase C engine swap rationale and the Phase C result + gate disposition.
- `flags.md` F20 — full postmortem of the dependency-hell saga.
- `flags.md` F21 — Phase C gate disposition (quantitative FAIL accepted via diagnostic + qualitative).
- `docs/sprint_plan.md` — Phase C ✅.
- Memory:
  - `project_phase_c_closed.md` — Phase C done; cross-dataset disposition rationale.
  - `feedback_research_before_iterating.md` — the key methodological lesson.
  - `feedback_comprehensive_evaluation.md` — eval-coverage and metric-rigor preferences.
  - Updated `project_openmmlab_torch_incompat.md`.

**Capstone commit:** `cc31cb6` "feat(stage2): rtmlib/ONNX RTMPose engine + per-landmark PCK eval; close Phase C".

This commit deliberately excluded `modal_app.py` (the run_rtmpose wiring + rtmpose_image): the parallel session had concurrent edits and an unfixed mypy error there (`unused "type: ignore"` at line 426 in `verify_wasb_checkpoint`) which would have failed the pre-commit hook. The Stage 2 logic was self-contained enough to commit cleanly without it.

### Phase D Drop (2026-06-02)

User: "I don't want to worry about phase D (maybe ever). so what's next"

Strategic implication I flagged: the PRD's headline Stage 7 metric (3D defender-wrist distance to ball trajectory) loses the ball component entirely. Stage 7 needs a pose-only redesign or it's not a runnable deliverable.

I recommended: pose-only Stage 7 contest metric. Two alternatives (help finish Phase E; start Stage 4/own footage protocol) were less leverage-positive given the parallel session's coverage of E/6.

### Stage 7 Contest Metric Redesign (2026-06-02)

User specified the components:
> "we will do wrist-to-wrist measurement as a proxy. we will still examine relative wrist locations in a manner that predicts if the defender's hand could be in or close to the ball's trajectory (estimate) and has thus affected the shot. ... consider both distance wrist to wrist and each wrist height to understand if the defender caused shot alteration. we should also measure the shooter's velocity to determine if they are fading away"

Then refined the velocity:
> "for (3), we don't so much care about vertical. that will be captured by (2). we care only if the shooter is fading in a direction away from a line to the basket (left, right, back). Forward is natural so can be represented as 0 fade."

Final design:

| | Definition |
|---|---|
| `wrist_to_wrist_distance_m` | 3D distance, closest defender's nearer wrist ↔ shooter's shooting wrist at release. |
| `wrist_height_advantage_m` | `defender_wrist_z − shooter_wrist_z`. Positive = at/above release point. |
| `shooter_fade_mps` | `sqrt(max(0, −v_forward)² + v_lateral²)` in the court frame. Forward toward rim = 0. Vertical dropped (handled by height advantage). |
| `floor_distance_m` | Unchanged (NBA-comparable). |
| `composite_score`, `flags` | Unchanged structure. |

**Schema changes** (`src/courtvision/schemas.py::ContestMeasurement`):
- `wrist_to_trajectory_distance_m` → `wrist_to_wrist_distance_m`
- 3 `shooter_velocity_*` fields → 1 `shooter_fade_mps`
- `wrist_height_advantage_m` semantics changed (vs shooter wrist, was vs ball)

`tests/test_schemas.py::test_contest_measurement_sequence` updated. `docs/prd.md` §7.7 intentionally not edited — PRD remains the source of truth for the original spec; deviations live in `decisions.md` and `flags.md`, matching the Decision 2 precedent.

**Commit:** `adaaf53` "feat(schemas): pose-only Stage 7 contest metric — Phase D-drop redesign". Decisions.md Decision 9 sub-bullet "Phase D-drop pose-only redesign" + flags.md F25.

### MultiSubjects Description Q&A (2026-06-02)

User asked for specifics on MultiSubjects + how Stage 5 was actually trained.

I verified per-clip + per-action counts from the local `MultiSubjects_person_GT` splits and confirmed:
- **6,147 unique clips, 1,001 subjects, 3 actions:** dribble 2,328 (38%), shoot 3,046 (50%), **layup 773 (13% — heavily under-represented)**.
- Split: train 3,073 / val 1,025 / test 2,049.

On actual Stage 5 training (from the parallel session's `project_stage5_detector_fix` memory + the Stage 5 code I had access to):
- Pipeline as-run = **Kinetics-400 pretrained PoseConv3D → MultiSubjects fine-tune only** (Path A, not the PRD's Path C). FineSports skipped (GT-pkl access never granted; user assumed no access). Own footage stage skipped (no iPhone clips yet).
- v1 (32f UniformSampleFrames over whole clip): val top-1 78%; **−21 pts on sliding contiguous windows** (57%) = the train↔deploy sampling gap.
- v2 (`stage2_multisubjects_64f_bal.pth`): 64f contiguous SampleFrames, inverse-freq class weights (layup ~2.6×), label smoothing 0.1, discriminative LR.
- v2 metrics: overall per-window 0.706; layup 0.485 (per-clip 0.575); shoot ~0.95; false-shoot 0.23; train↔deploy gap closed to ~7 pts.
- TrackID3x3 OOD eval (no shot GT, qualitative): user reviewed 13 overlays, confirmed precision ≈ 0.75 / recall ≈ 0.75 at thr 0.75.

Caveats made explicit for the report writeup:
- Path C collapse to Path A (FineSports unavailable).
- No background class (deferred until own footage; synthetic background unvalidatable).
- No Path B (proper temporal action detector) until own footage exists.

### MultiSubjects Pose Eval (2026-06-02)

User: "we should absolutely compare our stage 2 accuracy (for all coco positions) against the multisubject GT. why have we not done this?"

I acknowledged honestly: I'd shipped the code but never run it (the Phase C gate was already disposed of via TrackID3x3 + qualitative). User said run it.

Work performed:
- **Upgraded `evaluate_multisubjects`:**
  - Per-keypoint COCO-17 PCK breakdown (was overall + wrist only).
  - Multi-track per-frame matching by COCO-centroid closeness (handles Stage 1 emitting stray non-player tracks; e.g., a coach in frame).
  - Per-clip pixel resolution read from each clip's video file via cv2 — closes the F18 "MS coord scale unverified" loose end. Signature backward-compatible.
- **New `scripts/stage2_ms_batch.py`:** extracts requested val clips' videos + BlazePose CSVs from the on-Volume `.rar` archives, runs Stage 1 + Stage 2 in parallel Modal waves, runs the eval. One entrypoint.
- New test exercising multi-track matching + per-keypoint output on synthetic data.

**Stratified sampling from val (counted from local `new_val.csv`):**
- Sanity batch: 30 clips (10 per action). Overall PCK 0.864 — pipeline verified end-to-end.
- Report batch: 300 clips (100 per action) stratified-random. All 300 evaluated, **362,695 keypoints scored**.

**Result:**

| | PCK@0.05 |
|---|---:|
| Overall (all 17 COCO) | **0.841** |
| Wrist (mean L + R) | **0.680** |

Per-keypoint table (the headline for the report): nose 0.91, l_eye 0.90, r_eye 0.90, l_ear 0.83, r_ear 0.84, l_shoulder 0.89, r_shoulder 0.89, l_elbow 0.79, r_elbow 0.77, **l_wrist 0.69, r_wrist 0.67**, l_hip 0.86, r_hip 0.86, l_knee 0.88, r_knee 0.87, l_ankle 0.87, r_ankle 0.86.

Per-action overall (n=100 each): dribble 0.844 ± 0.157, layup 0.799 ± 0.116, shoot 0.798 ± 0.187. Outliers: 3 of 300 clips scored 0.0 (Stage 1 missed the player), negligible on aggregate.

**Commit:** `45babe1` "feat(stage2): MultiSubjects per-keypoint COCO-17 PCK secondary eval". F26 logged.

### Cross-Domain Comparison (2026-06-02)

User: "did we already do a comparison of our stage 1 + 2 keypoints compared to trackID3x3 indoor GT? sounds like we have LOTS of frames to compare against."

Confirmed yes (Indoor n ≈ 143,000 keypoints — 28,692 wrist counts × the 10 GT keypoints / 2 wrist keypoints), and produced the side-by-side:

| Landmark | Indoor | MultiSubjects (agg to Indoor groups) | Δ |
|---|---:|---:|---:|
| shoulder | 0.941 | 0.892 | −0.049 |
| ankle | 0.845 | 0.864 | +0.020 |
| hip | 0.802 | 0.857 | +0.054 |
| elbow | 0.727 | 0.781 | +0.055 |
| head | 0.675 | 0.909 | +0.233 *(mapping caveat)* |
| **wrist** | **0.655** | **0.680** | **+0.025** |

**5 of 6 landmarks converge within ~0.06** on two completely different domains (multi-player half-court vs single-player gym), camera setups, and GT-providing models (TrackID3x3 baseline vs MediaPipe BlazePose). The head gap is the known mapping artifact (TrackID3x3 head ≈ top-of-head, mapped to Halpe nose for the eval).

**Combined coverage:** ~506,000 keypoints across the two datasets. The wrist convergence (0.655 ↔ 0.680) is the decisive corroboration for the Phase C gate-disposition argument.

### Report-Ready Consolidation (2026-06-02)

User: "store all of this information somewhere so we can use it on the final report"

Created **`docs/stage2_evaluation_report.md`**:
- Executive summary
- Methodology (pipeline, PCK definition, multi-track matching, side-agnostic wrist rationale)
- Datasets table
- Indoor per-landmark table
- MS per-keypoint table + per-action breakdown
- Side-by-side cross-domain comparison (the headline table)
- Interpretation (eval soundness, proximal→distal, cross-domain convergence, per-action stability)
- Caveats (model-derived GTs, head mapping, L/R convention, MS outliers)
- Phase C gate disposition restated for the report
- Reproducibility pointers (commands + artifact paths)

**Commit:** `4da06ad` "docs(stage2): report-ready Stage 2 evaluation summary".

### TrackID3x3 Outdoor / Drone Exploration (2026-06-08)

User asked what's actually in the unused TrackID3x3 subsets.

**Outdoor:** Volume listing showed only **1 pose-GT file** (`IMG_0104_.json`, iPhone naming, different from Indoor's `basket_SxTy_post`). User dropped immediately: "i don't care about outdoor if there's only 1 clip. forget about it."

**Drone:** Volume recon revealed:
- 4 source game videos (`1-1.mov, 4-1.mov, 5-1.mov, 6-1.mov`) at **3840×2160 (4K)**, each 6,000 frames, plus pre-split sub-clips per game.
- MOT bbox GT organized per game (`MOT/<game>/<start>_<end>_gt.txt`).
- **Pose GT: 500 per-frame JSONs** (`frame_1.json` … `frame_500.json`) at the same 4K resolution, with the same 10-keypoint anatomy as Indoor, but with **richer player labels** (jersey colour + number, e.g. `Black_1, Black_10, White_0, White_9`).

**The catch:** the 500 pose JSONs use `file_name: "frame_N"` with no documented mapping back to which source video or which frame. So we have rich aerial labels but no obvious image source. Plus the aerial view is genuinely OOD for RTMPose (heads occlude shoulders; arms collapse onto torsos).

I described both Indoor and Drone exactly, with the frame-mapping problem flagged for Drone as the major practical blocker. User then asked whether we'd already done the Indoor per-keypoint eval (we had — Phase C) and the cross-domain side-by-side was sufficient for the report's robustness story. **No new eval runs were launched in this final exchange.**

### This Document (2026-06-08)

User: "can you please add the full transcript of this conversation to the Claude folder on my CourtVision Github? make a clean markdown file and push it there. the repo is private"

I noted the limitation (can't produce a verbatim transcript), offered three options, user chose option 1 (session journal). This file is the result.

---

## Commits in this session

| Hash | Message |
|---|---|
| `6701277` | Phase B: Stage 1 detection + tracking closed — gate met (test IDF1 0.785) |
| `6512e08` | feat(stage2): RTMPose-l Halpe-26 inference + wrist-PCK eval (Phase C) |
| `cc31cb6` | feat(stage2): rtmlib/ONNX RTMPose engine + per-landmark PCK eval; close Phase C |
| `adaaf53` | feat(schemas): pose-only Stage 7 contest metric — Phase D-drop redesign |
| `45babe1` | feat(stage2): MultiSubjects per-keypoint COCO-17 PCK secondary eval |
| `4da06ad` | docs(stage2): report-ready Stage 2 evaluation summary |

---

## Key decisions made in this session

1. **rtmlib/ONNX over MMPose for Stage 2 inference** (PRD §5.2 deviation, 2026-05-28). Same RTMPose-l Halpe-26 model. Reason: OpenMMLab stack unmaintained against the Modal image's torch 2.12 + CUDA 13 (mmcv#3310). Authoritative: `decisions.md` Decision 2, `flags.md` F20.
2. **Phase C accepted via documented deviation** (2026-05-28). Wrist PCK 0.655 vs 0.85 gate. Reason: shoulder 0.94 proves eval correct; proximal→distal falloff is the textbook pattern; model-derived GT makes 0.85 wrong bar; user-confirmed qualitative wrist overlay accuracy especially on shot frames. Authoritative: `decisions.md` Decision 2 ("Phase C result + gate disposition"), `flags.md` F21.
3. **Phase D dropped entirely** (2026-06-02). No ball detection this project. Phase 2 ★ escalation path documented if own iPhone footage allows revisit. Authoritative: `flags.md` F19/F20 (existing, from parallel session) + the Stage 7 Decision-9 sub-bullet.
4. **Stage 7 contest metric redesigned pose-only** (2026-06-02). Components: `wrist_to_wrist_distance_m`, `wrist_height_advantage_m`, single `shooter_fade_mps` scalar. Schema rename + condense. Authoritative: `decisions.md` Decision 9 sub-bullet "Phase D-drop pose-only redesign", `flags.md` F25, `schemas.py`.

---

## Headline results from this session

| | Wrist PCK@0.05 | All-keypoint PCK | n keypoints |
|---|---:|---:|---:|
| TrackID3x3 Indoor (13 clips) | 0.655 | 0.781 | ~143,000 |
| MultiSubjects val (300 clips) | 0.680 | 0.841 | 362,695 |
| **Combined** | | | **~506,000** |

Cross-domain wrist convergence: **0.655 ↔ 0.680** on completely different camera setups, GT providers, and player counts. This is the corroboration that allowed accepting the 0.85 gate as inappropriately strict for distal joints against model-derived GT, rather than as a Phase C miss.

---

## Datasets used (this session's work)

- **TrackID3x3 Indoor** — 13 pose-GT clips of 42 available Indoor clips, full pose evaluation; MOT bbox GT used in Phase B (closed earlier). Multi-player half-court fixed cam. License: CC BY 4.0. Source: open-starlab gdown.
- **MultiSubjects** — 300 stratified val clips (out of 6,147 total). BlazePose-33 per-frame pose GT subset to COCO-17. Single-player gym drills. License: CC BY-NC 4.0. Source: HuggingFace.

**TrackID3x3 Outdoor (1 clip)** was discussed and dropped. **TrackID3x3 Drone (500 per-frame pose JSONs)** was structurally mapped but not evaluated because of the unsolved frame→video mapping and aerial-view OOD concern.

---

## Methodological lessons captured in memory

- `feedback_research_before_iterating.md` — On hard dependency/build issues, research authoritative sources to predetermine viability rather than burn time on unbounded trial-and-error. Born directly from the ~10-rebuild mmcv chase. Counterpart to the existing `feedback_dont_ask_to_proceed` (act on clear tasks; research on uncertain ones).
- `feedback_comprehensive_evaluation.md` — Evaluate on ALL available GT (not a convenient subset); make metrics interpretable + task-specific; characterise GT provenance (model-derived vs hand-labelled changes how to read scores).
- `project_phase_c_closed.md` — Phase C done; the gate disposition with the per-landmark diagnostic + qualitative reasoning + the cross-dataset corroboration finding.
- `project_openmmlab_torch_incompat.md` — Don't re-attempt mmpose/mmaction2 on the shared image's torch 2.12; pivot to ONNX (rtmlib) for pose; isolate mmaction2 in a pinned-torch image for Phase E.

---

## Cross-session references

The parallel session — running concurrently on the same repo — handled the Phase D pivot (ball detection deferred), the Stage 5 classifier retrain (`64f_bal`), and the Stage 6 release-frame detector (pose-only `pose_apex` / `pose_forward`). That session's transcript is at `Claude/transcript.md`.

Shared files (`modal_app.py`, `flags.md`, `decisions.md`) were the main coordination point. Commit hygiene in this session: I deliberately did not commit `modal_app.py` (had an unfixed mypy error from the parallel session that would block the pre-commit hook), and was explicit in commit messages about what was and wasn't included.

All cross-session-authoritative content lives in repo:
- `docs/decisions.md` (Decisions 1–10)
- `flags.md` (F1–F26)
- `docs/sprint_plan.md`
- `docs/stage2_evaluation_report.md`
- `~/.claude/projects/.../memory/MEMORY.md` (and individual memory files)
