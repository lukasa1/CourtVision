# Flags

Living log of items flagged during phase execution. Each flag is appended when it's raised; updated or marked **RESOLVED** when it's addressed. Subsequent phases append to this file rather than starting a new one.

Format per flag:
- **Status:** OPEN / RESOLVED / DEFERRED
- **Phase raised:** which phase surfaced this
- **Description:** what the issue is
- **Why it matters:** consequence if left unaddressed
- **Resolution / next step:** what needs to happen and when

---

## F1 — OpenMMLab stack is Modal-only, not in `pyproject.toml`

- **Status:** OPEN (intentional; PRD-aligned mitigation in place)
- **Phase raised:** A
- **Description:** `mmengine`, `mmcv>=2.0`, `mmpose>=1.3`, `mmaction2>=1.2` were removed from `pyproject.toml`'s `dependencies` list during Phase A. Reason: `mmcv 2.2.0` has an undeclared `pkg_resources` build dependency and no macOS arm64 wheel, so `uv lock` / `uv sync` fail locally on Lukas's Mac. The `extra-build-dependencies` workaround that uv suggests didn't take effect for this package.
- **Why it matters:** PRD §5.2 lists these as core deps. Local Phase A code doesn't import any of them, so the deviation is invisible until Phase C/E tries to use them. The Modal image installs them via `pip_install(...)` in `src/courtvision/modal_app.py`, which is where the real ML runs — so production paths are unaffected. **PRD §13 risk #1 explicitly endorses this mitigation:** *"Use the Modal image as canonical; replicate locally only as needed."*
- **Resolution / next step:** Leave as-is unless a future phase needs to run mm-stack code locally (none currently planned in the sprint). If that need arises, options: (a) install via `pip install --no-build-isolation`, (b) use a Linux dev container, (c) pin mmcv to a version with prebuilt arm64 wheels (mmcv 2.1.x has some). First Modal image build will be slow as the stack installs from scratch; subsequent runs cached.

---

## F2 — Dataset source URLs not yet filled

- **Status:** OPEN (partial — see Progress)
- **Phase raised:** A
- **Description:** `scripts/download_datasets.py` has one Modal function per dataset. Each function body was originally `raise NotImplementedError("fill source URL during Phase A user-test step")` with the canonical source noted in a comment.
- **Why it matters:** Phases B/C/D/E need datasets on the `courtvision-data` Modal Volume. Without URLs filled in, `modal run scripts/download_datasets.py::download_*` won't actually download anything.
- **Progress (2026-05-26):**
  - `download_multisubjects` — RESOLVED (HF snapshot; on Volume).
  - `download_finesports` — RESOLVED for the frames zip (HF snapshot; on Volume). Fine-grained labels + bboxes + splits pending the FineSports-GT.pkl email request (separate; tracked in project memory).
  - `download_trackid3x3` — RESOLVED (gdown from public Google Drive folder per `open-starlab/TrackID3x3`).
  - `download_u_arizona` — REPLACED by `download_deepsportradar` (primary) + `download_roboflow_basketball` (secondary). See F12.
  - `download_wasb_basketball` — DEFERRED (upstream 404). See F11.
- **Resolution / next step:** Treat F2 as effectively closed for Phase A scope; remaining loose ends tracked under F11 (WASB) and F12 (U Arizona substitution) and the FineSports-GT.pkl email memory.

---

## F3 — Checkpoint download URLs not yet filled

- **Status:** RESOLVED (2026-05-26) — all 4 checkpoints on `/artifacts/checkpoints/`. WASB sanity-load **confirmed 2026-05-27** (Phase D, `modal_app.verify_wasb_checkpoint`); RTMPose-l / PoseConv3D sanity-loads still pending Phase C/E first-use.
- **Phase raised:** A
- **Description:** `scripts/download_checkpoints.py` has stubbed functions for the four pretrained checkpoints. YOLOv8m is trivial (ultralytics auto-downloads); the other three (RTMPose-l Halpe-26, WASB basketball, PoseConv3D K400) need their model-zoo URLs or HF repo IDs filled in.
- **Why it matters:** Phases B/C/D/E configs reference `/artifacts/checkpoints/<weight>.pt[h]`. If the file isn't on the Volume, the stages can't run.
- **Resolution:** All four downloaded. Actual sizes are smaller than the PRD's estimates for WASB (6 MB vs 200 MB est.) and PoseConv3D K400 (13 MB vs 250 MB est.) — possibly weights-only saves without optimizer state, or compact architecture variants. Not necessarily wrong, but worth a `torch.load` sanity check at first use in Phase C/D/E before assuming they're complete.
  - `yolov8m.pt` (52 MB) — ultralytics auto-fetch (`YOLO('yolov8m.pt')`)
  - `rtmpose_l_halpe26.pth` (113 MB) — MMPose `rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192`
  - `wasb_basketball.pth` (6.1 MB) — `nttcom/WASB-SBDT` MODEL_ZOO Drive link, via gdown single-file. **Sanity-load passed (2026-05-27):** loads into WASB's HRNet with 0 missing / 0 unexpected keys; 1,481,427 params (~5.9 MB fp32 ≈ the 6.1 MB file — so it's the *complete* model, not a truncated download). WASB's HRNet is just a deliberately tiny heatmap net; the PRD's 200 MB estimate was simply wrong. Confirmed the gdown id maps to the *basketball* WASB row of MODEL_ZOO, not another sport.
  - `poseconv3d_k400.pth` (13 MB) — MMAction2 `slowonly_r50_8xb32-u48-240e_k400-keypoint`

---

## F4 — Local mirror `_MIRROR_FILES` is empty

- **Status:** OPEN (chicken-and-egg with F2)
- **Phase raised:** A
- **Description:** `scripts/sync_mirror.py` has an empty `_MIRROR_FILES` list. It runs but is a no-op until populated with `(dataset, relpath)` tuples pointing at small test clips on the `courtvision-data` Volume.
- **Why it matters:** Phase B/C/D need local mirror clips for dataloader iteration and smoke tests on the laptop. Without a local mirror, every iteration goes through `modal run`, which is slow.
- **Resolution / next step:** Populate `_MIRROR_FILES` after F2 is resolved and the Volume has clips. Aim for ~5 short clips per dataset (~5-10 GB total).

---

## F5 — TrackID3x3 wrist-keypoint availability unverified

- **Status:** RESOLVED (2026-05-26) — wrists are at indices 5 and 6 of the 10-keypoint schema. Wrist-PCK eval on TrackID3x3 is viable.
- **Phase raised:** A (documented as a pre-existing PRD risk)
- **Description:** PRD §9.2 makes wrist PCK@0.05 on TrackID3x3 a primary Phase C acceptance gate. PRD §13 risk #5 flags that TrackID3x3 may only label a subset of keypoints that doesn't include wrists.
- **Why it matters:** If wrists aren't labeled, the Phase C PCK eval has to fall back to MultiSubjects-only, and the wrist-PCK ≥ 0.85 gate moves to a different dataset.
- **Resolution:** Rendered the 10 keypoints from `ground_truth/Indoor/pose/basket_S4T4_post.json` onto `videos/Indoor/raw/basket_S4T4_post.mp4` frame 221 — picked because it has the largest spatial spread (Red_Jersey mid-dribble, right hand extended on the ball). Keypoint 5 sits at the extended right hand (far from the body cluster), unambiguously identifying it as the right wrist. Inferred schema by anatomical position:

  | idx | landmark |
  |---:|---|
  | 0 | head/nose |
  | 1 | R-shoulder |
  | 2 | L-shoulder |
  | 3 | R-elbow |
  | 4 | L-elbow |
  | **5** | **R-wrist** |
  | **6** | **L-wrist** |
  | 7 | hip/pelvis center |
  | 8 | R-ankle |
  | 9 | L-ankle |

  Phase C wrist-PCK code should index `[5, 6]` from each player's keypoint array for the wrist landmarks. Note: the pose annotations are *per-frame, jersey-color-keyed*, not track-ID-keyed — Stage 2 inference will need to match its detections to jersey-color GT (via spatial proximity, since color isn't in the inference output). README: "10 pose keypoints of each player **for some frames only**" — confirmed (not every frame has annotations).

---

## F6 — Initial Modal image build will be slow

- **Status:** OPEN (one-time cost, not a defect)
- **Phase raised:** A
- **Description:** Because the openmm stack and the two git-installed packages (WASB, Pose2Sim) live inside the Modal image (per F1), the first `modal deploy` will trigger a fresh pip resolve + install of ~12 packages including mmcv. Likely 5-15 minutes.
- **Why it matters:** Sets expectations for the user-test step. Subsequent deploys are cached and fast unless `pyproject.toml` or the explicit `pip_install` lines change.
- **Resolution / next step:** No action needed; documented for awareness.

---

## F7 — `decisions.md` symlink vs. copy

- **Status:** RESOLVED (copy in `docs/`)
- **Phase raised:** A
- **Description:** PRD §6 says "decisions.md — Symlink or copy from Milestone 2 folder." Chose copy (`docs/decisions.md`) rather than symlink because the repo lives in the parent project folder and a symlink wouldn't work post-clone.
- **Resolution:** Done. If the source `decisions.md` is updated, re-copy manually.

---

## F8 — Named Modal `wandb-secret` replaced with local-env passthrough

- **Status:** RESOLVED (mid-Phase-A user-test step)
- **Phase raised:** A (caught during user test of `modal run scripts/download_datasets.py::download_trackid3x3`)
- **Description:** Originally `modal_app.py` referenced `modal.Secret.from_name("wandb-secret", required_keys=[])` at module level and attached it to the two training function decorators. Modal validates *all* secrets in an app at submit time, even for `modal run`s that target an unrelated function — so the download functions failed with `Secret 'wandb-secret' not found` until the user manually created the named secret.
- **Why it matters:** Phase A `modal run` commands should work end-to-end without a separate `modal secret create` step. The original README had that step, but it's brittle (silent skip if `.env` doesn't have the key) and adds friction.
- **Resolution:** Switched to `modal.Secret.from_local_environ(["WANDB_API_KEY"])`, attached conditionally only when `WANDB_API_KEY` is in the local shell environment at submit time. No persistent Modal secret needed. Behavior:
  - Local shell has `WANDB_API_KEY` → secret passes through to Modal function; W&B logging works.
  - Local shell doesn't have it → empty secrets list; `courtvision.logging.init_wandb` falls back to `WANDB_MODE=disabled`.
  - README updated: replaced "Modal auth + secret" step with just "Modal auth"; added a note about `set -a; source .env; set +a` before training runs.
- **Commit:** `0440048`

---

## F9 — WASB + Pose2Sim removed from shared Modal image

- **Status:** WASB half **RESOLVED (2026-05-27, Phase D)**; Pose2Sim half still OPEN (deferred to Stage 4).
- **Phase raised:** A (caught during user test of `download_multisubjects`)
- **Description:** The shared image previously did `pip_install("git+https://github.com/nttcom/WASB-SBDT.git", "git+https://github.com/perfanalytics/pose2sim.git")`. WASB-SBDT has neither `setup.py` nor `pyproject.toml` — it's research code meant to be cloned and used in-tree. The pip install failed, breaking the image build for every function on the app (including downloads that don't need WASB).
- **Why it matters:** Phase A `modal run` calls must succeed end-to-end without depending on libraries that are weeks away.
- **Resolution / next step:** Removed both git-installs from the shared image. Phase D (Stage 3) and Stage 4 declare per-function images:
  - **WASB (Stage 3): DONE.** `modal_app.wasb_image` = `debian_slim(py3.11).apt_install(ffmpeg,libgl1,libglib2.0-0,git).pip_install(...).run_commands("git clone --depth 1 …/WASB-SBDT.git /opt/wasb").add_local_python_source("courtvision")`. `run_wasb` / `verify_wasb_checkpoint` use it. Two deliberate version choices vs WASB's Dockerfile: `numpy<2` (their online tracker uses `np.Inf`, gone in NumPy 2.0) and torch 2.1.2 (no CUDA-matching 1.11 wheels on modern Modal; HRNet is plain nn modules, and torch<2.6 keeps `torch.load` `weights_only=False` so the `.pth.tar` loads). Deps needed beyond torch: opencv, omegaconf, hydra-core, tqdm, pandas, pillow, matplotlib, pyyaml (the `dataloaders`/`detectors`/`utils` import chain transitively needs pandas+matplotlib even for inference), plus pydantic+rich for `courtvision` schemas/logging. No sklearn/skimage/timm/einops (grep-confirmed unused on the inference path). WASB imported off `sys.path` (`/opt/wasb/src`) rather than `PYTHONPATH` env — it uses bare top-level imports like `from models import …`. First build ~80 s (cached after).
  - **Pose2Sim (Stage 4):** `pose2sim` is on PyPI; use `.pip_install("pose2sim")` in a per-function image. Still TODO.

---

## F10 — Modal caps function timeout at 24 h; PRD training estimates exceed this

- **Status:** OPEN (will revisit in Phase E3 / E4 if needed)
- **Phase raised:** A (caught during user test of `download_multisubjects`)
- **Description:** Modal rejected `train_stage2_multisubjects` (`timeout=3600*36` = 36 h) and `train_stage3_finesports` (`timeout=3600*48` = 48 h) with `InvalidError('Timeout must be between 10s and 86400s (inclusive).')`. Timeout validation happens at app-submit time, so this blocked every `modal run`, including downloads.
- **Why it matters:** PRD §9.5.2 estimates MultiSubjects fine-tune ≈ 12-24 h on A10G and FineSports fine-tune ≈ 24-48 h on A100-40GB. The A10G run fits inside a 24 h timeout; FineSports on A100-40GB may not.
- **Resolution / next step:** Both training timeouts capped at `_MODAL_MAX_TIMEOUT_S = 3600 * 24` (Modal's hard limit). Phase E4 will need to implement checkpoint-resume on FineSports so a multi-day run becomes a sequence of ≤24 h invocations — that's the right pattern for long jobs on Modal regardless of the cap.

---

## F11 — WASB basketball subset upstream link is 404

- **Status:** DEFERRED (Phase D)
- **Phase raised:** A (during dataset-download user-test step)
- **Description:** `nttcom/WASB-SBDT/GET_STARTED.md` points at `ruiyan1995.github.io/SAM.html` for the basketball-subset dataset; that URL returns HTTP 404. No Zenodo, HF, or alternate mirror found via public search.
- **Why it matters:** PRD §9.3 second acceptance criterion ("release-window recall ≥ 90% on ~30 release moments") depends on this dataset. Not needed for inference (the Stage 3 WASB *model* checkpoint is separate, tracked in F3) or for training. Dropping this eval gate trims one Phase 1 acceptance criterion.
- **Resolution / next step:** `download_wasb_basketball` is now a soft-defer (print + return, not raise) pointing here. Revisit when Phase D / Stage 3 eval needs to run. Options: (a) Wayback Machine + GitHub-issue search on `nttcom/WASB-SBDT` for a mirror; (b) email the nttcom authors; (c) substitute SportsMOT or another NBA broadcast dataset; (d) hand-label ~30 release moments from YouTube clips. Decision deferred to Phase D.

---

## F12 — "U Arizona Roboflow basketball" dataset could not be verified; substituted

- **Status:** RESOLVED (by substitution)
- **Phase raised:** A (during dataset-download user-test step)
- **Description:** PRD §8.3 and `decisions.md` Decision 3 reference a "U Arizona Roboflow basketball" dataset (~9,808 images; classes `ball, basketball, people, person, made, rim, shoot`) for the Stage 3 ball-detection mAP eval. Aggressive WebSearch + WebFetch (Roboflow Universe browse/search pages return 403; project pages not indexed) failed to surface a working URL. Search engines summarize the dataset's existence (likely from older training data) but never produce a clickable link — strong signal the URL is unverified.
- **Why it matters:** Stage 3 needs a downloadable basketball ball-detection dataset for the mAP@0.5 / mAP@0.5:0.95 acceptance gate (Decision 3).
- **Resolution:** Substituted two datasets per user direction (download both; use DeepSportradar as the headline number):
  - **Primary (headline mAP):** DeepSportradar Basketball Instants — `deepsportradar/basketball-instants-dataset` on Kaggle. Academic benchmark (ACM MMSports 2022, https://arxiv.org/abs/2208.08190), 1,456 images / 338 ball annotations, CC-BY-NC-SA 4.0, professional LNB-Pro-A broadcast video at 2-5 MP. mAP numbers are literature-comparable — what Decision 3 actually cares about.
  - **Secondary (volume + diversity):** OwnProjects `basketball-w2xcw` on Roboflow Universe. Closest class-signature match to the original PRD intent (`ball, made, person, rim, shoot`), ~2-5K images (page 403s to WebFetch; agent estimate), public domain. Reported alongside DeepSportradar to demonstrate cross-dataset robustness.
- **Action:** `decisions.md` Decision 3 should be updated in a follow-up commit to reference these two datasets by name instead of "U Arizona Roboflow basketball." Out of scope for this flag (decisions.md is a sourced Milestone-2 document; Lukas to decide whether to amend or add a footnote).

---

## F13 — `lapx` dependency added for ByteTrack

- **Status:** RESOLVED
- **Phase raised:** B
- **Description:** Ultralytics' ByteTrack imports `lap` (linear-assignment solver) at runtime. It's not a hard dependency of the `ultralytics` wheel, and ultralytics' built-in auto-install fails inside `uv`-managed venvs because they ship no `pip`. Added `lapx>=0.5.11` (maintained fork with prebuilt macOS arm64 + Linux wheels; exposes the same `lap` module) to `pyproject.toml`.
- **Why it matters:** Without it, `model.track(...)` raises `ModuleNotFoundError: No module named 'lap'`. New top-level dep — flagged per PRD §15.6; user approved.
- **Resolution:** Added + locked in `uv.lock`. Commit `a9f8a43`.

---

## F14 — numpy 2.0 / motmetrics + Modal Volume consistency fixes (Phase B)

- **Status:** RESOLVED (informational)
- **Phase raised:** B
- **Description:** Three small Phase-B runtime issues surfaced during the Stage 1 sweep:
  1. **`np.asfarray` removed in NumPy 2.0** — motmetrics 1.4.0 calls it; Modal's image pulls numpy 2.x via the OpenMMLab stack despite our `<2` pin. Shimmed `np.asfarray` in `stage1_detection/eval.py` before importing motmetrics.
  2. **Modal Volume eventual consistency** — `eval_sweep_run` / `make_stage1_overlay` start fresh containers that snapshot the volume at mount; tracks just committed by a sibling function aren't visible without an explicit `.reload()`. Added reloads.
  3. **`modal run` local-client disconnect** — orchestrating many sequential `.starmap()`/`.remote()` calls from a background `modal run` subprocess crashed mid-sweep (`ConflictError: app is stopped`). Moved the coordinate-descent loop into a server-side `@app.function` (`orchestrate_dev_sweep` / `orchestrate_test_run`); local entrypoint blocks on one `.remote()`.
- **Why it matters:** All three are general Modal+ML gotchas that will recur in Phases C/D/E (any cross-function volume read; any long orchestration). Documented so future stages reuse the patterns (`.reload()` on read; server-side orchestration for long multi-call jobs; shim deprecated numpy APIs the mm-stack drags in).
- **Resolution:** Commits `60327d6` (asfarray), `6858444` (reload), `bca8f35` (server-side orchestrator).

---

## F15 — Stage 1 track-persistence backlog (deferred tuning layers)

- **Status:** PARTIALLY RESOLVED — A1/B1/C1 landed, gate passed, Phase B closed. A2/A3/B2/C2/D1/D2 remain DEFERRED (backlog — revisit if downstream needs it).
- **Phase raised:** B (after qualitative review of tuned overlays — IDF1 ≈ 0.71 test, but visible ID-switching)
- **Description:** Coordinate-descent ByteTrack tuning lifted dev IDF1 0.5564 → 0.6286 and passed the test gate (0.7090). Knob winners: `fuse_score=false`, `new_track_thresh=0.8`. To push track *persistence* further, several layers were brainstormed. The user selected **A1 (player-count cap)**, **B1 (post-hoc stitching)**, and **C1-simplified (distance gate)** for immediate work, and is evaluating **B2 (Re-ID/BoT-SORT)**. The following are explicitly DEFERRED for later:
  - **A2 — court-region mask.** Reject detections whose foot-point falls outside the court polygon (from TrackID3x3 `court_keypoints/`). User: "try A2 after everything else... we'll do it if we need it." Contingent on whether A1+B1+C1 leave spurious off-court detections.
  - **A3 — team-colour clustering.** Cluster bbox-interior colour into 2 jersey groups; constrain a track's team membership from flipping. Helps cross-team confusion; weak within-team. Deferred.
  - **C2 — cross-camera identity reconciliation.** Use multi-view triangulation (Stage 4) to assign canonical IDs across the 3 cameras, outvoting single-camera ID swaps. *Highest leverage for the final pipeline / contest metric* but belongs to Stage 4 + own footage. Deferred to weeks 3-4.
  - **D1 — jersey-number OCR.** TrackID3x3's own baseline uses this (`jersey-number-pipeline`) to disambiguate teammates. ~3 days; SOTA for this dataset. Deferred.
  - **D2 — train a player-specific Re-ID model** on own footage. 1-2 weeks. Out of scope.
- **Why it matters:** Track stability *during shot-release windows* drives Stage 5 (classifier window integrity) and Stage 7 (closest-defender wrist-to-trajectory — the headline metric). Off-ball stability matters less. The downstream-impact analysis (this conversation) concluded the real fix for the contest metric is C2 cross-camera, with A1/B1/C1 as cheap single-camera robustness now.
- **Resolution / next step:** **DONE (2026-05-27).** A1 (`filter_non_players`), B1 (`stitch_tracks`), and C1-simplified (distance gate) implemented in `stage1_detection/postprocess.py`, calibrated, and baked into `yolov8m_bytetrack.yaml`. Re-eval: dev IDF1 0.760 ± 0.146, test 0.785 ± 0.083 (95% CI lower bound 0.750 > 0.70 gate). Phase B closed with detection/tracking deemed good-enough by the user. **B2 (BoT-SORT Re-ID) deferred** — not pursued; calibrated ByteTrack + post-processing clears the gate, and B2 is a PRD §5.2 deviation we chose not to take. The remaining deferred options (A2 court mask, A3 team-colour, B2 BoT-SORT, C2 cross-camera, D1 jersey OCR, D2 custom Re-ID) are mirrored in auto-memory `project_stage1_deferred_backlog.md` for cross-session recall. Revisit if Stage 5 window integrity or the Stage 7 contest metric needs more track stability — C2 (cross-camera, weeks 3-4) is the highest-leverage next lever.

---

## F16 — Off-the-shelf WASB-basketball has partial, clip-dependent ball coverage on TrackID3x3

- **Status:** OPEN / informational (flag for the week 3-4 Phase-2 ★ decision, per PRD §10 Phase D — "note it but do NOT pivot in this sprint").
- **Phase raised:** D (during the Stage 3 qualitative validation runs).
- **Description:** Ran the off-the-shelf WASB-basketball checkpoint on 3 Indoor TrackID3x3 clips at the default `conf_threshold=0.5`. Fraction of frames with *any* ball detection: `basket_S1T1_pre` 36/168 (21%), `basket_S4T6_post` 18/224 (8%), `basket_S4T4_post` 5/236 (2%). Where it does fire, the track is spatially coherent and smooth (e.g. S1T1 frames 12-125, ball drifting right ~6-10 px/frame, all in-bounds) — i.e. detections are *correct*, just *sparse*. Lowering the threshold 0.5→0.3 on the worst clip (S4T4) changed detections only 5→6, so the gap is **not** a threshold artifact — the model produces no heatmap peak for the ball in most frames of those clips (occlusion in post-shot scrambles + domain gap: WASB-basketball was trained on wide-angle NBA broadcast/SAM, TrackID3x3 is a closer 3x3 half-court rig).
- **Why it matters:** Stage 7's headline metric (3D defender-wrist distance to ball trajectory) needs the ball tracked *through the release window*. Sparse 2D coverage per camera may still triangulate acceptably across 3 synchronized views (a frame the ball is missed in one camera may be caught in another), but persistent gaps would hurt. This is the trigger condition for PRD Decision 3's Phase-2 ★ fallback (custom WASB fine-tuning).
- **Important caveat:** These percentages are **frames-with-a-detection, NOT recall** — TrackID3x3 has no ball ground truth (its annotations are player MOT boxes + player pose keypoints, not the ball). The PRD's actual Stage 3 quantitative gate is **mAP@0.5 ≥ 0.85 on DeepSportradar Basketball Instants** (ball GT, literature-comparable), which has not been run yet. Don't read these density numbers as accuracy.
- **Resolution / next step:** (a) Run the DeepSportradar mAP eval (PRD §9.3, `stage3_eval.py` — still a Phase-A stub) for the real quantitative number before judging the checkpoint. (b) For the contest metric specifically, evaluate ball coverage *after multi-view triangulation* (Stage 4, own footage, weeks 3-4) rather than per-camera. (c) If coverage is still inadequate then, trigger Phase-2 ★ custom WASB training. Decision deferred to weeks 3-4 per PRD.

---

## F17 — Stage 2 (RTMPose) inference is Modal-only; PRD "local dry run" check substituted

- **Status:** OPEN (intentional; PRD-aligned, follows from F1)
- **Phase raised:** C
- **Description:** PRD §10 Phase C lists an automated acceptance check: *"Local dry run on a single mirror clip + its Phase B tracks produces a schema-valid `Pose2DSequence`."* But Stage 2 inference needs the OpenMMLab stack (`mmpose`/`mmcv`), which is Modal-only and not installed on the laptop (F1 — mmcv has no macOS arm64 wheel). So the actual RTMPose dry run cannot run locally, unlike Stage 1 (ultralytics installs locally).
- **Why it matters:** The Phase C local automated-check set therefore *cannot* include a real inference dry run. If a reviewer expects the PRD's literal local check, they'll be confused when it's absent.
- **Resolution:** Local automated checks for Phase C are: `pytest tests/test_stage2_smoke.py` (pure-Python eval + Halpe→COCO conversion + GT parser, validated against the real example GT), `scripts/stage2_run.py --help`, `ruff`, `mypy` — all green. The schema-valid `Pose2DSequence` dry run is **substituted by a Modal T4 run** in the Phase C user-test step (`modal run scripts/stage2_run.py::run`). `rtmpose_inference.py` is written so the module imports cleanly without `mmpose` (the import is lazy inside `run_stage2`), so the laptop-side tooling stays green. Same pattern will apply to Stage 5 (Phase E).

---

## F18 — TrackID3x3 pose GT: structure, provenance, and the side-agnostic wrist-PCK decision

- **Status:** RESOLVED (2026-05-28) — parser + eval implemented and tested against the real example GT.
- **Phase raised:** C
- **Description:** Inspected the real TrackID3x3 Indoor pose GT (`examples/trackid3x3/ground_truth/Indoor/pose/basket_S4T4_post.json`, S4T6) to pin the eval. Findings:
  - **Structure:** `{"video": {file_name, resolution:[1280,720], frame_rate, total_frames}, "annotations": {"frame_<i>": {"<Colour>_Jersey": [[x, y], ...×10]}}}`. Keypoints are `[x, y]` only — **no per-keypoint score/visibility**. Frame keys are 0-based (match our predictions directly — unlike the 1-based MOT bbox GT used in Stage 1).
  - **Density:** for these Indoor clips, *not* sparse — all 6 players, every frame, 10 keypoints each (S4T4: 235 frames × 6, S4T6: 223 × 6). The README's "for some frames only" caveat may apply to other subsets; the Indoor eval clips are dense.
  - **Schema (confirms F5) by kinematic check:** R-wrist(5) sits nearer R-elbow(3) than L-elbow(4) in 86–90% of skeletons; L-wrist(6) nearer L-elbow(4) ~80%. The high-but-not-100% rate (hands meeting at the body) means **L/R is ambiguous ~10–20% of the time** even kinematically.
  - **Provenance:** coordinates carry heavy subpixel float precision (e.g. `1019.4202499389648`) → these are the dataset's **baseline pose-estimator output, not hand-labelled truth**. So wrist PCK on TrackID3x3 is *in-domain agreement with their baseline*, not accuracy vs ground truth.
- **Why it matters:** (1) The keying is **jersey-colour, not track-id** → the eval spatially matches predicted players to GT players per frame (Hungarian on keypoint centroids, capped at half the court spread). (2) The L/R ambiguity makes a same-side wrist PCK fragile, so the headline metric uses a **side-agnostic 2×2 min-cost wrist assignment** — it scores wrist *localisation* (what the Stage 7 contest metric needs) independent of the L/R label; a strict same-side number (`wrist_pck_sided`) is reported alongside. (3) Because GT is model-derived, the 0.85 gate is an agreement bar; qualitative overlay review (Phase C user-test step 5/6) matters more than usual.
- **Resolution / next step:** Implemented in `stage2_pose/eval.py` (`_load_trackid3x3_pose_gt`, `_score_trackid3x3_clip`, side-agnostic `_wrist_distances`), PCK normalised by the GT-skeleton bbox diagonal (GT carries no person bbox). Verified end-to-end by `test_evaluate_trackid3x3_on_real_gt` (perfect synthetic prediction from the real GT → wrist_pck ≈ 1.0). MultiSubjects secondary eval (`evaluate_multisubjects`) assumes BlazePose CSV coords are MediaPipe-normalised [0,1] — **unconfirmed** because all three local sample CSVs are empty (headers only); confirm against a populated CSV on the data Volume on the first Modal MS-eval run.

---

## F19 — Ball detector: adding a YOLOv8 appearance detector alongside WASB (PRD §5.2 deviation)

- **Status:** **DEFERRED (2026-05-28).** After qualitative review of the trained-YOLO overlays, off-the-shelf *and* trained ball detection are **not good enough on TrackID3x3** (frequent false positives on players' heads — see Outcome). Ball detection is **parked for the sprint**; the pipeline pivots to **pose-first** (shot classification from pose, optional pose-derived contest metric). Revisit the ball on own iPhone footage (closer cameras → larger ball) + multi-view fusion. All ball code/checkpoints kept in place for that revisit.
- **Phase raised:** D (during Stage 3 qualitative validation).
- **Description:** PRD §5.2 locks **WASB** as the ball detector. We are deviating to an **appearance-based YOLOv8 ball detector** for Stage 3.
- **Why (diagnosed, not guessed):** WASB ran end-to-end correctly (harness validated on `basket_S1T1_pre`: median heatmap peak 0.126, p90 0.74, clean smooth track), but coverage is *clip-dependent and often near-zero*. Heatmap diagnostic (`modal_app.diagnose_wasb` / `wasb_inference.diagnose_stage3`) on `basket_S4T4_post`: median peak **0.030**, p90 **0.059** — i.e. the heatmap sits at the noise floor (~0.012) across ~90% of frames, so the model is **blind to the ball there regardless of threshold** (lowering 0.5→0.3→0.05 changed detections 5→6→30, all near-noise argmaxes). Root cause is structural, threefold: (1) **WASB is motion-based** — 3-frame input tuned for fast balls (tennis/badminton/volleyball); a held/dribbled/post-shot basketball barely moves frame-to-frame → flat heatmap; (2) **tiny ball** — TrackID3x3 frames the whole half-court, ball ≈6 px after WASB's mandatory 512×288 downscale; (3) **appearance/domain gap** — WASB-basketball trained on wide-angle NBA broadcast. None of these are fixable by knob-tuning. An appearance detector fires on a *visible* ball whether or not it moves, which is the actual requirement ("detect the ball in most frames, even if not perfectly").
- **Plan:** (a) reuse Stage 1's YOLO stack; first sanity-check **stock yolov8m COCO "sports ball" (class 32)** at imgsz=1280 on TrackID3x3 (zero training); (b) train a single-class **`ball`** YOLOv8 on the Roboflow `basketball-w2xcw` data already on the Volume (`category_id 1`; train/valid/test COCO splits) → `/artifacts/checkpoints/yolo_ball.pt`; (c) wrap as `stage3_ball/yolo_ball.py::run_stage3_yolo` emitting the **same `Ball2DSequence`** schema so overlays/triangulation are unchanged; add `modal_app.run_yolo_ball`. WASB code is kept (not deleted) — it may still help as a trajectory smoother on fast-ball frames, and the contest metric ultimately runs on own footage (weeks 3-4) where camera distance is controllable.
- **Outcome (2026-05-28) — full postmortem of the ball-detection effort:**

  *Everything tried, in order, on the two Indoor clips (`basket_S4T4_post`, 236 frames, ball mostly held/post-shot; `basket_S1T1_pre`, 168 frames, open play). All counts are "frames with **a** detection," NOT validated recall — TrackID3x3 has **no ball ground truth**.*

  | Detector / config | S4T4 | S1T1 |
  |---|---|---|
  | WASB (motion-based, off-the-shelf) | 5/236 (2%) | 36/168 (21%) |
  | stock COCO yolov8m "sports ball" (cls 32), imgsz 1280 | 1/236 | — |
  | stock COCO, imgsz 1920, conf 0.1 | 72/236 (31%) | — |
  | stock COCO, imgsz 1280, 3×2 tiles | 56/236 | — |
  | trained YOLO (Roboflow `ball`), 3×2 tiles, conf 0.25 | 57/236 (24%) | 28/168 (17%) |
  | trained YOLO, 3×2 tiles, **conf 0.05** + motion gate (max_disp 150) | 135/236 (57%) | 109/168 (65%) |

  **The 57%/65% numbers are misleading and were a reporting error.** They count any per-frame detection; with no ball GT, precision was never measured. On full-video review the **majority of detections at conf 0.05 are false positives — the detector fires on players' heads** (round, ~ball-sized, comparable colour at this scale). So lowering the confidence threshold bought "recall" almost entirely by admitting head FPs. **Lesson:** never report detection % as success without a precision/GT check; sample many overlay frames (or build a precision proxy), not one. (One spot-checked frame looked correct, which masked the systemic FP problem the user caught by watching the whole clip.)

  **Root cause:** the ball is a **~10–15 px, low-contrast object in a wide elevated shot**, i.e. small-object detection at the limit, and **player heads are a strong false-positive attractor** at that scale. Compounding factors: (a) no in-domain ball labels to train/tune precision (Roboflow balls are larger/closer → poor scale transfer); (b) WASB is motion-based → blind on a held/slow ball; (c) tiling/high-res raise recall but also multiply head FPs.

  **Decision (user, 2026-05-28):** stop ball work for the sprint; **pivot to pose-first** — Stage 2 RTMPose → multi-view 3D pose → Stage 5 PoseConv3D shot classification, with a possible **pose-only contest metric** (closest-defender hand distance to the shooter's release hand/elbow) that needs no ball. This means the PRD's headline ball-trajectory contest metric (Stage 7) becomes **stretch / future-work**, gated on ball detection becoming viable.

  **Conditions to revisit the ball (likely on own iPhone footage, weeks 3-4):** (1) closer cameras → larger ball; (2) **multi-view fusion** — a frame where one camera loses the ball, another often sees it, and parabolic physics interpolates gaps; (3) **hand-label a few hundred in-domain ball frames** and fine-tune (the single highest-leverage fix); (4) a real **precision metric** (label ball boxes on a held-out set) before trusting any number; (5) consider a dedicated small-object pipeline (SAHI) + head-FP hard-negative mining.

  **Artifacts kept (NOT deleted) for the revisit:** `stage3_ball/wasb_inference.py` (+ `diagnose_stage3`), `stage3_ball/yolo_ball.py` (tiling + motion gate), `modal_app.{run_wasb,run_yolo_ball,diagnose_wasb,verify_wasb_checkpoint,wasb_image}`, `scripts/train_yolo_ball.py` (resumable, best-only checkpointing), checkpoints `/artifacts/checkpoints/{wasb_basketball.pth,yolo_ball.pt}`. All lint/type-clean. Overlays for review: `outputs/stage3_overlays/basket_{S4T4_post,S1T1_pre}-yball-c05.mp4`.

  **GPU cleanup TODO:** a detached training app (`ap-4fGnEOKJYkMPTcc7Pb41ud`, old code) may still be running an A10G — stop with `modal app stop <id>` to avoid spend.

---

## F20 — mmpose unusable on torch 2.12 / CUDA 13 → Stage 2 pivoted to rtmlib (ONNX)

- **Status:** RESOLVED (2026-05-28) by pivoting Stage 2 inference off MMPose onto **rtmlib** (ONNX) — see `decisions.md` Decision 2 "Phase C engine swap". *(Renumbered from a transient F19 that collided with the Phase D ball-detector flag.)*
- **Root cause (researched):** the OpenMMLab stack is effectively **unmaintained against modern PyTorch**. The Modal image runs **torch 2.12 + CUDA 13.0**; mmcv's last release (2.2.0, Apr 2024) ships prebuilt wheels only up to torch ~2.1, and an *open, unanswered* mmcv bug (#3310) reports `mmcv._ext` failing to compile on torch 2.9 + CUDA 13. So the whole error chain below was one disease (too-new env + unmaintained toolchain), and even forcing imports through would likely leave `mmcv._ext` CUDA ops broken at runtime — i.e. the remaining `mmcv==2.1.0` attempt was **predetermined not to reliably work** (confirmed by research before spending the rebuild).
- **Symptom chain (all now moot):** (1) numpy-2 ABI vs xtcocotools wheel ("dtype size 96 vs 88"); (2) xtcocotools 1.13 sdist broken (missing `maskApi.c`), 1.14 wheel numpy-2-only; (3) mmpose hard-imports `mmdet` (RTMO head); (4) mmdet 3.x caps `mmcv<2.2.0` while only mmcv 2.2.0 builds here; (5) mmcv source build fails on modern setuptools dropping `pkg_resources` (the F1 issue); (6) non-deterministic mmdet resolution across builds.
- **Resolution:** rtmlib depends only on `numpy`/`opencv`/`onnxruntime` (no torch/mmcv/mmpose/mmdet), so it is fully decoupled from the torch/CUDA/numpy web. Same model + weights (RTMPose-l Halpe-26, ONNX export of the `2abb7558` checkpoint, URL confirmed HTTP 200). Implemented in `stage2_pose/rtmpose_inference.py` on a dedicated lightweight `rtmpose_image`; the shared-image mm-stack experiments were reverted.
- **Carryover for Phase E:** PoseConv3D (mmaction2) will hit the same torch-2.12 wall. Plan: isolate it in a dedicated **pinned-torch image** (the pattern Phase D used for `wasb_image`), not the shared image. Do NOT re-attempt mmpose/mmaction2 on the bleeding-edge shared image.
- **Phase raised:** C (first attempt to `import mmpose` on the Stage 2 inference path).

---

## F21 — Phase C (Stage 2 pose) gate disposition: accepted via documented deviation

- **Status:** RESOLVED (2026-05-28) — Phase C closed. Full rationale in `decisions.md` Decision 2 "Phase C result + gate disposition".
- **Phase raised:** C.
- **Summary:** Wrist PCK@0.05 across all **13 TrackID3x3 Indoor pose-GT clips** = **0.655** (mean 0.648 ± 0.043), below the 0.85 gate — accepted as a **documented acceptable deviation** (PRD §9.2 permits qualitative acceptance). The per-landmark diagnostic shows the eval is sound (shoulder **0.94** — a broken eval couldn't score any joint that high) and a textbook proximal→distal falloff (shoulder 0.94 → elbow 0.73 → wrist 0.66); the GT is model-derived (F18) at a strict ~12 px tolerance, so 0.655 understates true accuracy. User reviewed 13 wrist-overlay videos and confirmed wrist location looks accurate, **especially on shot-release frames** (the Stage 7 contest-metric-critical frames).
- **Caveat:** per-landmark `head=0.675` is likely a GT-head-point vs Halpe-nose mapping offset, not a real localisation miss.
- **Escalation (if needed):** the Phase 2 ★ RTMPose fine-tune remains the documented fallback if wrist accuracy proves insufficient on own footage downstream.

---

## F22 — Stage 5 overlay: `classify_windows` mutated the caller's keypoint array in place

- **Status:** RESOLVED (2026-05-28). One-line fix in `stage5_classifier/infer.py`: `.copy()` the per-window keypoint/score slices before the posec3d pipeline.
- **Symptom:** qualitative overlay videos drew the COCO-17 skeleton **off the player** (shrunk + shifted to the top-right), while the underlying keypoints were verified correct (a standalone debug render placed them perfectly on the player). Same code, same byte-identical video, same val.pkl — yet the *deployed* overlay was wrong and a local render with dummy probs was right.
- **Root cause:** in `classify_windows`, the per-window slices `keypoint[s:e]` / `keypoint_score[s:e]` are **numpy views**, and `results["keypoint"] = kp[None]` shares that buffer. The posec3d transforms mutate in place — `PoseCompact` crop-*shifts* keypoints to the new bbox origin and `Resize(56,56)` *scales* them — so after inference the caller's array was corrupted. `render_overlay_clips` runs inference first, then hands the (now mutated) array to `make_shot_overlay_video`. The local debug skipped inference, so its array stayed pristine → masked the bug. Diagnosis hinged on noticing the only path difference was the preceding `classify_windows` call.
- **Secondary effect:** the in-place mutation also corrupted inference itself — overlapping windows re-cropped already-shifted keypoints (double-shift). After the fix the dribble clip's false-event count dropped 7→5 (still nonzero — that is the separate trimmed-classifier→untrimmed-detector gap, not this bug).
- **Lesson:** mmaction2/mmcv transforms are **in-place by contract**; never feed them a view of an array you intend to reuse. Copy at the window boundary, not just at the top (top-level `np.asarray(..., float32)` is a no-op when dtype already matches, so it does not protect callers).
- **Phase raised:** E5 (qualitative overlay rendering).

---

## F23 — Stage 6 release detection: 2D pose-only (no ball, no 3D); PRD method enum extended

- **Status:** ACTIVE (2026-05-29). Documented deviation from PRD §7.6 / §9.6.
- **Deviation:** the PRD's Stage 6 release rules (`forward_rule`, `backward_trace`, `fallback_apex`) are all **ball-based** and assume 3D triangulated pose (Stage 4). We have neither — ball detection was abandoned as infeasible (F19) and **the user has decided not to pursue a ball detector** for this project; Stage 4 (3D) is iPhone-gated. So Stage 6 is built as a **single-camera 2D pose-only** detector: `pose_apex` (PRIMARY — shooting-wrist peak-height/apex, the velocity zero-crossing) and `pose_forward` (SECONDARY — onset of overhead arm extension: wrist-above-shoulder + elbow≥160°). Jump shots only (layups deferred).
- **Schema change:** `ReleaseFrame.method` Literal extended with `"pose_apex"` and `"pose_forward"` (kept the three ball-based values for the future 3D variant). `n_cameras_at_release == 1` marks the 2D single-camera methods. Chosen over reusing `fallback_apex` so provenance stays truthful (user picked "Option B").
- **Migration:** when iPhone footage + Stage 4 exist, the SAME apex/forward logic runs on triangulated 3D pose (v2) — better vertical + multi-view occlusion robustness. No ball either way, so the ball-based PRD methods may never run; `pose_apex` is the de-facto primary, not a fallback.
- **Validation:** qualitative video review — MultiSubjects close-up shot clips first (clear release), then TrackID3x3 detected shots (realistic). Δframes-vs-GT quantification deferred until the overlays look right.
- **Phase raised:** post-sprint Stage 6 (built after Phase E).

---

## F24 — Stage 6 forward-confidence as a Stage 5 false-positive filter (cross-method verification)

- **Status:** PROPOSED (2026-05-29). Small-sample evidence is promising (n=9 TrackID3x3 events); LOCK-IN pending validation on iPhone footage. **Logging now so the rationale can go into the final report if the pattern holds.**

- **The core idea — same input, two methods, cross-check:**
  - Stage 5 (action recognition) and Stage 6 (release detection) **both operate on the exact same per-player COCO-17 pose track** — there is no other input. They differ only in *method*: Stage 5 is a deep CNN (PoseConv3D / SlowOnly-R50) that learns implicit spatio-temporal pose patterns from heatmap volumes; Stage 6 encodes the *explicit physical geometry* of a jump-shot release (shooting wrist above the shoulder + elbow extended ≥160° for ≥2 frames).
  - When the CNN fires "shot" but the geometric rule cannot find any sustained overhead arm extension in the search window, the two methods *disagree* on what the same pose sequence depicts. That disagreement is a principled FP signal — the pose probably *resembles* a shot to the CNN without actually containing the geometric signature of a release.

- **The signal in code:** `Stage6 ReleaseFrame.confidence_flag == "high"` iff the forward rule found a sustained extension interval (a true overhead-extension). When no interval is found, Stage 6 falls back to the wrist-apex computation and tags `confidence_flag = "low"` (and forward_frame == apex_frame as the visible symptom).

- **Small-sample evidence (TrackID3x3, 9 Stage-5-detected events at threshold ≥0.6 — user manually labelled real-vs-FP):**

  | Stage 5 conf | clip / track / peak | Stage 6 forward_conf | user verdict |
  |---|---|---|---|
  | 0.94 | S4T4 t4 f200 | **high** (186 ≠ 213) | real ✓ |
  | 0.92 | S5T7 t3 f112 | **high** (122 ≠ 126) | real ✓ |
  | 0.90 | S5T5 t4 f112 | **high** (123 ≠ 127) | real ✓ |
  | 0.82 | S6T4 t4 f88 | low (65 == 65) | FP |
  | 0.73 | S6T4 t9 f237 | low (241 == 241) | FP |
  | 0.67 | S6T1 t6 f96 | low (142 == 142) | FP |
  | 0.66 | S4T6 t3 f192 | **high** (194 ≠ 198) | real ✓ |
  | 0.65 | S4T6 t6 f176 | high (135 ≠ 214) | FP ✗ MISSED |
  | 0.63 | S4T6 t8 f148 | low (104 == 104) | FP (bg player) |

  Stage 5 alone @ 0.75 threshold: **3 TP / 1 FP / 1 FN (S4T6 0.66 missed)** → 75% precision, 75% recall.
  Stage 5 @ 0.6 + Stage 6 `forward_conf=="high"` gate: **4 TP / 1 FP / 0 FN** → 80% precision, 100% recall.

  Strictly dominates on this sample. The lowered Stage 5 threshold (0.6) recovers a real shot (S4T6 0.66) that 0.75 was dropping; Stage 6 absorbs the FPs that come along, except one (S4T6 t6) where the rule found extension at the wrong time — a separate failure mode the geometric filter cannot catch.

- **What's implemented:**
  - `score_threshold` in `run_shot_inference` lowered from **0.75 → 0.6** (raw shot-event firing).
  - `is_verified_shot(release_frame)` helper added to `stage6_release` — returns `True` iff `release_frame.confidence_flag == "high"`. Pure cross-method gate; no new ML.
  - Combined-pipeline eval entrypoint in `stage5_modal.py` (`eval_actions_combined`) that runs Stage 5 @ 0.6 + Stage 6 verification across a clip set and reports raw vs verified counts.

- **Honest caveats (must repeat in the report):**
  1. **n = 9 events on 13 short clips.** Too few for a reliable precision/recall claim. Re-validate on iPhone footage with hand-labelled shot ground truth.
  2. **The filter cuts both ways with pose quality:** an occluded shooter with weak Stage 2 keypoints will trip a low-confidence Stage 6 → false rejection of a real shot. Net win on this sample, but TrackID3x3 has reasonable pose quality; iPhone may differ.
  3. **It does not catch every FP type.** S4T6 t6 (a non-shooting motion that happened to satisfy the forward condition somewhere in the window) is missed by the rule alone — would need a tighter spatial/temporal gate (e.g. require the extension to be near the Stage 5 peak frame).
  4. The mechanism depends on Stage 5 and Stage 6 being **same-input, different-method**. If/when a ball or 3D cue is added (changing the input set), this cross-method symmetry weakens and the filter needs re-evaluation.

- **Validation plan (re-test on iPhone data):** rerun `eval_actions_combined` with hand-labelled GT shots → compute precision/recall pre- and post-Stage-6 gate → confirm: (a) recall is preserved, (b) precision improves materially, (c) the failure-mode distribution matches (most FPs caught by the fallback, a residual tail of "wrong-time extension" misses).

- **Phase raised:** post-sprint Stage 5 + Stage 6 composition.

---

## F25 — Pose-only Stage 7 contest metric (PRD §9.7 / Decision 9 deviation following Phase D drop)

- **Status:** RESOLVED (2026-06-02) — design + schema landed. Implementation runs in weeks 3-4 with Stage 4 triangulation + own iPhone footage.
- **Phase raised:** post-Phase-C planning, after the user decided to drop Phase D entirely (ball detection — see `project_ball_detection_deferred` memory, `flags.md` F19).
- **Description:** PRD §9.7 / `decisions.md` Decision 9 specced a 3D **defender-wrist-to-ball-trajectory** distance as the headline contest measurement. Without Phase D, no ball trajectory is available, so the metric is redesigned to be **pose-only** while preserving Decision 9's framework (5 measurements at release).
- **Schema changes (`src/courtvision/schemas.py::ContestMeasurement`):**
  - `wrist_to_trajectory_distance_m` → **`wrist_to_wrist_distance_m`** (closest defender's nearer wrist ↔ shooter's shooting wrist, 3D).
  - `wrist_height_advantage_m` *semantics* changed (defender wrist z − shooter wrist z, was vs ball position); name unchanged.
  - `shooter_velocity_{forward,lateral,vertical}_mps` (3 fields) → **`shooter_fade_mps`** (single scalar): `sqrt(max(0, −v_forward)² + v_lateral²)` in the court frame. Forward = toward rim → 0 fade. Vertical dropped (captured by wrist-height advantage).
  - `floor_distance_m`, `composite_score`, `closest_defender_track_id`, `flags` unchanged.
- **`tests/test_schemas.py::test_contest_measurement_sequence`** updated accordingly. **`docs/prd.md` §7.7 / §9.7 intentionally NOT edited** — the PRD remains the source of truth for the original spec; deviations live here + in `decisions.md` (Decision 9 sub-bullet "Phase D-drop pose-only redesign"), matching the Decision 2 / engine-swap precedent.
- **Why it matters:** keeps the project's headline contest measurement on a runnable path without a ball detector. Phase 2 escalation = revisit the original 3D wrist-to-trajectory metric if/when iPhone footage enables a working ball detector.

---

## F26 — MultiSubjects per-keypoint COCO-17 PCK secondary corroboration (closes F18 MS-scale loose end)

- **Status:** RESOLVED (2026-06-02). The Stage 2 pose pipeline was evaluated end-to-end against MultiSubjects BlazePose-33 GT (subset to COCO-17) across n = 300 stratified val clips.
- **Phase raised:** post-Phase-C report-prep (user pushed for the secondary all-COCO-17 corroboration that F18 had flagged as "unconfirmed / not yet run").
- **What changed:**
  - `evaluate_multisubjects` extended with: (a) per-keypoint COCO-17 PCK breakdown (was overall + wrist only), (b) multi-track per-frame spatial matching by COCO-centroid closeness (handles Stage 1 emitting extra non-player tracks), (c) per-clip pixel resolution read from each clip's video file via cv2 (replaces the global `frame_w`/`frame_h` assumption that F18 flagged as unverified — `videos_dir` param). Signature is backward-compatible.
  - New `scripts/stage2_ms_batch.py`: extracts the requested val clips' videos + BlazePose CSVs from the on-Volume RAR archives, runs Stage 1 + Stage 2 in parallel waves, then runs the eval.
- **Result (n = 300, 362,695 keypoints, full per-clip + per-landmark table in `decisions.md` Decision 2):**
  - **Overall COCO-17 PCK@0.05 = 0.841.** Wrist PCK = 0.680.
  - Per-landmark proximal→distal falloff matches TrackID3x3's per-landmark diagnostic; shoulders/face/legs 0.85–0.91 (eval is sound), elbow ~0.78, wrist ~0.68.
  - Cross-dataset wrist convergence: TrackID3x3 0.655 ↔ MultiSubjects 0.680 — same number on completely different domain + camera + GT-provider, confirming the wrist measurement reflects RTMPose's distal-joint accuracy vs model-derived GT, not a TrackID3x3 artifact.
- **F18 loose end closed:** the "MultiSubjects coord scale assumed not yet confirmed" note in F18 is resolved by the per-clip resolution lookup; the eval works end-to-end and the numbers are stable.
- **Eval JSON:** `outputs/stage2_gate/eval_stage2_multisubjects_300.json` (gitignored). Reproducible with `scripts/stage2_ms_batch.py::run --clips <300 stems>`.
