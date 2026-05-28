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

- **Status:** RESOLVED (2026-05-26) — all 4 checkpoints on `/artifacts/checkpoints/`. Sanity-load to verify they're real weights pending Phase C/D/E first-use.
- **Phase raised:** A
- **Description:** `scripts/download_checkpoints.py` has stubbed functions for the four pretrained checkpoints. YOLOv8m is trivial (ultralytics auto-downloads); the other three (RTMPose-l Halpe-26, WASB basketball, PoseConv3D K400) need their model-zoo URLs or HF repo IDs filled in.
- **Why it matters:** Phases B/C/D/E configs reference `/artifacts/checkpoints/<weight>.pt[h]`. If the file isn't on the Volume, the stages can't run.
- **Resolution:** All four downloaded. Actual sizes are smaller than the PRD's estimates for WASB (6 MB vs 200 MB est.) and PoseConv3D K400 (13 MB vs 250 MB est.) — possibly weights-only saves without optimizer state, or compact architecture variants. Not necessarily wrong, but worth a `torch.load` sanity check at first use in Phase C/D/E before assuming they're complete.
  - `yolov8m.pt` (52 MB) — ultralytics auto-fetch (`YOLO('yolov8m.pt')`)
  - `rtmpose_l_halpe26.pth` (113 MB) — MMPose `rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192`
  - `wasb_basketball.pth` (6.1 MB) — `nttcom/WASB-SBDT` MODEL_ZOO Drive link, via gdown single-file
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

- **Status:** OPEN (deferred to Phase D / Stage 4)
- **Phase raised:** A (caught during user test of `download_multisubjects`)
- **Description:** The shared image previously did `pip_install("git+https://github.com/nttcom/WASB-SBDT.git", "git+https://github.com/perfanalytics/pose2sim.git")`. WASB-SBDT has neither `setup.py` nor `pyproject.toml` — it's research code meant to be cloned and used in-tree. The pip install failed, breaking the image build for every function on the app (including downloads that don't need WASB).
- **Why it matters:** Phase A `modal run` calls must succeed end-to-end without depending on libraries that are weeks away.
- **Resolution / next step:** Removed both git-installs from the shared image. Phase D (Stage 3) and Stage 4 will declare per-function images:
  - **WASB (Stage 3):** `Image.debian_slim(...).run_commands("git clone https://github.com/nttcom/WASB-SBDT.git /opt/wasb").env({"PYTHONPATH": "/opt/wasb:$PYTHONPATH"})` plus WASB's own runtime deps from their `environment.yml`.
  - **Pose2Sim (Stage 4):** `pose2sim` is on PyPI; use `.pip_install("pose2sim")` in a per-function image.

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
