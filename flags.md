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

- **Status:** OPEN (needs user input during Phase A user-test step)
- **Phase raised:** A
- **Description:** `scripts/download_datasets.py` has one Modal function per dataset (`download_trackid3x3`, `download_multisubjects`, `download_finesports`, `download_u_arizona`, `download_wasb_basketball`). Each function body is `raise NotImplementedError("fill source URL during Phase A user-test step")` with the canonical source noted in a comment.
- **Why it matters:** Phases B/C/D/E need datasets on the `courtvision-data` Modal Volume. Without URLs filled in, `modal run scripts/download_datasets.py::download_*` won't actually download anything.
- **Resolution / next step:** Lukas fills in URLs as datasets are sourced. Suggested order: TrackID3x3 (smallest, smoke test) → U Arizona → WASB basketball → MultiSubjects → FineSports. **FineSports note (PRD §13 open item #2):** access is gated; if request denied, fall back to Path A (skip transfer-stage-3) and proceed.

---

## F3 — Checkpoint download URLs not yet filled

- **Status:** OPEN (needs user input during Phase A user-test step)
- **Phase raised:** A
- **Description:** `scripts/download_checkpoints.py` has stubbed functions for the four pretrained checkpoints. YOLOv8m is trivial (ultralytics auto-downloads); the other three (RTMPose-l Halpe-26, WASB basketball, PoseConv3D K400) need their model-zoo URLs or HF repo IDs filled in.
- **Why it matters:** Phases B/C/D/E configs reference `/artifacts/checkpoints/<weight>.pt[h]`. If the file isn't on the Volume, the stages can't run.
- **Resolution / next step:** Fill in:
  - `yolov8m.pt` — ultralytics auto-fetch (`YOLO('yolov8m.pt')`)
  - `rtmpose_l_halpe26.pth` — MMPose model zoo
  - `wasb_basketball.pth` — `github.com/nttcom/WASB-SBDT` release page
  - `poseconv3d_k400.pth` — MMAction2 model zoo (`slowonly_r50_8xb16-u48-240e_k400-keypoint`)

---

## F4 — Local mirror `_MIRROR_FILES` is empty

- **Status:** OPEN (chicken-and-egg with F2)
- **Phase raised:** A
- **Description:** `scripts/sync_mirror.py` has an empty `_MIRROR_FILES` list. It runs but is a no-op until populated with `(dataset, relpath)` tuples pointing at small test clips on the `courtvision-data` Volume.
- **Why it matters:** Phase B/C/D need local mirror clips for dataloader iteration and smoke tests on the laptop. Without a local mirror, every iteration goes through `modal run`, which is slow.
- **Resolution / next step:** Populate `_MIRROR_FILES` after F2 is resolved and the Volume has clips. Aim for ~5 short clips per dataset (~5-10 GB total).

---

## F5 — TrackID3x3 wrist-keypoint availability unverified

- **Status:** OPEN (will be checked first thing in Phase C, per PRD §13 risk #5)
- **Phase raised:** A (documented as a pre-existing PRD risk)
- **Description:** PRD §9.2 makes wrist PCK@0.05 on TrackID3x3 a primary Phase C acceptance gate. PRD §13 risk #5 flags that TrackID3x3 may only label a subset of keypoints that doesn't include wrists.
- **Why it matters:** If wrists aren't labeled, the Phase C PCK eval has to fall back to MultiSubjects-only, and the wrist-PCK ≥ 0.85 gate moves to a different dataset.
- **Resolution / next step:** First action of Phase C: load TrackID3x3 annotation files and check whether wrist is in the keypoint schema. Update Phase C scope on the answer.

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
