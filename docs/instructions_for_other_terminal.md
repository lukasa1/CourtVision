# Instructions: filling in the report-tracking docs

You and I are collaborating on the same CourtVision project from two different
Claude Code terminals. I (this terminal) have been focused on Stages 1, 2, 3,
5, 6, plus the F24 cross-method filter. You have been with Lukas longer and own
the 3D pipeline (Stages 4 and 7), plus likely some other work I haven't seen
(Decision 9 redesign / F25 / schema updates for `wrist_to_wrist_distance_m`,
`shooter_fade_mps`, etc.).

I just wrote two consolidated markdown trackers in `docs/` for Lukas's final
CS231N report. They're nearly complete for my side; **your side has explicit
gaps that need filling, plus latitude to add anything else you think the report
needs.** Treat this document as the hand-off note.

---

## Files you'll be editing

- **`docs/development_log.md`** (660 lines). Narrative, organised by stage. Has
  a "Report figures + tables" section in the back half that maps recommended
  figures to CS231N report sections.
- **`docs/report_content.md`** (481 lines). Factual dump: exact numbers, file
  paths, hyperparameters, BibTeX. **`NOT DONE`** appears wherever something
  was never measured or never built.

## Read these first (so you understand structure + tone before editing)

1. `docs/development_log.md` and `docs/report_content.md` — the targets.
2. `flags.md` — you may have already authored **F25** (Stage 7 pose-only
   redesign post-Phase-D-drop, 2026-06-02). Cross-check that it's reflected in
   both target docs.
3. `docs/decisions.md` — you may have authored the **Decision 9 redesign**
   block (2026-06-02). Same cross-check.
4. `Milestone 3 - Preliminary Results/Milestone3_Report.md` — Lukas's draft
   report. If you've been editing it, make sure the trackers don't contradict
   what's there.

---

## Rules (apply to both docs, same as the rest of the content)

- **Exact numbers only.** No estimation, no invented metrics, no
  "approximately." If something wasn't measured, write `NOT DONE` explicitly —
  do *not* fill a number to make the doc look complete.
- **Paths must exist.** Every file path you cite should be a real path on disk
  or on the Modal volume.
- **Code-level claims** point at the file/function that implements them.
- **Tone differs between the two docs.**
  - `development_log.md` is narrative ("we did X because Y; result: Z; key files:
    A, B, C"). Match the prose style of the existing stage sections.
  - `report_content.md` is bullet-and-table factual ("Optimizer: SGD. LR:
    0.01."). No prose; just facts.

---

## Specific gaps to fill

### A. `docs/development_log.md` § 5 — **Stage 4 (3D Triangulation)**

Currently a placeholder with suggested subsection structure. Replace with the
actual content. Fill in each subsection:

- **Built.** What you implemented (Pose2Sim wrapper? custom DLT? RANSAC across
  3 cameras? per-frame visibility logging? ball pipeline retained or dropped
  given F19?).
- **Deferred / Changed.** Any deviations from Decision 4 and Decision 10.
- **Decisions.** Decision 4 (locked) + Decision 10 (calibration) + any new
  ones you made.
- **Results.** Reprojection error on held-out 3rd camera (Decision 4 target
  < 5 px). Visibility-count histogram (3 / 2 / 0 cameras per keypoint).
  Calibration-drift detection behaviour. **`NOT DONE`** if any of these aren't
  measured yet (iPhone footage may still be pending).
- **Key files / outputs.** Paths under `src/courtvision/stage4_triangulation/`
  + any calibration JSONs (`intrinsics_*.json`, `extrinsics_*.json`).

### B. `docs/development_log.md` § 10 — **Stage 7 (Contest Metric)**

Same — replace placeholder. Subsections to fill:

- **Built.** The five measurement functions per Decision 9 redesign /
  F25 (`wrist_to_wrist_distance_m`, `wrist_height_advantage_m`,
  `shooter_fade_mps`, `floor_distance_m`, `composite_score`). Schema
  consumer / writer code. Selection rules ("closest defender" by
  pelvis-to-pelvis; "shooting hand" by higher wrist at release).
- **Deferred.** Rater study (★ stretch, per Decision 9); composite weight
  learning; whatever else you scoped out.
- **Decisions.** Decision 9 (original) + Decision 9 redesign (2026-06-02
  pose-only) + F25.
- **Results.** Mikic-2024 effect reproduction (top vs. bottom contest
  quartile → expected higher release point + quicker release).
  Make / miss Spearman ρ. Distribution sanity check. **`NOT DONE`** for any
  you haven't run.
- **Key files / outputs.** Paths under `src/courtvision/stage7_contest/`.

### C. `docs/development_log.md` "Report figures + tables" section

The big mapping section at the back. Add Stage 4 / Stage 7 figure
recommendations alongside the existing ones I wrote for Stages 1–6:

- 3D pose visualisation (frame triangulated into a 3D scatter or wireframe)
- Reprojection-error histogram or per-frame line plot
- Visibility-count over time
- Contest-metric distribution (histogram of one of the 5 measurements across
  detected shots)
- A "shot release with defender contest" composite figure (release frame + 3D
  pose + the closest defender's wrist marker)
- Any other figure you think the report needs that I missed

For each: file path (or note as `NOT DONE` if it has to be produced), plus a
one-line caption.

Also: **if you have a pipeline diagram asset**, add it. I flagged the pipeline
diagram as `NOT DONE` in both docs because none exists in `outputs/` yet; if
you've drawn one for Milestone 2 / 3, point at it.

### D. `docs/report_content.md` § 1 — Pipeline inventory

Two rows currently say `NOT DONE in this terminal`:

- **"3D reconstruction"** row — fill in exact model/tooling (Pose2Sim version,
  OpenCV `cv2.triangulatePoints`, RANSAC params), off-the-shelf vs custom,
  what you wrote vs starter code.
- **"Contest metric (Stage 7)"** row — fill in what's built, point at the
  measurement functions in `src/courtvision/stage7_contest/`.

### E. `docs/report_content.md` § 2 — Datasets

Add subsections if relevant:

- **Own iPhone footage** — if collected, fill in number of sessions, number of
  shots labelled, resolution / fps, hand-labelling protocol for releases.
- **Calibration captures** — chessboard videos used for intrinsics;
  court-landmark click data for extrinsics.

If still uncollected: keep my current `NOT DONE` markers, just confirm.

### F. `docs/report_content.md` § 5 — Results

Add subsections that are currently `NOT DONE`:

- **§ 5.6 Stage 4 reprojection error** — exact values if you measured.
  Otherwise leave `NOT DONE` with reason (e.g. "gated on iPhone footage; no
  measurement run yet").
- **New § 5.X Stage 7 contest-metric measurements** — distribution stats,
  Mikic-effect reproduction, Spearman ρ, etc. — if any are measured.
- **§ 5.7 release-frame accuracy** — if you ran release detection on 3D pose,
  add Δframes numbers if you have GT (likely still `NOT DONE`).

Match my existing tone in the section: tables with headers + exact decimals.

### G. `docs/report_content.md` § 6 — Figures

Add file paths + captions for any figures you produced (3D viz, triangulation
quality plots, contest-metric distributions, calibration diagnostic plots).

### H. `docs/report_content.md` § 8 — References

Add BibTeX for citations I missed:

- Pose2Sim (Pagnon et al., already in my list as a `@misc` — upgrade to the
  actual JOSS paper if you used it).
- DLT / camera-calibration foundational papers (Tsai, Zhang chessboard).
- Mikic 2024 (proper citation if you have it).
- Anything from your literature review that I didn't include.

### I. `docs/report_content.md` § 9 — "What's NOT in this document" list

Cross off items you've now completed. Add anything that's still genuinely
`NOT DONE`.

---

## Open invitation: add anything else you think the report needs

You have been with Lukas the longest and have context I don't. If any of these
apply, add them to the appropriate doc:

- **Pivots or decisions I didn't capture.** Whole approaches that were tried
  and dropped, design conversations that shaped the architecture, lessons
  learned that don't fit neatly into a flag entry.
- **Methodology I missed.** Data-labelling protocols, calibration procedures,
  evaluation scripts, the Milestone-2 / Milestone-3 presentation logistics.
- **Failure cases** I didn't see (your view on the project's broader weak
  spots — including Stage 4 / Stage 7 specific ones).
- **References I missed.** Papers from your literature review that aren't in
  my § 8 of `report_content.md`.
- **Process / engineering insights worth documenting** — Modal workflow
  lessons, the multi-terminal collaboration pattern, anything else that's a
  genuine project-level finding.

For `development_log.md`, you can insert new sub-sections under existing stage
sections, or add a new section after § 12 (e.g. "§ 13 — Process /
collaboration notes"). For `report_content.md`, prefer extending existing § 1–9
sections; only add a new top-level section if your content doesn't fit
logically.

---

## Hand-off facts from this terminal (so you don't have to dig)

- **Canonical Stage 5 model** = `/artifacts/checkpoints/stage5/stage2_multisubjects.pth`
  (the v1 64f_bal model, promoted via `promote_ckpt`). The previous 32f model
  is archived at `stage2_multisubjects_32f.pth`.
- **v2 ablation checkpoints** for the before/after fine-tuning report figure:
  - `stage2_multisubjects_64f_bal_v2.pth` (full fine-tune)
  - `stage2_multisubjects_64f_probe_v2.pth` (linear probe)
  - `stage2_multisubjects_64f_bal_v2_reg.pth` (stronger regularisation)
- **Stage 6** locked at `pose_forward` primary, `release_offset = -1`,
  `score_thresh` 0.2, elbow extension threshold 160°, `min_run = 2`,
  `gap_tol = 2`.
- **F24 cross-method gate**: `Stage5 score_threshold = 0.6` (lowered from 0.75)
  fed into `Stage6 is_verified_shot()`. On the 9 TrackID3x3 detected events
  this gave precision 0.80 / recall 1.00 vs. Stage 5 alone @ 0.75 giving
  0.75 / 0.75. Caveat: n = 9, re-validate on iPhone footage.
- **v2 subject-disjoint split** is the canonical evaluation split going
  forward. Algorithm: `src/courtvision/stage5_classifier/split.py::compute_subject_disjoint_split(seed=0)`.
  The native train.pkl / val.pkl split is **contaminated for "test holdout"
  purposes** because the canonical v1 model trained on it — if you re-run any
  Stage 5 eval, use v2.
- If your Stage 4 work changed the schemas (it did — added
  `wrist_to_wrist_distance_m` etc.), make sure my report-content.md doesn't
  still cite the *old* schema names. Search for "wrist_to_trajectory" — if
  it's in my doc, the schema-redesign reference is stale and you should fix.

When you're done editing both files, just save them. No need to ping back.
Thank you.
