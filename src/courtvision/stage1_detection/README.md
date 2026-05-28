# Stage 1 — Player Detection + ByteTrack

Per-camera player detection (YOLOv8m, COCO-pretrained, person class only) +
multi-object tracking (ByteTrack via ultralytics' built-in). Outputs one
`TrackSequence` JSON per (session, camera). See PRD §9.1.

## Public API

```python
from courtvision.stage1_detection.yolo_bytetrack import run_stage1
from courtvision.stage1_detection.eval import evaluate_trackid3x3
```

- **`run_stage1(video_path: Path, config: dict) -> TrackSequence`** — runs
  inference end-to-end on one clip. `config` is shaped like
  `configs/stage1_detection/yolov8m_bytetrack.yaml`. If the configured
  weights path doesn't exist on disk, falls back to ultralytics'
  auto-download (~/.cache/ultralytics/yolov8m.pt) so local dev iterates
  without the Modal artifacts volume mounted.
- **`evaluate_trackid3x3(tracks_dir, gt_dir) -> dict`** — pairs predicted
  `tracks_*.json` files with MOT-format GT files (one per sequence) and
  returns per-sequence + overall MOTA / IDF1 / IDP / IDR. Delimiter-agnostic
  (handles TrackID3x3 Indoor + Outdoor CSV and Drone whitespace formats).

## Locked knobs (PRD §9.1)

| Knob | Value |
|---|---|
| Model | `yolov8m` (COCO-pretrained) |
| Classes | `[0]` (person only) |
| Confidence | 0.4 |
| IoU | 0.5 |
| Image size | 1280 |
| Tracker | ByteTrack (ultralytics built-in `bytetrack.yaml`) |
| Bbox format | top-left `[x, y, w, h]` (MOT convention) |

## Output schema

`TrackSequence` (PRD §7.1) — `session_id`, `camera_id`, `fps`, `n_frames`,
and a flat `tracks: list[Track]`. Each `Track` has `track_id`, `bbox`,
`confidence`, `frame_idx`. Bbox coordinates are pixels at the source clip's
native resolution.

Modal writes outputs to `/artifacts/stage1/<session>/tracks_<session>_<cam>.json`.

## Running it

**Local (laptop) — quick smoke / debug:**

```bash
uv run python scripts/stage1_run.py \
    input=data/mirror/<clip>.mp4 \
    session_id=demo camera_id=cam0
```

**Modal (T4 GPU) — production runs:**

```bash
uv run modal run scripts/stage1_run.py::run \
    --video-relpath trackid3x3/videos/<clip>.mp4 \
    --session-id <session> --camera-id <cam>
```

`--video-relpath` is relative to the data-volume mount (`/data`); the
function writes its output to the artifacts volume and prints the relpath.

## Evaluation

**Modal (CPU, reads tracks + GT off Volumes):**

```bash
uv run modal run scripts/stage1_eval.py::run \
    --tracks-relpath stage1/<session> \
    --gt-relpath trackid3x3/ground_truth/Indoor/MOT
```

**Local:**

```bash
uv run python scripts/stage1_eval.py \
    tracks_dir=outputs/stage1/<session> \
    gt_dir=path/to/MOT
```

The script prints a JSON line with `mota`, `idf1`, and a `PASS`/`FAIL`
banner against the 0.70 IDF1 gate.

### Naming convention for GT pairing

`_pair_files` strips the `tracks_` prefix from each pred filename and
looks for `{stem}.txt` in the GT dir, falling back to
`{stem-without-last-_segment}.txt`. So both of these match a GT file
named `basket_S2T1_pre.txt`:

- `session_id=basket_S2T1, camera_id=pre` → `tracks_basket_S2T1_pre.json`
- `session_id=basket_S2T1_pre, camera_id=cam0` → `tracks_basket_S2T1_pre_cam0.json`

## TrackID3x3 GT format

| Subset | Layout | Notes |
|---|---|---|
| Indoor | 10-field CSV | `frame, id, x, y, w, h, conf=-1, x=-1, y=-1, z=-1` |
| Outdoor | 9-field CSV | Same minus trailing `z`; `conf` is real (0.8x) not -1 |
| Drone | 10-field space-separated | Sub-directory per game; one file per clip segment (`<start>_<end>_gt.txt`) |

`_load_gt_frames` handles all three. Frame indices are 1-based per MOT
convention; we normalise to 0-based on load to match `run_stage1` outputs.

## Tuning + post-processing (Phase B persistence work)

Off-the-shelf ByteTrack on TrackID3x3 Indoor scored ~0.56-0.71 IDF1 with
visible ID churn. Two layers fixed it (calibrated, baked into
`yolov8m_bytetrack.yaml`):

1. **ByteTrack knob sweep** (coordinate descent, `scripts/stage1_sweep.py`)
   — winners: `new_track_thresh=0.8`, `fuse_score=false`, `yolo conf=0.3`.
2. **Post-processing** (`postprocess.py`, config block `postprocess:`):
   - **stitch** — rejoin a track that ended at frame F with one starting
     in `[F+1, F+max_gap]` within `base_radius_px * gap` (a player can
     only drift so far in `gap` frames).
   - **filter_non_players** — drop stationary tracks (refs/coaches:
     `mean_speed ≈ 0`) by normalised speed, with a floor of 6 so a real
     player is never pruned and no ceiling so subs survive.

Measured IDF1 (local eval vs GT, mean per-clip ± std):

| Config | dev (S1-S3, pre) | test (S4-S6, post) |
|---|---|---|
| off-the-shelf | 0.556 | — |
| + knob sweep | 0.629 | 0.709 |
| **+ post-processing** | **0.760 ± 0.146** | **0.785 ± 0.083** |

## Acceptance gate

IDF1 ≥ 0.70 on TrackID3x3 (PRD §9.1). **Met** on the held-out test split:
0.785 ± 0.083 (95% CI [0.750, 0.821], lower bound above the gate). Dev
split (harder pre-game footage) also clears it at 0.760. Below this, PRD
escalates to a Phase 2 ★ fine-tune (out of scope for the sprint).

Deferred persistence layers (court mask, team-colour, cross-camera
reconciliation, jersey OCR) are logged in `flags.md` F15.

## Tests

`tests/test_stage1_smoke.py` — synthetic-clip end-to-end smoke (empty
detections expected), perfect-overlap eval correctness (MOTA = IDF1 = 1.0),
space-separated GT regression, and schema round-trip. `uv run pytest
tests/test_stage1_smoke.py`.
