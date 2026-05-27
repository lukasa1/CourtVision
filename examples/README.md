# examples/

Small representative samples (~160 MB total, 152 files) from every dataset on the `courtvision-data` Modal Volume. Use these locally for dataloader iteration and visual inspection without pulling 100+ GB to the laptop.

**Contents are git-ignored** — only this README is committed. Regenerate any time by re-running the sampler and resyncing (instructions at the bottom).

| Subdir | Source on Volume | Files | Size | What's here |
|---|---|---:|---:|---|
| `deepsportradar/` | `/data/deepsportradar/` | 5 | 18 MB | Master `basketball-instants-dataset.json` + 2 images each from 2 French-arena dirs (Stage 3 ball-detection mAP primary eval — published ACM MMSports 2022 benchmark). |
| `finesports/` | `/data/finesports/FineSports_CVPR2024.zip` (nested zip) | 102 | 17 MB | 5 frames per coarse action class (24 classes) from each class's first clip under `NewFrames/<class>/<video_id>_<person_id>/*.jpg`. Stream-extracted through both zip layers; full videos never extracted. |
| `multisubjects/` | `/data/multisubjects/` | 19 | 112 MB | SA annotations (`train.pkl` + `val.pkl`), full person GT (4 CSVs + tubes XLSX), 3 small keypoint CSVs, all 4 train/val/test manifests + `Action_class_label.txt` (3-class: dribble/layup/shoot → 0/1/2), and one `.mp4` clip per action class (`809_d_2.mp4`, `42_p_1.mp4`, `372_s_5.mp4`). The `372_s_5.mp4` clip cross-references the `372/372_s_5.csv` keypoints — useful for dataloader sanity checks. |
| `roboflow_basketball/` | `/data/roboflow_basketball/` | 20 | 7 MB | 5 images per split (`train/`/`valid/`/`test/`) + each split's `_annotations.coco.json` + both READMEs. Stage 3 ball-detection mAP secondary eval. |
| `trackid3x3/` | `/data/trackid3x3/` | 9 | 16 MB | `README.md`, 2 pose JSONs + 2 MOT GT files from `ground_truth/Indoor/`, one sample GIF, **and 3 actual Indoor `.mp4` clips** (`basket_S1T1_pre.mp4`, `basket_S4T4_post.mp4`, `basket_S4T6_post.mp4`). The S4T4 and S4T6 clips pair with the pose JSONs of the same name — overlay the 10 keypoints on the video to do a Stage-2 wrist-PCK sanity check. Stage 1 IDF1 eval + Stage 2 wrist-PCK eval pipeline. |

## Why some datasets aren't fully present

- **TrackID3x3 videos** — Indoor MP4s are now on the Volume at `/data/trackid3x3/videos/Indoor/raw/` (42 clips, ~110 MB total). Outdoor + Drone videos and the upstream baseline-tracker outputs are uploading in a background job at the time of this writing (~61 GB → will land at `/data/trackid3x3/videos/{Outdoor,Drone}/` and `/data/trackid3x3/output/`). Indoor is what Phase B/C eval actually needs; Outdoor/Drone are out of PRD scope but preserved for stretch experiments.
- **MultiSubjects bulk data** — the 4 big RAR archives (~40 GB) stay on the Volume; we only pulled a representative slice (3 small `.mp4`s, the manifests, full person GT, and a few keypoint CSVs). When the Stage-5 dataloader runs in Phase E3, it'll read straight from `/data/multisubjects/data/*.rar` via the `rarfile` + `unar` tooling now in the shared Modal image.
- **FineSports videos / fine-grained labels** — only frames are on the Volume; the 52-class fine-grained labels + spatio-temporal bboxes + official splits live in `FineSports-GT.pkl`, gated behind an email request to the FineSports authors (separate ongoing thread).
- **WASB basketball subset** — upstream link is 404; deferred per `flags.md` F11.

## How to regenerate

After re-downloading any source dataset on Modal:

```sh
modal run scripts/download_datasets.py::build_examples_archive

# Preserve this README, since modal volume get won't merge into an existing dir.
cp examples/README.md /tmp/courtvision-examples-README.md
rm -rf examples _examples
modal volume get courtvision-data _examples ./
mv _examples examples
mv /tmp/courtvision-examples-README.md examples/README.md
```

The sampler in `scripts/download_datasets.py::build_examples_archive` is deterministic — re-runs produce the same examples.
