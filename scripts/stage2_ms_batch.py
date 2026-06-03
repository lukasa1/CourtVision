"""MultiSubjects per-keypoint COCO-17 PCK eval — Stage 1 + Stage 2 pipeline.

Single-player gym clips with BlazePose-33 GT (subsetted to COCO-17). Workflow:
  1. Extract the requested val clips + their BlazePose CSVs from the on-Volume
     RAR archives to ``/data/multisubjects/eval_workspace/``.
  2. For each clip in parallel waves: Stage 1 (YOLO+ByteTrack) → Stage 2
     (rtmlib RTMPose) → ``Pose2DSequence`` JSON in
     ``/artifacts/stage2_ms_batch/``.
  3. Run ``evaluate_multisubjects`` with per-clip resolution and the per-keypoint
     COCO-17 diagnostic.

Usage::

    uv run modal run scripts/stage2_ms_batch.py::run --clips 372_s_5,938_p_2,...
"""

from __future__ import annotations

import json

from courtvision.logging import get_logger
from courtvision.modal_app import (
    VOLUMES,
    app,
    artifacts_volume,
    data_volume,
    image,
    run_rtmpose,
    run_yolo_bytetrack,
)

_log = get_logger(__name__)

# Stage 1 for MS: single-player gym clips. Calibrated tracker overrides for ID
# stability; no `postprocess` (the calibrated postprocess assumes 6 on-court
# players, see Phase B / flags F15).
_S1_MODEL = {
    "name": "yolov8m",
    "weights": "/artifacts/checkpoints/yolov8m.pt",
    "classes": [0],
    "conf": 0.4,
    "iou": 0.5,
    "imgsz": 1280,
}
_S1_TRACKER = {
    "name": "bytetrack",
    "cfg": "bytetrack.yaml",
    "overrides": {"new_track_thresh": 0.8, "fuse_score": False},
}
_POSE_MODEL = {
    "name": "rtmpose-l",
    "onnx_model": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip"
    ),
    "keypoint_format": "halpe26",
    "num_keypoints": 26,
    "input_size": [192, 256],
    "backend": "onnxruntime",
    "device": "cpu",
}

_WORK_DIR = "multisubjects/eval_workspace"


@app.function(image=image, volumes=VOLUMES, timeout=1800, cpu=2)
def extract_ms_clips(clip_stems: list[str]) -> dict[str, list[str]]:
    """Extract the named clips' videos + BlazePose CSVs from the on-Volume RARs.

    Idempotent: skips files already extracted. Returns counts of extracted files
    and the Volume-relative paths of the workspace dirs.
    """
    from pathlib import Path as _Path

    import rarfile

    data_volume.reload()
    work = _Path("/data") / _WORK_DIR
    videos_dir = work / "videos"
    csvs_dir = work / "key_points_csv"
    videos_dir.mkdir(parents=True, exist_ok=True)
    csvs_dir.mkdir(parents=True, exist_ok=True)

    wanted_videos = {f"{stem}.mp4" for stem in clip_stems}
    wanted_csvs = {f"{stem}.csv" for stem in clip_stems}
    video_rar = _Path("/data/multisubjects/data/MultiSubjects_train_val_test.rar")
    csv_rar = _Path("/data/multisubjects/data/key_points_csv.rar")

    extracted_videos: list[str] = []
    extracted_csvs: list[str] = []

    with rarfile.RarFile(str(video_rar)) as rf:
        for info in rf.infolist():
            base = _Path(info.filename).name
            if base in wanted_videos:
                out = videos_dir / base
                if not out.exists():
                    with rf.open(info) as src, out.open("wb") as dst:
                        dst.write(src.read())
                extracted_videos.append(base)

    with rarfile.RarFile(str(csv_rar)) as rf:
        for info in rf.infolist():
            base = _Path(info.filename).name
            if base in wanted_csvs:
                out = csvs_dir / base
                if not out.exists():
                    with rf.open(info) as src, out.open("wb") as dst:
                        dst.write(src.read())
                extracted_csvs.append(base)

    data_volume.commit()
    _log.info(
        "extracted %d/%d videos, %d/%d csvs",
        len(extracted_videos),
        len(clip_stems),
        len(extracted_csvs),
        len(clip_stems),
    )
    return {
        "videos": extracted_videos,
        "csvs": extracted_csvs,
        "videos_relpath": f"{_WORK_DIR}/videos",
        "csvs_relpath": f"{_WORK_DIR}/key_points_csv",
    }


@app.function(image=image, volumes=VOLUMES, timeout=3600, cpu=2)
def eval_ms(
    poses_relpath: str, gt_relpath: str, videos_relpath: str, pck_threshold: float = 0.05
) -> dict:
    """Run evaluate_multisubjects with per-clip resolution + per-keypoint output."""
    from pathlib import Path as _Path

    from courtvision.stage2_pose.eval import evaluate_multisubjects

    artifacts_volume.reload()
    data_volume.reload()
    poses_dir = _Path("/artifacts") / poses_relpath
    gt_dir = _Path("/data") / gt_relpath
    videos_dir = _Path("/data") / videos_relpath
    result = evaluate_multisubjects(
        poses_dir, gt_dir, pck_threshold=pck_threshold, videos_dir=videos_dir
    )
    out = _Path("/artifacts") / "stage2_ms_batch/eval_stage2_multisubjects.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    artifacts_volume.commit()
    return result


@app.local_entrypoint()
def run(clips: str, pck_threshold: float = 0.05) -> None:
    """Process a comma-separated list of MS val clip stems (e.g. ``372_s_5``)."""
    clip_list = [c.strip() for c in clips.split(",") if c.strip()]
    _log.info("MS batch: %d clips", len(clip_list))

    # 1) Extract videos + csvs.
    ext = extract_ms_clips.remote(clip_list)
    _log.info("extraction: %d videos, %d csvs", len(ext["videos"]), len(ext["csvs"]))

    # 2) Stage 1 (parallel).
    s1 = {}
    for clip in clip_list:
        cfg = {
            "session_id": clip,
            "camera_id": "ms",
            "model": _S1_MODEL,
            "tracker": _S1_TRACKER,
            "output_relpath": f"stage2_ms_batch/tracks_{clip}.json",
        }
        s1[clip] = run_yolo_bytetrack.spawn(f"{_WORK_DIR}/videos/{clip}.mp4", cfg)
    tracks = {c: s1[c].get() for c in clip_list}

    # 3) Stage 2 (parallel).
    s2 = {}
    for clip in clip_list:
        pose_cfg = {
            "model": _POSE_MODEL,
            "output_relpath": f"stage2_ms_batch/pose2d_{clip}.json",
        }
        s2[clip] = run_rtmpose.spawn(f"{_WORK_DIR}/videos/{clip}.mp4", tracks[clip], pose_cfg)
    for c in clip_list:  # wait for poses
        s2[c].get()

    # 4) Eval over the dir.
    result = eval_ms.remote(
        "stage2_ms_batch", ext["csvs_relpath"], ext["videos_relpath"], pck_threshold
    )

    s = result["summary"]
    print(
        json.dumps(
            {
                "n_clips_requested": len(clip_list),
                "n_videos_extracted": len(ext["videos"]),
                "n_clips_evaluated": result["n_sequences"],
                "summary": {
                    "pck": round(s["pck"], 4),
                    "wrist_pck": round(s["wrist_pck"], 4),
                    "n_keypoints": s["n_keypoints"],
                    "per_keypoint": {k: round(v, 4) for k, v in s["per_keypoint"].items()},
                },
                "eval_relpath": "stage2_ms_batch/eval_stage2_multisubjects.json",
                "pull_command": (
                    "uv run modal volume get courtvision-artifacts "
                    "stage2_ms_batch/eval_stage2_multisubjects.json outputs/stage2_gate/"
                ),
            },
            indent=2,
        )
    )
