"""Stage 6 Modal runner — 2D pose-only release detection + qualitative overlays.

Single-camera, no ball, no 3D (flags F23). Validates on MultiSubjects close-up
shot clips first (whole-clip window = the trimmed shot), then TrackID3x3
detected shots. Pure NumPy + cv2, so it runs on the shared CPU image.
"""

from __future__ import annotations

from courtvision.modal_app import VOLUMES, app, artifacts_volume, data_volume, image

_SA_DIR = "/data/multisubjects/data/MultiSubjects_SA"
_RAR = "/data/multisubjects/data/MultiSubjects_train_val_test.rar"


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def release_multisubjects(n_clips: int = 12, min_frames: int = 40, max_frames: int = 120) -> dict:
    """Run the 2D pose release detector on N MultiSubjects shot clips (label 2,
    well-tracked wrist, clean single-shot length) and render pose_apex vs
    pose_forward comparison overlays. Whole clip is the search window."""
    import glob
    import json
    import pickle
    import zipfile
    from pathlib import Path

    import numpy as np
    import rarfile

    from courtvision.stage6_release import analyze_release
    from courtvision.stage6_release.overlay import make_release_overlay_video

    data_volume.reload()
    artifacts_volume.reload()
    sa = "/root/sa_rel"
    with zipfile.ZipFile(f"{_SA_DIR}.zip") as zf:
        zf.extractall(sa)
    val = pickle.load(open(glob.glob(f"{sa}/**/val.pkl", recursive=True)[0], "rb"))

    def wrist_quality(x: dict) -> float:
        ks = np.asarray(x["keypoint_score"])[0]  # [T, 17]
        return float(np.maximum(ks[:, 9], ks[:, 10]).mean())

    cands = [
        x
        for x in val
        if x["label"] == 2
        and min_frames <= x["total_frames"] <= max_frames
        and float(np.asarray(x["keypoint"]).max()) > 0
    ]
    cands.sort(key=wrist_quality, reverse=True)
    chosen = cands[:n_clips]

    rf = rarfile.RarFile(_RAR)
    names = rf.namelist()
    src_dir = Path("/artifacts/stage6_release_src")
    out_dir = Path("/artifacts/stage6_release_overlays")
    src_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for c in chosen:
        fd = c["frame_dir"]
        member = next((n for n in names if n.endswith(f"/{fd}.mp4")), None)
        if member is None:
            continue
        vid = src_dir / f"{fd}.mp4"
        vid.write_bytes(rf.read(member))
        kp = np.asarray(c["keypoint"])[0]
        ks = np.asarray(c["keypoint_score"])[0]
        analysis = analyze_release(kp, ks, fd)
        make_release_overlay_video(vid, kp, ks, analysis, out_dir / f"{fd}_release.mp4")
        results.append(
            {
                "frame_dir": fd,
                "n_frames": int(c["total_frames"]),
                "side": analysis["side"],
                "apex_frame": analysis["apex"].release_frame_idx,
                "apex_conf": analysis["apex"].confidence_flag,
                "forward_frame": analysis["forward"].release_frame_idx,
                "forward_conf": analysis["forward"].confidence_flag,
                "overlay": f"stage6_release_overlays/{fd}_release.mp4",
            }
        )
    artifacts_volume.commit()
    print(json.dumps(results, indent=2))
    return {"clips": results}


@app.local_entrypoint()
def release_ms(n_clips: int = 12) -> None:
    """uv run modal run scripts/stage6_modal.py::release_ms — render 2D release
    overlays on N MultiSubjects shot clips for qualitative review."""
    import json

    res = release_multisubjects.remote(n_clips)
    print(json.dumps(res, indent=2))
    print(
        "\npull each: uv run modal volume get --force courtvision-artifacts "
        "<overlay-path> outputs/stage6_release_overlays/"
    )


# Stage 5 detected shot events (≥0.6) on TrackID3x3 indoor clips, from
# `eval_actions`. User assessment (2026-05-28): first 4 are real shots,
# S6T4/S6T1 fires were false positives, S4T6 has a real shot at 0.66 + 2 FPs.
_T3_DETECTED_EVENTS: list[dict] = [
    {"clip": "basket_S4T4_post", "track": 4, "peak": 200, "stage5_conf": 0.94},
    {"clip": "basket_S5T7_post", "track": 3, "peak": 112, "stage5_conf": 0.92},
    {"clip": "basket_S5T5_post", "track": 4, "peak": 112, "stage5_conf": 0.90},
    {"clip": "basket_S6T4_post", "track": 4, "peak": 88, "stage5_conf": 0.82},
    {"clip": "basket_S6T4_post", "track": 9, "peak": 237, "stage5_conf": 0.73},
    {"clip": "basket_S6T1_post", "track": 6, "peak": 96, "stage5_conf": 0.67},
    {"clip": "basket_S4T6_post", "track": 3, "peak": 192, "stage5_conf": 0.66},
    {"clip": "basket_S4T6_post", "track": 6, "peak": 176, "stage5_conf": 0.65},
    {"clip": "basket_S4T6_post", "track": 8, "peak": 148, "stage5_conf": 0.63},
]


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def release_trackid3x3(events: list[dict] | None = None, window_half: int = 50) -> dict:
    """Apply Stage 6 release detection to Stage 5 detected shots on TrackID3x3.

    For each event, take the shooter track's pose (Halpe-26 → COCO-17, gaps
    held), open a ``±window_half``-frame search window around the Stage 5
    peak, and render a focused release overlay. Realistic cross-domain test:
    multi-player gameplay rather than the MultiSubjects close-ups."""
    import json
    from collections import defaultdict
    from pathlib import Path

    import cv2
    import numpy as np

    from courtvision.schemas import Pose2DSequence
    from courtvision.stage2_pose.halpe_to_coco17 import halpe26_to_coco17
    from courtvision.stage6_release import analyze_release
    from courtvision.stage6_release.overlay import make_release_overlay_video

    data_volume.reload()
    artifacts_volume.reload()
    events = events if events is not None else list(_T3_DETECTED_EVENTS)
    out_dir = Path("/artifacts/stage6_release_overlays/trackid3x3")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    pose_cache: dict[str, Pose2DSequence] = {}
    for ev in events:
        clip, tid, peak = ev["clip"], int(ev["track"]), int(ev["peak"])
        vid_path = f"/data/trackid3x3/videos/Indoor/raw/{clip}.mp4"
        pose_path = f"/artifacts/stage2_batch/pose2d_{clip}.json"
        if not Path(vid_path).exists() or not Path(pose_path).exists():
            continue
        if clip not in pose_cache:
            pose_cache[clip] = Pose2DSequence.model_validate_json(Path(pose_path).read_text())
        seq = pose_cache[clip]
        cap = cv2.VideoCapture(vid_path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        by_track: dict[int, list] = defaultdict(list)
        for p in seq.poses:
            by_track[p.track_id].append(p)
        tposes = sorted(by_track.get(tid, []), key=lambda p: p.frame_idx)
        if not tposes:
            continue
        present = {p.frame_idx: p for p in tposes}
        kp = np.zeros((nframes, 17, 2), np.float32)
        ks = np.zeros((nframes, 17), np.float32)
        last: tuple | None = None
        for i in range(nframes):
            p = present.get(i)
            if p is not None:
                c = halpe26_to_coco17(p)
                kp[i] = [[k.x, k.y] for k in c.keypoints]
                ks[i] = [k.score for k in c.keypoints]
                last = (kp[i].copy(), ks[i].copy())
            elif last is not None:
                kp[i], ks[i] = last
        lo, hi = max(0, peak - window_half), min(nframes, peak + window_half)
        ev_id = f"{clip}_t{tid}_f{peak}"
        analysis = analyze_release(kp, ks, ev_id, window=(lo, hi))
        out_rel = f"stage6_release_overlays/trackid3x3/{ev_id}_release.mp4"
        make_release_overlay_video(Path(vid_path), kp, ks, analysis, Path("/artifacts") / out_rel)
        results.append(
            {
                "event_id": ev_id,
                "clip": clip,
                "track": tid,
                "stage5_peak": peak,
                "stage5_conf": ev["stage5_conf"],
                "side": analysis["side"],
                "forward_frame": analysis["forward"].release_frame_idx,
                "forward_conf": analysis["forward"].confidence_flag,
                "apex_frame": analysis["apex"].release_frame_idx,
                "apex_conf": analysis["apex"].confidence_flag,
                "overlay": out_rel,
            }
        )
    artifacts_volume.commit()
    print(json.dumps(results, indent=2))
    return {"events": results}


@app.local_entrypoint()
def release_t3(window_half: int = 50) -> None:
    """uv run modal run scripts/stage6_modal.py::release_t3 — apply Stage 6 to
    every Stage 5 detected shot on TrackID3x3 indoor clips (≥0.6 confidence)."""
    import json

    res = release_trackid3x3.remote(None, window_half)
    print(json.dumps(res, indent=2))
    print(
        "\npull all: for f in $(uv run modal volume ls courtvision-artifacts "
        "stage6_release_overlays/trackid3x3 | grep -oE 'stage6_release_overlays/trackid3x3/.+\\.mp4'); "
        "do uv run modal volume get --force courtvision-artifacts \"$f\" "
        "outputs/stage6_release_overlays/trackid3x3/; done"
    )
