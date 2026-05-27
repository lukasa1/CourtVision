"""Stage 1 smoke — Phase B."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtvision.io_utils import read_json_as, write_json
from courtvision.schemas import Track, TrackSequence
from courtvision.stage1_detection.eval import evaluate_trackid3x3


@pytest.fixture
def _base_cfg() -> dict[str, object]:
    return {
        "session_id": "smoke",
        "camera_id": "cam0",
        "model": {
            # Force ultralytics auto-download — local dev has no Modal mount.
            "weights": None,
            "classes": [0],
            "conf": 0.4,
            "iou": 0.5,
            "imgsz": 128,
        },
        "tracker": {"cfg": "bytetrack.yaml"},
    }


def test_run_stage1_emits_valid_sequence(
    synthetic_clip: Path, _base_cfg: dict[str, object]
) -> None:
    """End-to-end: synthetic clip → run_stage1 → schema-valid TrackSequence.

    No people in the clip, so we expect an empty tracks list. The point is
    to exercise decode → tracker → schema round-trip. This triggers a
    one-time ultralytics yolov8m.pt download (~50MB) into ``~/.cache``.
    """
    from courtvision.stage1_detection.yolo_bytetrack import run_stage1

    seq = run_stage1(synthetic_clip, _base_cfg)
    assert isinstance(seq, TrackSequence)
    assert seq.session_id == "smoke"
    assert seq.camera_id == "cam0"
    assert seq.fps > 0
    assert seq.n_frames == 5
    assert seq.tracks == []


def test_evaluate_trackid3x3_perfect_overlap(tmp_path: Path) -> None:
    """Synthetic GT == predictions → MOTA = IDF1 = 1.0."""
    tracks_dir = tmp_path / "tracks"
    gt_dir = tmp_path / "gt"
    tracks_dir.mkdir()
    gt_dir.mkdir()

    # 3 frames, 2 stable tracks, identical pred + GT.
    boxes = {
        0: {1: [10.0, 10.0, 30.0, 60.0], 2: [80.0, 20.0, 30.0, 60.0]},
        1: {1: [12.0, 10.0, 30.0, 60.0], 2: [82.0, 20.0, 30.0, 60.0]},
        2: {1: [14.0, 10.0, 30.0, 60.0], 2: [84.0, 20.0, 30.0, 60.0]},
    }

    pred_tracks = [
        Track(track_id=tid, bbox=bb, confidence=0.99, frame_idx=f)
        for f, frame_dets in boxes.items()
        for tid, bb in frame_dets.items()
    ]
    pred_seq = TrackSequence(
        session_id="syn",
        camera_id="cam0",
        fps=30.0,
        n_frames=3,
        tracks=pred_tracks,
    )
    write_json(pred_seq, tracks_dir / "tracks_syn_cam0.json")

    # MOT format is 1-based frames.
    gt_lines = []
    for f, frame_dets in boxes.items():
        for tid, (x, y, w, h) in frame_dets.items():
            gt_lines.append(f"{f + 1},{tid},{x},{y},{w},{h},1,-1,-1,-1")
    (gt_dir / "syn_cam0.txt").write_text("\n".join(gt_lines))

    result = evaluate_trackid3x3(tracks_dir, gt_dir)
    assert result["n_sequences"] == 1
    assert result["summary"]["mota"] == pytest.approx(1.0)
    assert result["summary"]["idf1"] == pytest.approx(1.0)
    assert "syn_cam0" in result["per_sequence"]


def test_evaluate_trackid3x3_handles_space_separated_gt(tmp_path: Path) -> None:
    """Drone subset GT is whitespace-separated; eval should still match it."""
    tracks_dir = tmp_path / "tracks"
    gt_dir = tmp_path / "gt"
    tracks_dir.mkdir()
    gt_dir.mkdir()

    box = [50.0, 60.0, 30.0, 80.0]
    seq = TrackSequence(
        session_id="drone",
        camera_id="cam0",
        fps=30.0,
        n_frames=1,
        tracks=[Track(track_id=1, bbox=box, confidence=0.9, frame_idx=0)],
    )
    write_json(seq, tracks_dir / "tracks_drone_cam0.json")
    (gt_dir / "drone_cam0.txt").write_text(
        f"1 1 {box[0]} {box[1]} {box[2]} {box[3]} -1 -1 -1 -1\n"
    )

    result = evaluate_trackid3x3(tracks_dir, gt_dir)
    assert result["n_sequences"] == 1
    assert result["summary"]["idf1"] == pytest.approx(1.0)


def test_make_overlay_video_smoke(synthetic_clip: Path, tmp_path: Path) -> None:
    """make_overlay_video on the synthetic clip + empty sequence → valid MP4 with the same frame count."""
    import cv2

    from courtvision.stage1_detection.overlay import make_overlay_video

    seq = TrackSequence(
        session_id="overlay", camera_id="cam0", fps=30.0, n_frames=5, tracks=[]
    )
    out = tmp_path / "overlay.mp4"
    make_overlay_video(synthetic_clip, seq, out)
    assert out.exists() and out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    try:
        n_out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    assert n_out_frames == 5


def test_make_overlay_video_caps_max_seconds(synthetic_clip: Path, tmp_path: Path) -> None:
    """max_seconds=0.1 on a 30fps 5-frame clip should produce ~3 frames."""
    import cv2

    from courtvision.stage1_detection.overlay import make_overlay_video

    seq = TrackSequence(
        session_id="overlay", camera_id="cam0", fps=30.0, n_frames=5, tracks=[]
    )
    out = tmp_path / "overlay_capped.mp4"
    make_overlay_video(synthetic_clip, seq, out, max_seconds=0.1)

    cap = cv2.VideoCapture(str(out))
    try:
        n_out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    # 0.1s @ 30fps = 3 frames; allow ±1 for container quirks.
    assert 2 <= n_out_frames <= 4


def test_evaluate_trackid3x3_round_trip_schema(tmp_path: Path) -> None:
    """Empty-tracks TrackSequence round-trips through write/read."""
    seq = TrackSequence(
        session_id="rt", camera_id="cam0", fps=60.0, n_frames=0, tracks=[]
    )
    p = tmp_path / "tracks_rt_cam0.json"
    write_json(seq, p)
    loaded = read_json_as(TrackSequence, p)
    assert loaded == seq
