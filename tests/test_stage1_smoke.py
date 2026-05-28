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


def _track(tid: int, frame: int, x: float, y: float, w: float = 20, h: float = 40) -> Track:
    return Track(track_id=tid, bbox=[x, y, w, h], confidence=0.9, frame_idx=frame)


def test_postprocess_features_distinguish_mover_from_stationary() -> None:
    from courtvision.stage1_detection.postprocess import compute_track_features

    tracks = []
    # id 1: roams across the frame
    for f in range(10):
        tracks.append(_track(1, f, x=10 + f * 50, y=10 + f * 30))
    # id 2: stationary (a "ref")
    for f in range(10):
        tracks.append(_track(2, f, x=500, y=500))
    seq = TrackSequence(session_id="s", camera_id="c", fps=30, n_frames=10, tracks=tracks)
    feats = compute_track_features(seq, frame_w=1000, frame_h=1000)
    assert feats[1].spatial_extent > feats[2].spatial_extent
    assert feats[2].spatial_extent == pytest.approx(0.0)
    assert feats[1].mean_speed > feats[2].mean_speed


def test_postprocess_filter_drops_stationary_keeps_floor() -> None:
    from courtvision.stage1_detection.postprocess import filter_non_players

    tracks = []
    # 6 movers (players) + 1 stationary (ref)
    for tid in range(1, 7):
        for f in range(10):
            tracks.append(_track(tid, f, x=10 + f * 40 + tid * 5, y=10 + f * 25))
    for f in range(10):
        tracks.append(_track(99, f, x=900, y=900))  # ref, no motion
    seq = TrackSequence(session_id="s", camera_id="c", fps=30, n_frames=10, tracks=tracks)

    out = filter_non_players(seq, frame_w=1000, frame_h=1000, min_players=6, mobility_floor=0.01)
    kept_ids = {t.track_id for t in out.tracks}
    assert 99 not in kept_ids  # ref dropped
    assert kept_ids == {1, 2, 3, 4, 5, 6}


def test_postprocess_filter_floor_prevents_overprune() -> None:
    """If only 5 movers + 1 ref, floor of 6 keeps all 6 (won't drop a real player below floor)."""
    from courtvision.stage1_detection.postprocess import filter_non_players

    tracks = []
    for tid in range(1, 6):  # 5 movers
        for f in range(10):
            tracks.append(_track(tid, f, x=10 + f * 40, y=10 + f * 25 + tid * 7))
    for f in range(10):
        tracks.append(_track(99, f, x=900, y=900))  # 1 stationary
    seq = TrackSequence(session_id="s", camera_id="c", fps=30, n_frames=10, tracks=tracks)
    out = filter_non_players(seq, frame_w=1000, frame_h=1000, min_players=6, mobility_floor=0.5)
    # mobility_floor very high → nothing passes → floor keeps top 6 (all of them).
    assert len({t.track_id for t in out.tracks}) == 6


def test_postprocess_stitch_rejoins_fragment() -> None:
    from courtvision.stage1_detection.postprocess import stitch_tracks

    tracks = []
    # id 1: frames 0-4 moving right
    for f in range(5):
        tracks.append(_track(1, f, x=100 + f * 10, y=200))
    # id 2: frames 7-11, starts near where id 1 left off (gap=2, small distance)
    for f in range(7, 12):
        tracks.append(_track(2, f, x=150 + (f - 7) * 10, y=205))
    seq = TrackSequence(session_id="s", camera_id="c", fps=30, n_frames=12, tracks=tracks)
    out = stitch_tracks(seq, max_gap_frames=10, base_radius_px=50.0)
    # id 2 should be relabelled to id 1.
    assert {t.track_id for t in out.tracks} == {1}


def test_postprocess_stitch_respects_distance_gate() -> None:
    from courtvision.stage1_detection.postprocess import stitch_tracks

    tracks = []
    for f in range(5):
        tracks.append(_track(1, f, x=100, y=100))
    # id 2 starts far away (gap=1, distance ~800) → should NOT merge.
    for f in range(5, 10):
        tracks.append(_track(2, f, x=900, y=100))
    seq = TrackSequence(session_id="s", camera_id="c", fps=30, n_frames=10, tracks=tracks)
    out = stitch_tracks(seq, max_gap_frames=10, base_radius_px=50.0)
    assert {t.track_id for t in out.tracks} == {1, 2}  # unchanged


def test_sweep_set_by_path_creates_nested_keys() -> None:
    from courtvision.stage1_detection.sweep import set_by_path

    base = {"model": {"conf": 0.4}, "tracker": {"overrides": {}}}
    out = set_by_path(base, "tracker.overrides.track_buffer", 90)
    assert out["tracker"]["overrides"]["track_buffer"] == 90
    # original unchanged
    assert base["tracker"]["overrides"] == {}

    out2 = set_by_path({}, "a.b.c", 1)
    assert out2 == {"a": {"b": {"c": 1}}}


def test_sweep_clip_name_split() -> None:
    from courtvision.stage1_detection.sweep import clip_name_to_session_camera

    assert clip_name_to_session_camera("basket_S1T1_pre") == ("basket_S1T1", "pre")
    assert clip_name_to_session_camera("basket_S6T7_post") == ("basket_S6T7", "post")


def test_sweep_summarise_and_pick_best() -> None:
    from courtvision.stage1_detection.sweep import pick_best, summarise_eval_result

    eval_a = {"per_sequence": {f"c{i}": {"idf1": v, "mota": 0.5} for i, v in enumerate([0.5, 0.6, 0.7])}}
    eval_b = {"per_sequence": {f"c{i}": {"idf1": v, "mota": 0.5} for i, v in enumerate([0.4, 0.4, 0.4])}}

    sa = summarise_eval_result(tag="a", knob="x", value=1, eval_result=eval_a)
    sb = summarise_eval_result(tag="b", knob="x", value=2, eval_result=eval_b)

    assert sa.idf1_mean == pytest.approx(0.6)
    assert sa.n_clips == 3
    assert sb.idf1_mean == pytest.approx(0.4)
    best = pick_best([sa, sb])
    assert best.tag == "a"


def test_resolve_tracker_cfg_default_passes_through() -> None:
    """No overrides → return the bundled cfg name unchanged."""
    from courtvision.stage1_detection.yolo_bytetrack import _resolve_tracker_cfg

    assert _resolve_tracker_cfg({"cfg": "bytetrack.yaml"}) == "bytetrack.yaml"
    assert _resolve_tracker_cfg({"cfg": "bytetrack.yaml", "overrides": {}}) == "bytetrack.yaml"


def test_resolve_tracker_cfg_writes_yaml_for_overrides() -> None:
    """With overrides → returns a path to a temp YAML containing the merged config."""
    import yaml

    from courtvision.stage1_detection.yolo_bytetrack import _resolve_tracker_cfg

    overrides = {"track_buffer": 90, "match_thresh": 0.7}
    out = _resolve_tracker_cfg({"cfg": "bytetrack.yaml", "overrides": overrides})
    out_path = Path(out)
    assert out_path.exists()
    merged = yaml.safe_load(out_path.read_text())
    assert merged["track_buffer"] == 90
    assert merged["match_thresh"] == pytest.approx(0.7)
    assert merged["tracker_type"] == "bytetrack"  # default key preserved


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
