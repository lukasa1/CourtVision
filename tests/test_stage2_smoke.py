"""Stage 2 smoke — Phase C.

All laptop-runnable: the eval + conversion logic is pure Python. The real
RTMPose inference needs the OpenMMLab stack (Modal-only, F1/F17), so that test
is guarded with ``importorskip("mmpose")`` and skips on the laptop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from courtvision.io_utils import read_json_as, write_json
from courtvision.schemas import Keypoint2D, Pose2D, Pose2DSequence
from courtvision.stage2_pose.eval import (
    BLAZEPOSE33_TO_COCO17,
    _bbox_diag,
    _load_trackid3x3_pose_gt,
    _score_trackid3x3_clip,
    _wrist_distances,
    evaluate_trackid3x3,
)
from courtvision.stage2_pose.halpe_to_coco17 import HALPE26_TO_COCO17, halpe26_to_coco17

# Real TrackID3x3 pose-GT samples shipped under examples/.
_GT_DIR = Path(__file__).parent.parent / "examples/trackid3x3/ground_truth/Indoor/pose"


def _pose(
    frame_idx: int,
    track_id: int,
    overrides: dict[int, tuple[float, float]],
    base: tuple[float, float] = (0.0, 0.0),
) -> Pose2D:
    """Build a 26-keypoint Halpe Pose2D; `overrides` sets specific indices."""
    kps = [Keypoint2D(x=base[0], y=base[1], score=1.0) for _ in range(26)]
    for i, (x, y) in overrides.items():
        kps[i] = Keypoint2D(x=x, y=y, score=1.0)
    return Pose2D(
        frame_idx=frame_idx, track_id=track_id, keypoints=kps, keypoint_format="halpe26"
    )


# ---------------------------------------------------------------------------
# Keypoint conversions
# ---------------------------------------------------------------------------


def test_halpe26_to_coco17() -> None:
    kps = [Keypoint2D(x=float(i), y=float(i) * 2, score=0.9) for i in range(26)]
    pose = Pose2D(frame_idx=0, track_id=0, keypoints=kps, keypoint_format="halpe26")
    converted = halpe26_to_coco17(pose)
    assert converted.keypoint_format == "coco17"
    assert len(converted.keypoints) == 17
    for new_i, src_i in enumerate(HALPE26_TO_COCO17):
        assert converted.keypoints[new_i] == kps[src_i]


def test_halpe26_to_coco17_rejects_wrong_format() -> None:
    kps = [Keypoint2D(x=0.0, y=0.0, score=0.0) for _ in range(17)]
    bad = Pose2D(frame_idx=0, track_id=0, keypoints=kps, keypoint_format="coco17")
    with pytest.raises(ValueError, match="halpe26"):
        halpe26_to_coco17(bad)


def test_blazepose33_to_coco17_mapping() -> None:
    assert len(BLAZEPOSE33_TO_COCO17) == 17
    # COCO nose←BlazePose0; COCO L-wrist(9)←BlazePose15; COCO R-wrist(10)←BlazePose16.
    assert BLAZEPOSE33_TO_COCO17[0] == 0
    assert BLAZEPOSE33_TO_COCO17[9] == 15
    assert BLAZEPOSE33_TO_COCO17[10] == 16


# ---------------------------------------------------------------------------
# Wrist scoring
# ---------------------------------------------------------------------------


def test_wrist_distances_perfect_and_swapped() -> None:
    # GT wrists: R(idx5)=(100,100), L(idx6)=(200,100).
    gt = [(0.0, 0.0)] * 10
    gt[5] = (100.0, 100.0)
    gt[6] = (200.0, 100.0)

    correct = _pose(0, 1, {10: (100.0, 100.0), 9: (200.0, 100.0)})
    sided, agn = _wrist_distances(correct, gt)
    assert sided == [0.0, 0.0]
    assert agn == [0.0, 0.0]

    # L/R swapped: sided off by 100 each, agnostic recovers 0.
    swapped = _pose(0, 1, {10: (200.0, 100.0), 9: (100.0, 100.0)})
    sided_s, agn_s = _wrist_distances(swapped, gt)
    assert sided_s == [100.0, 100.0]
    assert agn_s == [0.0, 0.0]


def test_score_clip_perfect_and_threshold() -> None:
    gt_kpts = [
        (0.0, 0.0), (10.0, 50.0), (30.0, 50.0), (8.0, 90.0), (32.0, 90.0),
        (5.0, 130.0), (35.0, 130.0), (20.0, 150.0), (12.0, 200.0), (28.0, 200.0),
    ]
    gt = {0: {"Red_Jersey": gt_kpts}}
    diag = _bbox_diag(gt_kpts)

    pred_perfect = _pose(0, 7, {10: gt_kpts[5], 9: gt_kpts[6]}, base=(20.0, 100.0))
    seq = Pose2DSequence(session_id="s", camera_id="c", fps=20.0, poses=[pred_perfect])
    s = _score_trackid3x3_clip(seq, gt, pck_threshold=0.05)
    assert s["wrist_pck"] == pytest.approx(1.0)
    assert s["n_matched_players"] == 1.0

    off = 0.05 * diag + 1.0
    pred_off = _pose(
        0, 7,
        {10: (gt_kpts[5][0] + off, gt_kpts[5][1]), 9: (gt_kpts[6][0] + off, gt_kpts[6][1])},
        base=(20.0, 100.0),
    )
    seq_off = Pose2DSequence(session_id="s", camera_id="c", fps=20.0, poses=[pred_off])
    s_off = _score_trackid3x3_clip(seq_off, gt, pck_threshold=0.05)
    assert s_off["wrist_pck"] == pytest.approx(0.0)


def test_score_clip_matches_players_regardless_of_order() -> None:
    a = [
        (0.0, 0.0), (10.0, 40.0), (30.0, 40.0), (8.0, 80.0), (32.0, 80.0),
        (5.0, 120.0), (35.0, 120.0), (20.0, 140.0), (12.0, 180.0), (28.0, 180.0),
    ]
    b = [(x + 500.0, y) for (x, y) in a]
    gt = {0: {"Red_Jersey": a, "Blue_Jersey": b}}
    pred_b = _pose(0, 2, {10: b[5], 9: b[6]}, base=(520.0, 90.0))
    pred_a = _pose(0, 1, {10: a[5], 9: a[6]}, base=(20.0, 90.0))
    seq = Pose2DSequence(session_id="s", camera_id="c", fps=20.0, poses=[pred_b, pred_a])
    s = _score_trackid3x3_clip(seq, gt, pck_threshold=0.05)
    assert s["n_matched_players"] == 2.0
    assert s["wrist_pck"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end on real GT data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _GT_DIR.exists(), reason="TrackID3x3 pose GT sample not present")
def test_evaluate_trackid3x3_on_real_gt(tmp_path: Path) -> None:
    """Synthesize a perfect prediction from a real GT file → wrist_pck ≈ 1.0.

    Exercises the real GT parser + spatial matching + scoring end-to-end.
    """
    gt_file = _GT_DIR / "basket_S4T4_post.json"
    gt = _load_trackid3x3_pose_gt(gt_file)

    poses: list[Pose2D] = []
    for frame_idx, players in gt.items():
        for tid, (_colour, kpts) in enumerate(players.items()):
            cx = sum(p[0] for p in kpts) / len(kpts)
            cy = sum(p[1] for p in kpts) / len(kpts)
            poses.append(_pose(frame_idx, tid, {10: kpts[5], 9: kpts[6]}, base=(cx, cy)))
    seq = Pose2DSequence(session_id="basket_S4T4", camera_id="post", fps=20.0, poses=poses)

    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    write_json(seq, poses_dir / "pose2d_basket_S4T4_post.json")

    result = evaluate_trackid3x3(poses_dir, _GT_DIR, pck_threshold=0.05)
    assert result["n_sequences"] == 1
    assert result["summary"]["wrist_pck"] == pytest.approx(1.0)
    assert result["summary"]["n_wrist_keypoints"] > 1000  # dense: 6 players x frames x 2


def test_pose2dsequence_round_trip(tmp_path: Path) -> None:
    seq = Pose2DSequence(
        session_id="rt", camera_id="cam0", fps=20.0,
        poses=[_pose(0, 1, {9: (1.0, 2.0), 10: (3.0, 4.0)})],
    )
    p = tmp_path / "pose2d_rt_cam0.json"
    write_json(seq, p)
    assert read_json_as(Pose2DSequence, p) == seq


# ---------------------------------------------------------------------------
# Inference (Modal-only stack) — skipped on the laptop
# ---------------------------------------------------------------------------


def test_run_stage2_importable_without_mmpose() -> None:
    """The module imports cleanly even though mmpose isn't installed locally."""
    from courtvision.stage2_pose.rtmpose_inference import run_stage2  # noqa: F401


def test_run_stage2_inference_smoke() -> None:
    pytest.importorskip("mmpose", reason="OpenMMLab stack is Modal-only (F1/F17)")
    # Real inference is validated on Modal via the Phase C user-test step.
