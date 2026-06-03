"""Stage 2 eval — wrist PCK@0.05 on TrackID3x3 (primary), COCO-17 PCK on MultiSubjects.

PRD §9.2. The headline gate is **wrist PCK@0.05 ≥ 0.85 on TrackID3x3** (the
deployment domain: half-court multi-player). MultiSubjects is a secondary,
single-player corroboration.

TrackID3x3 pose GT specifics (verified — see ``flags.md`` F18):
- Per-frame, **jersey-colour-keyed** (``"Red_Jersey": [[x, y], ...x10]``),
  *not* track-id-keyed — so predictions are spatially matched to GT players
  per frame before scoring.
- 10 keypoints, wrists at GT index **5 (R)** and **6 (L)**; predictions are
  Halpe-26 with wrists at index **10 (R)** and **9 (L)**.
- L/R is ambiguous ~10-20% of the time (hands meeting at the body), so wrist
  PCK uses a **side-agnostic** 2x2 min-cost wrist assignment — it scores
  wrist *localisation*, which is what the Stage 7 contest metric needs,
  independent of the L/R label. A strict same-side number is reported
  alongside for transparency.
- Labels are the dataset's baseline-estimator output, not hand truth, so the
  number is in-domain *agreement with their baseline* (F18).

Normalisation: PCK threshold is ``pck_threshold x diag``, where ``diag`` is
the diagonal of the GT skeleton's keypoint bounding box (the GT pose JSON
carries no person bbox).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment

from courtvision.io_utils import read_json_as
from courtvision.logging import get_logger
from courtvision.schemas import Pose2D, Pose2DSequence

_log = get_logger(__name__)

# Halpe-26 wrist indices (COCO order in the first 17): left=9, right=10.
_HALPE_WRIST_RL = (10, 9)  # (right, left)
# TrackID3x3 10-keypoint GT wrist indices (F5/F18).
_GT_WRIST_RL = (5, 6)  # (right, left)

# TrackID3x3 GT 10-keypoint index -> Halpe-26 index, grouped by landmark type
# (F5 schema: 0 head/nose, 1 R-shoulder, 2 L-shoulder, 3 R-elbow, 4 L-elbow,
# 5 R-wrist, 6 L-wrist, 7 hip-centre, 8 R-ankle, 9 L-ankle; Halpe-26 COCO order:
# 0 nose, 5/6 L/R shoulder, 7/8 L/R elbow, 9/10 L/R wrist, 15/16 L/R ankle,
# 19 hip-centre). Used for the per-landmark diagnostic (wrist remains the gate).
# L/R-paired groups are scored side-agnostically (TrackID3x3 L/R labels are
# flipped vs RTMPose — F18); central groups (head, hip) compare directly.
_GT_LANDMARK_GROUPS: dict[str, list[tuple[int, int]]] = {
    "head": [(0, 0)],
    "shoulder": [(1, 6), (2, 5)],
    "elbow": [(3, 8), (4, 7)],
    "wrist": [(5, 10), (6, 9)],
    "hip": [(7, 19)],
    "ankle": [(8, 16), (9, 15)],
}

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _euclid(a: Point, b: Point) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _bbox_diag(points: list[Point]) -> float:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5)


def _centroid(points: list[Point]) -> Point:
    n = max(len(points), 1)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def _match_predictions_to_gt(
    pred_centroids: list[Point], gt_centroids: list[Point], *, max_dist: float
) -> list[tuple[int, int]]:
    """Optimal 1-1 assignment of predicted players to GT players by centroid.

    Returns ``(pred_idx, gt_idx)`` pairs whose centroid distance is below
    ``max_dist`` (drops cross-court pairings when counts differ).
    """
    if not pred_centroids or not gt_centroids:
        return []
    cost = [[_euclid(pc, gc) for gc in gt_centroids] for pc in pred_centroids]
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r][c] <= max_dist]


# ---------------------------------------------------------------------------
# Wrist PCK scoring (side-agnostic + strict)
# ---------------------------------------------------------------------------


def _wrist_distances(pred: Pose2D, gt_kpts: list[Point]) -> tuple[list[float], list[float]]:
    """Return ``(sided, agnostic)`` wrist-distance pairs for one matched player.

    ``sided`` pairs GT-R↔pred-R and GT-L↔pred-L. ``agnostic`` picks the
    cheaper of the two 2x2 wrist assignments (R/L-label invariant).
    """
    pr = (pred.keypoints[_HALPE_WRIST_RL[0]].x, pred.keypoints[_HALPE_WRIST_RL[0]].y)
    pl = (pred.keypoints[_HALPE_WRIST_RL[1]].x, pred.keypoints[_HALPE_WRIST_RL[1]].y)
    gr = gt_kpts[_GT_WRIST_RL[0]]
    gl = gt_kpts[_GT_WRIST_RL[1]]
    sided = [_euclid(gr, pr), _euclid(gl, pl)]
    opt_a = _euclid(gr, pr) + _euclid(gl, pl)
    opt_b = _euclid(gr, pl) + _euclid(gl, pr)
    agnostic = sided if opt_a <= opt_b else [_euclid(gr, pl), _euclid(gl, pr)]
    return sided, agnostic


def _group_distances(
    pred: Pose2D, gt_kpts: list[Point], group: list[tuple[int, int]]
) -> list[float]:
    """Per-keypoint distances for one landmark group (used by the diagnostic).

    A 2-entry L/R group is scored side-agnostically (cheaper of the 2x2
    assignment); a 1-entry central group compares directly.
    """
    g = [gt_kpts[gi] for gi, _ in group]
    p = [(pred.keypoints[hi].x, pred.keypoints[hi].y) for _, hi in group]
    if len(group) == 1:
        return [_euclid(g[0], p[0])]
    opt_a = _euclid(g[0], p[0]) + _euclid(g[1], p[1])
    opt_b = _euclid(g[0], p[1]) + _euclid(g[1], p[0])
    if opt_a <= opt_b:
        return [_euclid(g[0], p[0]), _euclid(g[1], p[1])]
    return [_euclid(g[0], p[1]), _euclid(g[1], p[0])]


# ---------------------------------------------------------------------------
# TrackID3x3 — primary wrist-PCK gate
# ---------------------------------------------------------------------------


def _load_trackid3x3_pose_gt(path: Path) -> dict[int, dict[str, list[Point]]]:
    """Parse a TrackID3x3 pose JSON → ``{frame_idx: {jersey: [[x, y], ...x10]}}``.

    Frame keys are ``frame_<i>`` (0-based, matching our predictions).
    """
    raw = json.loads(Path(path).read_text())
    out: dict[int, dict[str, list[Point]]] = {}
    for frame_key, players in raw["annotations"].items():
        idx = int(frame_key.split("_", 1)[1])
        out[idx] = {
            colour: [(float(p[0]), float(p[1])) for p in kpts] for colour, kpts in players.items()
        }
    return out


def _pair_files(
    pred_dir: Path, gt_dir: Path, *, pred_prefix: str, gt_ext: str
) -> list[tuple[Path, Path, str]]:
    """Match ``{pred_prefix}<key>.json`` predictions to ``<key>{gt_ext}`` GT.

    Mirrors Stage 1's matcher: strip the prefix, try ``<key>`` then
    ``<key>`` minus its last ``_segment`` (so ``session_camera`` files match
    a clip-named GT).
    """
    pairs: list[tuple[Path, Path, str]] = []
    for pred in sorted(pred_dir.glob(f"{pred_prefix}*.json")):
        key = pred.stem.removeprefix(pred_prefix)
        candidates = [gt_dir / f"{key}{gt_ext}"]
        if "_" in key:
            candidates.append(gt_dir / f"{key.rsplit('_', 1)[0]}{gt_ext}")
        gt = next((c for c in candidates if c.exists()), None)
        if gt is None:
            _log.warning("no GT for %s (tried %s)", pred.name, [str(c) for c in candidates])
            continue
        pairs.append((pred, gt, gt.stem))
    return pairs


def _score_trackid3x3_clip(
    seq: Pose2DSequence, gt: dict[int, dict[str, list[Point]]], *, pck_threshold: float
) -> dict[str, Any]:
    """PCK for one clip: wrist gate + per-landmark + all-keypoint diagnostic.

    Returns per-clip rates plus private ``_group_correct``/``_group_total``/
    ``_wrist_sided_correct`` raw counts so the caller can micro-average across
    clips. ``wrist`` is scored exactly as the gate (side-agnostic 2x2); the
    other landmark groups are the diagnostic.
    """
    pred_by_frame: dict[int, list[Pose2D]] = defaultdict(list)
    for p in seq.poses:
        pred_by_frame[p.frame_idx].append(p)

    g_correct = {g: 0 for g in _GT_LANDMARK_GROUPS}
    g_total = {g: 0 for g in _GT_LANDMARK_GROUPS}
    wrist_sided_correct = 0
    matched_players = 0
    for frame_idx, gt_players in gt.items():
        preds = pred_by_frame.get(frame_idx)
        if not preds:
            continue
        gt_items = list(gt_players.items())  # [(colour, kpts)]
        gt_centroids = [_centroid(kpts) for _, kpts in gt_items]
        pred_centroids = [_centroid([(k.x, k.y) for k in p.keypoints[:17]]) for p in preds]
        # Drop cross-court mismatches: cap match distance at half the spread of
        # the GT players across the court. True matches (a prediction near its
        # own player) sit far below this; a pairing to a player on the far side
        # exceeds it. With a single GT player there's nothing to confuse, so
        # accept any match (inf).
        court_spread = _bbox_diag(gt_centroids) if len(gt_centroids) > 1 else 0.0
        max_dist = 0.5 * court_spread if court_spread > 0 else float("inf")
        for pi, gi in _match_predictions_to_gt(pred_centroids, gt_centroids, max_dist=max_dist):
            gt_kpts = gt_items[gi][1]
            diag = _bbox_diag(gt_kpts)
            if diag <= 0:
                continue
            thr = pck_threshold * diag
            for gname, group in _GT_LANDMARK_GROUPS.items():
                if gname == "wrist":
                    sided, agnostic = _wrist_distances(preds[pi], gt_kpts)
                    g_correct["wrist"] += sum(d <= thr for d in agnostic)
                    g_total["wrist"] += 2
                    wrist_sided_correct += sum(d <= thr for d in sided)
                else:
                    dists = _group_distances(preds[pi], gt_kpts, group)
                    g_correct[gname] += sum(d <= thr for d in dists)
                    g_total[gname] += len(dists)
            matched_players += 1

    wrist_total = g_total["wrist"]
    all_correct = sum(g_correct.values())
    all_total = sum(g_total.values())
    per_landmark = {
        g: (g_correct[g] / g_total[g] if g_total[g] else 0.0) for g in _GT_LANDMARK_GROUPS
    }
    return {
        "wrist_pck": per_landmark["wrist"],
        "wrist_pck_sided": (wrist_sided_correct / wrist_total) if wrist_total else 0.0,
        "all_keypoint_pck": (all_correct / all_total) if all_total else 0.0,
        "per_landmark": per_landmark,
        "n_wrist_keypoints": float(wrist_total),
        "n_matched_players": float(matched_players),
        "_group_correct": g_correct,
        "_group_total": g_total,
        "_wrist_sided_correct": wrist_sided_correct,
    }


def evaluate_trackid3x3(
    poses_dir: Path, gt_dir: Path, *, pck_threshold: float = 0.05
) -> dict[str, Any]:
    """Wrist PCK@``pck_threshold`` (the gate) + per-landmark / all-keypoint diagnostic.

    Returns ``{"summary": {...}, "per_sequence": {...}, "n_sequences": int}``.
    ``summary`` micro-averages over all matched keypoints (sum correct / sum
    total) so clips with more annotated frames weigh proportionally. The wrist
    group is the gate; the other landmark groups + ``all_keypoint_pck`` are a
    diagnostic to see whether wrists are uniquely weak or just tracking the
    overall (model-derived) GT disagreement (F18).
    """
    poses_dir = Path(poses_dir)
    gt_dir = Path(gt_dir)
    pairs = _pair_files(poses_dir, gt_dir, pred_prefix="pose2d_", gt_ext=".json")
    if not pairs:
        raise FileNotFoundError(f"no matchable (pred, gt) pairs under {poses_dir} vs {gt_dir}")

    per_sequence: dict[str, dict[str, Any]] = {}
    agg_correct = {g: 0 for g in _GT_LANDMARK_GROUPS}
    agg_total = {g: 0 for g in _GT_LANDMARK_GROUPS}
    agg_wrist_sided = 0
    for pred_path, gt_path, key in pairs:
        seq = read_json_as(Pose2DSequence, pred_path)
        gt = _load_trackid3x3_pose_gt(gt_path)
        s = _score_trackid3x3_clip(seq, gt, pck_threshold=pck_threshold)
        per_sequence[key] = {k: v for k, v in s.items() if not k.startswith("_")}
        for g in _GT_LANDMARK_GROUPS:
            agg_correct[g] += s["_group_correct"][g]
            agg_total[g] += s["_group_total"][g]
        agg_wrist_sided += s["_wrist_sided_correct"]

    wrist_total = agg_total["wrist"]
    all_correct = sum(agg_correct.values())
    all_total = sum(agg_total.values())
    summary = {
        "wrist_pck": (agg_correct["wrist"] / wrist_total) if wrist_total else 0.0,
        "wrist_pck_sided": (agg_wrist_sided / wrist_total) if wrist_total else 0.0,
        "all_keypoint_pck": (all_correct / all_total) if all_total else 0.0,
        "per_landmark": {
            g: (agg_correct[g] / agg_total[g] if agg_total[g] else 0.0) for g in _GT_LANDMARK_GROUPS
        },
        "n_wrist_keypoints": float(wrist_total),
        "pck_threshold": pck_threshold,
    }
    return {"summary": summary, "per_sequence": per_sequence, "n_sequences": len(pairs)}


# ---------------------------------------------------------------------------
# MultiSubjects — secondary COCO-17 corroboration (single player)
# ---------------------------------------------------------------------------

# BlazePose-33 index for each COCO-17 keypoint (in COCO order).
BLAZEPOSE33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
_COCO_WRIST = (9, 10)  # left, right (COCO order)
_COCO17_NAMES = (
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
)


def _video_resolution(path: Path) -> tuple[float, float] | None:
    """Return (W, H) in pixels from a video file via cv2; None if unreadable."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return (w, h) if w > 0 and h > 0 else None


def _load_blazepose_csv(path: Path) -> dict[int, list[Point | None]]:
    """Parse a MultiSubjects BlazePose CSV → ``{frame_idx: [COCO-17 (x, y) or None]}``.

    Columns: ``Frame, Keypoint_0_X, _0_Y, _0_Z, ... _32_*`` (1 + 33x3). Coords
    are MediaPipe-normalised [0, 1] (consistent with the normalised person-GT
    bboxes); rows with empty keypoint cells (no detection) are skipped.
    Frames are 1-based in the CSV; normalised to 0-based here.
    """
    import csv

    out: dict[int, list[Point | None]] = {}
    with Path(path).open() as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not row or not row[0].strip():
                continue
            vals = row[1:]
            if not any(v.strip() for v in vals):
                continue
            frame = int(float(row[0])) - 1
            coco: list[Point | None] = []
            for bp in BLAZEPOSE33_TO_COCO17:
                xs, ys = vals[bp * 3], vals[bp * 3 + 1]
                coco.append((float(xs), float(ys)) if xs.strip() and ys.strip() else None)
            out[frame] = coco
    return out


def evaluate_multisubjects(
    poses_dir: Path,
    gt_dir: Path,
    *,
    pck_threshold: float = 0.05,
    frame_w: float = 1.0,
    frame_h: float = 1.0,
    videos_dir: Path | None = None,
) -> dict[str, Any]:
    """Per-keypoint COCO-17 PCK@``pck_threshold`` vs MultiSubjects BlazePose-33 GT.

    Predicted Halpe-26 poses are subset to COCO-17; GT BlazePose-33 is subset to
    COCO-17 (`BLAZEPOSE33_TO_COCO17`). PCK is normalised by the GT COCO-17 bbox
    diagonal in pixel space.

    **Coordinate handling:** GT coords are MediaPipe-normalised [0, 1]; predicted
    coords are in pixel space. If ``videos_dir`` is provided, per-clip pixel
    resolution is read from each clip's ``<key>.mp4`` (via cv2); otherwise the
    global ``frame_w``/``frame_h`` are applied to every clip.

    **Multi-track matching:** Stage 1 may emit more than one predicted track per
    frame (e.g. a coach in the background); for each annotated frame we pick the
    predicted pose whose COCO-17 centroid is closest to the GT keypoint centroid,
    so a stray non-player track doesn't corrupt the per-keypoint comparison.

    Returns ``{"summary": {...}, "per_sequence": {...}, "n_sequences": int}``.
    ``summary`` micro-averages over all matched keypoints across clips; both
    summary and per_sequence include a ``per_keypoint`` breakdown keyed by COCO
    keypoint name.
    """
    from courtvision.stage2_pose.halpe_to_coco17 import halpe26_to_coco17

    poses_dir = Path(poses_dir)
    gt_dir = Path(gt_dir)
    videos_dir = Path(videos_dir) if videos_dir is not None else None
    pairs = _pair_files(poses_dir, gt_dir, pred_prefix="pose2d_", gt_ext=".csv")
    if not pairs:
        raise FileNotFoundError(f"no matchable (pred, gt) pairs under {poses_dir} vs {gt_dir}")

    per_sequence: dict[str, dict[str, Any]] = {}
    g_correct = [0] * 17
    g_total = [0] * 17

    for pred_path, gt_path, key in pairs:
        seq = read_json_as(Pose2DSequence, pred_path)
        gt = _load_blazepose_csv(gt_path)
        pred_by_frame: dict[int, list[Pose2D]] = defaultdict(list)
        for p in seq.poses:
            pred_by_frame[p.frame_idx].append(p)

        # Per-clip resolution.
        w, h = frame_w, frame_h
        if videos_dir is not None:
            vp = videos_dir / f"{key}.mp4"
            res = _video_resolution(vp) if vp.exists() else None
            if res is not None:
                w, h = res

        clip_correct = [0] * 17
        clip_total = [0] * 17
        n_matched_frames = 0
        for frame_idx, gt_coco in gt.items():
            preds = pred_by_frame.get(frame_idx)
            if not preds:
                continue
            gt_px: list[Point | None] = [
                (g[0] * w, g[1] * h) if g is not None else None for g in gt_coco
            ]
            valid = [(i, g) for i, g in enumerate(gt_px) if g is not None]
            if not valid:
                continue
            gt_centroid = _centroid([g for _, g in valid])

            pred_cocos = [halpe26_to_coco17(p) for p in preds]
            best = min(
                range(len(preds)),
                key=lambda i: _euclid(
                    _centroid([(k.x, k.y) for k in pred_cocos[i].keypoints]),
                    gt_centroid,
                ),
            )
            pred_coco = pred_cocos[best]
            n_matched_frames += 1

            diag = _bbox_diag([g for _, g in valid])
            if diag <= 0:
                continue
            thr = pck_threshold * diag
            for i, g in valid:
                d = _euclid(g, (pred_coco.keypoints[i].x, pred_coco.keypoints[i].y))
                ok = int(d <= thr)
                clip_correct[i] += ok
                clip_total[i] += 1
                g_correct[i] += ok
                g_total[i] += 1

        per_keypoint_clip = {
            _COCO17_NAMES[i]: (clip_correct[i] / clip_total[i] if clip_total[i] else 0.0)
            for i in range(17)
        }
        clip_n = sum(clip_total)
        clip_c = sum(clip_correct)
        w_c = clip_correct[_COCO_WRIST[0]] + clip_correct[_COCO_WRIST[1]]
        w_n = clip_total[_COCO_WRIST[0]] + clip_total[_COCO_WRIST[1]]
        per_sequence[key] = {
            "pck": (clip_c / clip_n) if clip_n else 0.0,
            "wrist_pck": (w_c / w_n) if w_n else 0.0,
            "per_keypoint": per_keypoint_clip,
            "n_keypoints": float(clip_n),
            "n_wrist_keypoints": float(w_n),
            "n_matched_frames": n_matched_frames,
            "resolution_px": [w, h],
        }

    summary_per_kp = {
        _COCO17_NAMES[i]: (g_correct[i] / g_total[i] if g_total[i] else 0.0) for i in range(17)
    }
    all_c = sum(g_correct)
    all_n = sum(g_total)
    w_c = g_correct[_COCO_WRIST[0]] + g_correct[_COCO_WRIST[1]]
    w_n = g_total[_COCO_WRIST[0]] + g_total[_COCO_WRIST[1]]
    summary = {
        "pck": (all_c / all_n) if all_n else 0.0,
        "wrist_pck": (w_c / w_n) if w_n else 0.0,
        "per_keypoint": summary_per_kp,
        "n_keypoints": all_n,
        "n_wrist_keypoints": w_n,
        "pck_threshold": pck_threshold,
    }
    return {"summary": summary, "per_sequence": per_sequence, "n_sequences": len(pairs)}


__all__ = [
    "BLAZEPOSE33_TO_COCO17",
    "evaluate_multisubjects",
    "evaluate_trackid3x3",
]
