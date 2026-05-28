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
    return [
        (int(r), int(c))
        for r, c in zip(rows, cols, strict=True)
        if cost[r][c] <= max_dist
    ]


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
            colour: [(float(p[0]), float(p[1])) for p in kpts]
            for colour, kpts in players.items()
        }
    return out


def _pair_files(pred_dir: Path, gt_dir: Path, *, pred_prefix: str, gt_ext: str) -> list[
    tuple[Path, Path, str]
]:
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
) -> dict[str, float]:
    """Wrist PCK for one clip. Returns correct/total counts + derived rates."""
    pred_by_frame: dict[int, list[Pose2D]] = defaultdict(list)
    for p in seq.poses:
        pred_by_frame[p.frame_idx].append(p)

    correct_agn = correct_sided = total = 0
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
        for pi, gi in _match_predictions_to_gt(
            pred_centroids, gt_centroids, max_dist=max_dist
        ):
            gt_kpts = gt_items[gi][1]
            diag = _bbox_diag(gt_kpts)
            if diag <= 0:
                continue
            thr = pck_threshold * diag
            sided, agnostic = _wrist_distances(preds[pi], gt_kpts)
            correct_sided += sum(d <= thr for d in sided)
            correct_agn += sum(d <= thr for d in agnostic)
            total += 2
            matched_players += 1

    return {
        "wrist_pck": (correct_agn / total) if total else 0.0,
        "wrist_pck_sided": (correct_sided / total) if total else 0.0,
        "n_wrist_keypoints": float(total),
        "n_matched_players": float(matched_players),
    }


def evaluate_trackid3x3(
    poses_dir: Path, gt_dir: Path, *, pck_threshold: float = 0.05
) -> dict[str, Any]:
    """Side-agnostic wrist PCK@``pck_threshold`` on TrackID3x3.

    Returns ``{"summary": {...}, "per_sequence": {...}, "n_sequences": int}``.
    ``summary`` micro-averages over all matched wrists (sum correct / sum
    total) so clips with more annotated frames weigh proportionally.
    """
    poses_dir = Path(poses_dir)
    gt_dir = Path(gt_dir)
    pairs = _pair_files(poses_dir, gt_dir, pred_prefix="pose2d_", gt_ext=".json")
    if not pairs:
        raise FileNotFoundError(f"no matchable (pred, gt) pairs under {poses_dir} vs {gt_dir}")

    per_sequence: dict[str, dict[str, float]] = {}
    tot_correct = tot_correct_sided = tot = 0.0
    for pred_path, gt_path, key in pairs:
        seq = read_json_as(Pose2DSequence, pred_path)
        gt = _load_trackid3x3_pose_gt(gt_path)
        s = _score_trackid3x3_clip(seq, gt, pck_threshold=pck_threshold)
        per_sequence[key] = s
        tot_correct += s["wrist_pck"] * s["n_wrist_keypoints"]
        tot_correct_sided += s["wrist_pck_sided"] * s["n_wrist_keypoints"]
        tot += s["n_wrist_keypoints"]

    summary = {
        "wrist_pck": (tot_correct / tot) if tot else 0.0,
        "wrist_pck_sided": (tot_correct_sided / tot) if tot else 0.0,
        "n_wrist_keypoints": tot,
        "pck_threshold": pck_threshold,
    }
    return {"summary": summary, "per_sequence": per_sequence, "n_sequences": len(pairs)}


# ---------------------------------------------------------------------------
# MultiSubjects — secondary COCO-17 corroboration (single player)
# ---------------------------------------------------------------------------

# BlazePose-33 index for each COCO-17 keypoint (in COCO order).
BLAZEPOSE33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
_COCO_WRIST = (9, 10)  # left, right (COCO order)


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
) -> dict[str, Any]:
    """COCO-17 PCK@``pck_threshold`` vs MultiSubjects BlazePose-33 GT.

    Single-player clips, so no cross-player matching. Predicted Halpe-26
    poses are subset to COCO-17; GT BlazePose is subset+normalised to COCO-17.
    GT coords are normalised [0, 1]; pass the clip ``frame_w``/``frame_h`` to
    bring GT into the prediction's pixel space (default 1.0 keeps GT
    normalised — only valid if predictions are also normalised). PCK is
    normalised by the GT COCO-17 bbox diagonal. **Coordinate scale is assumed,
    not yet confirmed against populated CSVs — see F18.**
    """
    from courtvision.stage2_pose.halpe_to_coco17 import halpe26_to_coco17

    poses_dir = Path(poses_dir)
    gt_dir = Path(gt_dir)
    pairs = _pair_files(poses_dir, gt_dir, pred_prefix="pose2d_", gt_ext=".csv")
    if not pairs:
        raise FileNotFoundError(f"no matchable (pred, gt) pairs under {poses_dir} vs {gt_dir}")

    per_sequence: dict[str, dict[str, float]] = {}
    tot_correct = tot = tot_w_correct = tot_w = 0.0
    for pred_path, gt_path, key in pairs:
        seq = read_json_as(Pose2DSequence, pred_path)
        gt = _load_blazepose_csv(gt_path)
        pred_by_frame = {p.frame_idx: p for p in seq.poses}

        c_all = n_all = c_w = n_w = 0
        for frame_idx, gt_coco in gt.items():
            pred = pred_by_frame.get(frame_idx)
            if pred is None:
                continue
            pred_coco = halpe26_to_coco17(pred)
            gt_px: list[Point | None] = [
                (g[0] * frame_w, g[1] * frame_h) if g is not None else None for g in gt_coco
            ]
            valid = [(i, g) for i, g in enumerate(gt_px) if g is not None]
            if not valid:
                continue
            diag = _bbox_diag([g for _, g in valid])
            if diag <= 0:
                continue
            thr = pck_threshold * diag
            for i, g in valid:
                d = _euclid(g, (pred_coco.keypoints[i].x, pred_coco.keypoints[i].y))
                ok = d <= thr
                c_all += ok
                n_all += 1
                if i in _COCO_WRIST:
                    c_w += ok
                    n_w += 1

        per_sequence[key] = {
            "pck": (c_all / n_all) if n_all else 0.0,
            "wrist_pck": (c_w / n_w) if n_w else 0.0,
            "n_keypoints": float(n_all),
            "n_wrist_keypoints": float(n_w),
        }
        tot_correct += c_all
        tot += n_all
        tot_w_correct += c_w
        tot_w += n_w

    summary = {
        "pck": (tot_correct / tot) if tot else 0.0,
        "wrist_pck": (tot_w_correct / tot_w) if tot_w else 0.0,
        "n_keypoints": tot,
        "n_wrist_keypoints": tot_w,
        "pck_threshold": pck_threshold,
    }
    return {"summary": summary, "per_sequence": per_sequence, "n_sequences": len(pairs)}


__all__ = [
    "BLAZEPOSE33_TO_COCO17",
    "evaluate_multisubjects",
    "evaluate_trackid3x3",
]
