"""Stage 1 eval — MOTA / IDF1 on TrackID3x3. PRD §9.1, §10 Phase B.

Pairs a directory of predicted ``tracks_{session}_{camera}.json`` files
(written by :func:`courtvision.stage1_detection.yolo_bytetrack.run_stage1`)
with a directory of MOT-format ground-truth files, and computes per-pair +
aggregate MOTA / IDF1 using `motmetrics`.

GT format assumed: one ``{session}_{camera}.txt`` per pred file, MOT-CSV
schema ``frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z``.
The frame index in GT files is 1-based per MOT convention; we normalize to
0-based to match our predictions. If a future TrackID3x3 release ships a
different format, add another adapter to ``_load_gt_frames`` rather than
rewriting the eval logic.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import motmetrics as mm
import numpy as np

from courtvision.io_utils import read_json_as
from courtvision.logging import get_logger
from courtvision.schemas import TrackSequence

_log = get_logger(__name__)
_DELIM = re.compile(r"[\s,]+")

# Per-frame record: track_id -> [x, y, w, h] (top-left).
FrameDets = dict[int, list[float]]
# All frames for a sequence: frame_idx -> FrameDets.
SequenceDets = dict[int, FrameDets]


def _pred_frames_from_sequence(seq: TrackSequence) -> SequenceDets:
    out: SequenceDets = defaultdict(dict)
    for t in seq.tracks:
        out[t.frame_idx][t.track_id] = list(t.bbox)
    return out


def _load_gt_frames(gt_path: Path) -> SequenceDets:
    """Load MOT-format GT into the same per-frame dict shape as predictions.

    Delimiter-agnostic: handles CSV (TrackID3x3 Indoor + Outdoor) and
    whitespace-separated (TrackID3x3 Drone) variants. Trailing fields
    beyond ``frame, id, x, y, w, h`` are ignored — Indoor has 10
    columns, Outdoor 9.
    """
    out: SequenceDets = defaultdict(dict)
    with gt_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = _DELIM.split(line)
            frame_1b = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            # MOT convention is 1-based; our run_stage1 emits 0-based.
            out[frame_1b - 1][tid] = [x, y, w, h]
    return out


def _accumulate_one(pred: SequenceDets, gt: SequenceDets) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(pred) | set(gt))
    for f in frames:
        gt_f = gt.get(f, {})
        pr_f = pred.get(f, {})
        gt_ids = list(gt_f.keys())
        pr_ids = list(pr_f.keys())
        gt_boxes = np.array([gt_f[i] for i in gt_ids], dtype=float).reshape(-1, 4)
        pr_boxes = np.array([pr_f[i] for i in pr_ids], dtype=float).reshape(-1, 4)
        # iou_matrix returns distance = 1 - IoU; max_iou=0.5 (standard MOT).
        dist = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
        acc.update(gt_ids, pr_ids, dist)
    return acc


def _pair_files(tracks_dir: Path, gt_dir: Path) -> list[tuple[Path, Path, str]]:
    """Return list of ``(pred_json, gt_txt, key)`` for matchable sequences.

    Strips the ``tracks_`` prefix from the pred filename, then tries:
    (1) ``{stem}.txt``; (2) ``{stem-without-last-_segment}.txt``. The
    fallback lets a user name a clip ``session_id=basket_S2T1_pre`` with
    ``camera_id=cam0`` and still match a GT file ``basket_S2T1_pre.txt``.
    """
    pairs: list[tuple[Path, Path, str]] = []
    for pred in sorted(tracks_dir.glob("tracks_*.json")):
        key = pred.stem.removeprefix("tracks_")
        candidates = [gt_dir / f"{key}.txt"]
        if "_" in key:
            candidates.append(gt_dir / f"{key.rsplit('_', 1)[0]}.txt")
        gt = next((c for c in candidates if c.exists()), None)
        if gt is None:
            _log.warning(
                "no GT file for %s (tried %s)", pred.name, [str(c) for c in candidates]
            )
            continue
        pairs.append((pred, gt, gt.stem))
    return pairs


def evaluate_trackid3x3(tracks_dir: Path, gt_dir: Path) -> dict[str, Any]:
    """Compute MOTA / IDF1 against TrackID3x3 ground truth.

    Returns ``{"summary": {...}, "per_sequence": {...}, "n_sequences": int}``
    where ``summary`` is the motmetrics overall row.
    """
    tracks_dir = Path(tracks_dir)
    gt_dir = Path(gt_dir)
    pairs = _pair_files(tracks_dir, gt_dir)
    if not pairs:
        raise FileNotFoundError(
            f"no matchable (pred, gt) pairs under {tracks_dir} vs {gt_dir}"
        )

    accs: list[mm.MOTAccumulator] = []
    names: list[str] = []
    for pred_path, gt_path, key in pairs:
        seq = read_json_as(TrackSequence, pred_path)
        pred = _pred_frames_from_sequence(seq)
        gt = _load_gt_frames(gt_path)
        accs.append(_accumulate_one(pred, gt))
        names.append(key)

    metrics_host = mm.metrics.create()
    wanted = ["mota", "idf1", "idp", "idr", "num_frames", "num_unique_objects"]
    summary = metrics_host.compute_many(accs, names=names, metrics=wanted, generate_overall=True)
    # `summary` is a pandas DataFrame; convert to plain dicts for JSON serialisation.
    per_sequence = {row: summary.loc[row].to_dict() for row in summary.index if row != "OVERALL"}
    overall = summary.loc["OVERALL"].to_dict()
    return {
        "summary": overall,
        "per_sequence": per_sequence,
        "n_sequences": len(pairs),
    }


__all__ = ["evaluate_trackid3x3"]
