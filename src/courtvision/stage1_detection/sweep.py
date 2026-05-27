"""ByteTrack coordinate-descent sweep — pure logic.

This module is import-safe (no Modal, no plotting). The Modal-driving
parts live in ``scripts/stage1_sweep.py``; the plotting in
``stage1_detection/report.py``. Keeping this module pure lets the unit
tests run on the laptop without Modal credentials.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Config manipulation
# ---------------------------------------------------------------------------


def set_by_path(cfg: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Return a deep copy of ``cfg`` with the dotted ``path`` set to ``value``.

    Creates intermediate dicts as needed (so a path like
    ``tracker.overrides.track_buffer`` works on a baseline whose
    ``tracker.overrides`` is empty or missing).
    """
    out = copy.deepcopy(cfg)
    parts = path.split(".")
    cursor: dict[str, Any] = out
    for p in parts[:-1]:
        nxt = cursor.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[p] = nxt
        cursor = nxt
    cursor[parts[-1]] = value
    return out


def build_run_config(
    base: dict[str, Any],
    *,
    session_id: str,
    camera_id: str,
    output_relpath: str,
) -> dict[str, Any]:
    """Add per-clip identity + output to a sweep config dict."""
    out = copy.deepcopy(base)
    out["session_id"] = session_id
    out["camera_id"] = camera_id
    out["output_relpath"] = output_relpath
    return out


def clip_name_to_session_camera(clip_name: str) -> tuple[str, str]:
    """Split ``"basket_S1T1_pre"`` → ``("basket_S1T1", "pre")``.

    Last underscore-separated segment is the camera id; everything
    before it is the session id. Matches TrackID3x3's MOT GT file
    naming so ``_pair_files`` finds the right GT.
    """
    if "_" not in clip_name:
        raise ValueError(f"clip name {clip_name!r} has no _segment to split")
    head, _, tail = clip_name.rpartition("_")
    return head, tail


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """One sweep run's aggregate metrics across N dev clips."""

    tag: str
    knob: str | None
    value: Any | None
    n_clips: int
    idf1_per_clip: list[float] = field(default_factory=list)
    mota_per_clip: list[float] = field(default_factory=list)

    @property
    def idf1_mean(self) -> float:
        return _mean(self.idf1_per_clip)

    @property
    def idf1_std(self) -> float:
        return _std(self.idf1_per_clip)

    @property
    def idf1_ci95(self) -> float:
        """95% CI half-width = 1.96 * sigma / sqrt(n)."""
        n = len(self.idf1_per_clip)
        if n <= 1:
            return float("nan")
        return 1.96 * self.idf1_std / math.sqrt(n)

    @property
    def mota_mean(self) -> float:
        return _mean(self.mota_per_clip)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "knob": self.knob,
            "value": self.value,
            "n_clips": self.n_clips,
            "idf1_mean": self.idf1_mean,
            "idf1_std": self.idf1_std,
            "idf1_ci95": self.idf1_ci95,
            "mota_mean": self.mota_mean,
            "idf1_per_clip": list(self.idf1_per_clip),
            "mota_per_clip": list(self.mota_per_clip),
        }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def summarise_eval_result(
    *, tag: str, knob: str | None, value: Any | None, eval_result: dict[str, Any]
) -> RunSummary:
    """Turn ``evaluate_trackid3x3``'s ``per_sequence`` dict into a RunSummary."""
    per_seq = eval_result.get("per_sequence", {})
    idf1s = [float(m["idf1"]) for m in per_seq.values() if m.get("idf1") is not None]
    motas = [float(m["mota"]) for m in per_seq.values() if m.get("mota") is not None]
    return RunSummary(
        tag=tag,
        knob=knob,
        value=value,
        n_clips=len(per_seq),
        idf1_per_clip=idf1s,
        mota_per_clip=motas,
    )


def pick_best(summaries: list[RunSummary]) -> RunSummary:
    """Return the summary with the highest IDF1 mean (breaks ties by smaller std)."""
    if not summaries:
        raise ValueError("no summaries to pick from")
    return min(summaries, key=lambda s: (-s.idf1_mean, s.idf1_std))


__all__ = [
    "RunSummary",
    "build_run_config",
    "clip_name_to_session_camera",
    "pick_best",
    "set_by_path",
    "summarise_eval_result",
]
