"""Track post-processing — persistence + non-player filtering.

Pure functions over a :class:`courtvision.schemas.TrackSequence`. No
Modal, no torch — runs + unit-tests on the laptop. Three layers
(PRD-flagged backlog F15):

- **B1 stitch** + **C1 distance gate** (``stitch_tracks``): rejoin a track
  that ended at frame F with one that started a few frames later nearby,
  so a brief occlusion doesn't spawn a fresh id.
- **A1 non-player filter** (``filter_non_players``): drop low-mobility
  tracks (refs / coaches stand still; players roam), with a floor so we
  never prune below the known player count and a ceiling-free policy so
  genuine extra players survive.

``postprocess`` chains them (stitch first, then filter).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from courtvision.logging import get_logger
from courtvision.schemas import Track, TrackSequence

_log = get_logger(__name__)


def _centroid(bbox: list[float]) -> tuple[float, float]:
    """Centroid of a top-left ``[x, y, w, h]`` box."""
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


# ---------------------------------------------------------------------------
# Per-track features
# ---------------------------------------------------------------------------


@dataclass
class TrackFeatures:
    track_id: int
    n_frames: int
    first_frame: int
    last_frame: int
    first_centroid: tuple[float, float]
    last_centroid: tuple[float, float]
    spatial_extent: float  # fraction of frame area the centroid roamed
    mean_speed: float  # px/frame, path length / (n_frames-1)
    mean_bbox_area: float  # px^2


def compute_track_features(
    seq: TrackSequence, frame_w: float, frame_h: float
) -> dict[int, TrackFeatures]:
    """Compute per-track mobility features.

    ``frame_w``/``frame_h`` normalise ``spatial_extent``. Pass the source
    clip's pixel dimensions; if unknown, pass 1.0 to get raw px^2 extent.
    """
    by_id: dict[int, list[Track]] = defaultdict(list)
    for t in seq.tracks:
        by_id[t.track_id].append(t)

    feats: dict[int, TrackFeatures] = {}
    frame_area = max(frame_w * frame_h, 1.0)
    for tid, ts in by_id.items():
        ts = sorted(ts, key=lambda t: t.frame_idx)
        cxs = [_centroid(t.bbox)[0] for t in ts]
        cys = [_centroid(t.bbox)[1] for t in ts]
        extent_px = (max(cxs) - min(cxs)) * (max(cys) - min(cys))
        path = sum(
            ((cxs[i] - cxs[i - 1]) ** 2 + (cys[i] - cys[i - 1]) ** 2) ** 0.5
            for i in range(1, len(ts))
        )
        mean_speed = path / (len(ts) - 1) if len(ts) > 1 else 0.0
        mean_area = sum(t.bbox[2] * t.bbox[3] for t in ts) / len(ts)
        feats[tid] = TrackFeatures(
            track_id=tid,
            n_frames=len(ts),
            first_frame=ts[0].frame_idx,
            last_frame=ts[-1].frame_idx,
            first_centroid=(cxs[0], cys[0]),
            last_centroid=(cxs[-1], cys[-1]),
            spatial_extent=extent_px / frame_area,
            mean_speed=mean_speed,
            mean_bbox_area=mean_area,
        )
    return feats


# ---------------------------------------------------------------------------
# B1 + C1 — stitch fragmented tracks
# ---------------------------------------------------------------------------


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def stitch_tracks(
    seq: TrackSequence,
    *,
    max_gap_frames: int = 10,
    base_radius_px: float = 50.0,
) -> TrackSequence:
    """Merge a track that ended into one that started shortly after, nearby.

    For each ended track (last frame F), consider tracks whose first frame
    is in ``[F+1, F+max_gap_frames]``. The spatial gate (C1) is
    ``dist(ended.last_centroid, candidate.first_centroid) <=
    base_radius_px * gap`` — a player can only have drifted so far in
    ``gap`` frames. Among passing candidates, pick the nearest; relabel
    the candidate's id to the ended track's id. Iterates so a chain of
    fragments collapses to one id.
    """
    # Work on a mutable copy of (track_id, bbox, conf, frame).
    tracks = [
        Track(track_id=t.track_id, bbox=list(t.bbox), confidence=t.confidence, frame_idx=t.frame_idx)
        for t in seq.tracks
    ]

    def _features() -> dict[int, TrackFeatures]:
        tmp = TrackSequence(
            session_id=seq.session_id,
            camera_id=seq.camera_id,
            fps=seq.fps,
            n_frames=seq.n_frames,
            tracks=tracks,
        )
        # frame dims irrelevant here (we only use centroids/frames).
        return compute_track_features(tmp, 1.0, 1.0)

    n_merges = 0
    changed = True
    while changed:
        changed = False
        feats = _features()
        # Sort ended tracks by their last frame so earlier ends stitch first.
        ended = sorted(feats.values(), key=lambda f: f.last_frame)
        used_targets: set[int] = set()
        for ef in ended:
            best_cand: int | None = None
            best_d = float("inf")
            for cf in feats.values():
                if cf.track_id == ef.track_id or cf.track_id in used_targets:
                    continue
                gap = cf.first_frame - ef.last_frame
                if gap < 1 or gap > max_gap_frames:
                    continue
                d = _dist(ef.last_centroid, cf.first_centroid)
                if d <= base_radius_px * gap and d < best_d:
                    best_d = d
                    best_cand = cf.track_id
            if best_cand is not None:
                # Relabel candidate id → ended id.
                for t in tracks:
                    if t.track_id == best_cand:
                        t.track_id = ef.track_id
                used_targets.add(ef.track_id)
                n_merges += 1
                changed = True
                break  # recompute features after each merge (ids shifted)

    if n_merges:
        _log.info("stitch: merged %d fragment(s)", n_merges)
    return TrackSequence(
        session_id=seq.session_id,
        camera_id=seq.camera_id,
        fps=seq.fps,
        n_frames=seq.n_frames,
        tracks=tracks,
    )


# ---------------------------------------------------------------------------
# A1 — non-player filter
# ---------------------------------------------------------------------------


def filter_non_players(
    seq: TrackSequence,
    *,
    frame_w: float,
    frame_h: float,
    min_players: int = 6,
    speed_floor_frac: float = 0.00068,
) -> TrackSequence:
    """Drop stationary tracks (refs / coaches), keeping a floor of players.

    Discriminator is ``mean_speed`` normalised by the frame diagonal —
    resolution-independent, so the same threshold transfers across clip
    resolutions (TrackID3x3 720p → own footage 1080p). Refs stand still
    (speed ≈ 0); players move (speed well above the floor) even in short
    fragments — which is why speed beats ``spatial_extent`` here (a brief
    fragment of a real player has high speed but low extent).

    Default ``speed_floor_frac`` = 0.00068 ≈ 1 px/frame at 1280x720.
    Keeps every track at or above the floor; if that leaves fewer than
    ``min_players``, keeps the ``min_players`` fastest regardless. Never
    caps above the number of moving tracks — genuine extras survive.
    """
    feats = compute_track_features(seq, frame_w, frame_h)
    if not feats:
        return seq

    diag = (frame_w**2 + frame_h**2) ** 0.5 or 1.0
    speed_floor_px = speed_floor_frac * diag

    ranked = sorted(feats.values(), key=lambda f: f.mean_speed, reverse=True)
    above_floor = [f.track_id for f in ranked if f.mean_speed >= speed_floor_px]
    if len(above_floor) >= min_players:
        keep = set(above_floor)
    else:
        keep = {f.track_id for f in ranked[:min_players]}

    dropped = [f.track_id for f in ranked if f.track_id not in keep]
    if dropped:
        _log.info(
            "filter_non_players: kept %d, dropped %d stationary (ids=%s)",
            len(keep),
            len(dropped),
            dropped,
        )
    kept_tracks = [t for t in seq.tracks if t.track_id in keep]
    return TrackSequence(
        session_id=seq.session_id,
        camera_id=seq.camera_id,
        fps=seq.fps,
        n_frames=seq.n_frames,
        tracks=kept_tracks,
    )


# ---------------------------------------------------------------------------
# Chained entrypoint
# ---------------------------------------------------------------------------


def postprocess(
    seq: TrackSequence, config: dict[str, Any], *, frame_w: float, frame_h: float
) -> TrackSequence:
    """Apply stitch then filter per ``config`` (both optional).

    ``config`` shape::

        {
          "stitch": {"max_gap_frames": 10, "base_radius_px": 50},
          "filter_non_players": {"min_players": 6, "mobility_floor": 0.01},
        }

    A missing/empty sub-dict skips that step.
    """
    out = seq
    stitch_cfg = config.get("stitch")
    if stitch_cfg:
        out = stitch_tracks(
            out,
            max_gap_frames=int(stitch_cfg.get("max_gap_frames", 10)),
            base_radius_px=float(stitch_cfg.get("base_radius_px", 50.0)),
        )
    filt_cfg = config.get("filter_non_players")
    if filt_cfg:
        out = filter_non_players(
            out,
            frame_w=frame_w,
            frame_h=frame_h,
            min_players=int(filt_cfg.get("min_players", 6)),
            speed_floor_frac=float(filt_cfg.get("speed_floor_frac", 0.00068)),
        )
    return out


__all__ = [
    "TrackFeatures",
    "compute_track_features",
    "filter_non_players",
    "postprocess",
    "stitch_tracks",
]
