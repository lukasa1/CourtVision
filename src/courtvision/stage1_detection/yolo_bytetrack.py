"""Stage 1 — YOLOv8m + ByteTrack inference. PRD §9.1.

Off-the-shelf player detection + tracking. Public entrypoint ``run_stage1``
returns a :class:`courtvision.schemas.TrackSequence`. Locked tracker is
ByteTrack via ultralytics' built-in ``bytetrack.yaml`` config; locked
detector knobs are ``classes=[0]`` (person), ``conf=0.4``, ``iou=0.5``.

Weights resolution: if ``config['model']['weights']`` points to a real file
on disk, we use it (this is the Modal path,
``/artifacts/checkpoints/yolov8m.pt``). Otherwise we fall back to passing
the bare name (``yolov8m.pt``) to ``ultralytics.YOLO``, which auto-downloads
into ``~/.cache/ultralytics``. That keeps local dev unblocked when the
Modal Volume isn't mounted.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.logging import get_logger
from courtvision.schemas import Track, TrackSequence

_log = get_logger(__name__)


def _to_numpy(x: Any) -> np.ndarray[Any, Any]:
    """Coerce a torch tensor *or* numpy array off the GPU into a numpy array."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _resolve_weights(cfg_weights: str | None) -> str:
    """Return a weights spec the ``YOLO`` constructor accepts.

    Uses the configured path when it exists on disk; otherwise returns the
    bare filename so ultralytics will auto-download.
    """
    if cfg_weights and Path(cfg_weights).exists():
        return cfg_weights
    return "yolov8m.pt"


def _resolve_tracker_cfg(tracker_cfg: dict[str, Any]) -> str:
    """Return a tracker spec for ``model.track(tracker=...)``.

    With no ``overrides`` key, returns the configured cfg name (e.g.
    ``"bytetrack.yaml"``) so ultralytics uses the bundled defaults.
    With overrides, loads the bundled bytetrack.yaml, merges the
    overrides, and writes the result to a temp file whose path is
    returned. Lets the sweep harness tweak ``track_buffer``,
    ``match_thresh``, etc. without forking the ultralytics YAML.
    """
    overrides = tracker_cfg.get("overrides") or {}
    cfg_name: str = tracker_cfg.get("cfg", "bytetrack.yaml")
    if not overrides:
        return cfg_name

    import ultralytics
    import yaml  # heavy imports deferred

    default_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / cfg_name
    if not default_path.exists():
        raise FileNotFoundError(
            f"cannot find ultralytics tracker config {cfg_name} at {default_path}"
        )
    with default_path.open() as f:
        merged = dict(yaml.safe_load(f))
    merged.update(overrides)

    tmp_dir = Path(tempfile.mkdtemp(prefix="cv_tracker_"))
    tmp_path = tmp_dir / "bytetrack_custom.yaml"
    with tmp_path.open("w") as f:
        yaml.safe_dump(merged, f)
    _log.info("tracker overrides=%s → %s", overrides, tmp_path)
    return str(tmp_path)


def _video_meta(video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return fps, n_frames, width, height


def run_stage1(video_path: Path, config: dict[str, Any]) -> TrackSequence:
    """Per-camera player tracking. Returns a :class:`TrackSequence`.

    Expects ``config`` shaped like
    ``configs/stage1_detection/yolov8m_bytetrack.yaml``: ``session_id``,
    ``camera_id``, ``model.{weights,classes,conf,iou,imgsz}``, ``tracker.cfg``.
    """
    from ultralytics import YOLO  # type: ignore[attr-defined]  # heavy import, deferred

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    session_id = str(config["session_id"])
    camera_id = str(config["camera_id"])
    model_cfg = config["model"]
    tracker_cfg = config.get("tracker", {})

    fps, n_frames, width, height = _video_meta(video_path)
    _log.info(
        "stage1: %s cam=%s fps=%.2f n_frames=%d %dx%d",
        session_id,
        camera_id,
        fps,
        n_frames,
        width,
        height,
    )

    weights = _resolve_weights(model_cfg.get("weights"))
    model = YOLO(weights)

    tracks: list[Track] = []
    # ``stream=True`` yields one Results per frame, so we can stream rather
    # than buffering an entire clip in memory.
    results_iter = model.track(
        source=str(video_path),
        tracker=_resolve_tracker_cfg(tracker_cfg),
        classes=list(model_cfg.get("classes", [0])),
        conf=float(model_cfg.get("conf", 0.4)),
        iou=float(model_cfg.get("iou", 0.5)),
        imgsz=int(model_cfg.get("imgsz", 1280)),
        persist=True,
        stream=True,
        verbose=False,
    )

    observed_frames = 0
    for frame_idx, result in enumerate(results_iter):
        observed_frames = frame_idx + 1
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            continue
        # xyxy: [N, 4] top-left + bottom-right in pixels. Convert to
        # top-left xywh — MOT convention, matches motmetrics expectations.
        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        ids = _to_numpy(boxes.id).astype(int)
        for i in range(len(ids)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            tracks.append(
                Track(
                    track_id=int(ids[i]),
                    bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    confidence=float(confs[i]),
                    frame_idx=int(frame_idx),
                )
            )

    # Prefer the stream's observed frame count when the container header
    # disagreed with reality (some iPhone HEVCs report 0 for n_frames).
    effective_n_frames = observed_frames if n_frames <= 0 else n_frames
    seq = TrackSequence(
        session_id=session_id,
        camera_id=camera_id,
        fps=fps if fps > 0 else 0.0,
        n_frames=effective_n_frames,
        tracks=tracks,
    )

    # Optional post-processing: stitch fragmented tracks + drop stationary
    # non-players (refs/coaches). Config-gated; no-op when absent.
    pp_cfg = config.get("postprocess")
    if pp_cfg:
        from courtvision.stage1_detection.postprocess import postprocess

        seq = postprocess(seq, pp_cfg, frame_w=float(width), frame_h=float(height))
    return seq


__all__ = ["run_stage1"]
