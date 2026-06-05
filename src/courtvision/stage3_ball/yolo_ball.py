"""Stage 3 (alt) — appearance-based YOLOv8 ball detection. flags.md F19.

WASB (the PRD §5.2-locked detector) is motion-based and goes blind on
held/static balls in TrackID3x3 (see ``flags.md`` F19 diagnosis). This is an
appearance detector: it fires on a *visible* ball regardless of motion, which
is what we actually need. It emits the same :class:`Ball2DSequence` as
``wasb_inference.run_stage3`` so overlays and downstream consumers are
unchanged.

Works with either weight:
- **Stock** ``yolov8m.pt`` (COCO) with ``classes=[32]`` — the COCO "sports
  ball" category. Zero training; a fast off-the-shelf baseline.
- **Custom** single-class ``ball`` weight (``classes=[0]``) trained on the
  Roboflow basketball data.

Because the TrackID3x3 ball is tiny (~10-15 px in a wide elevated shot),
detection at the native frame size misses it. Two small-object levers:
- raise ``model.imgsz`` (the whole frame is upscaled before detection);
- enable **tiled** inference (``config['tile']``): each frame is split into
  overlapping tiles detected at ``imgsz`` and merged back — so the ball is
  effectively larger per tile. This is the SAHI-style fix for tiny objects.

Per frame we keep the single highest-confidence ball, optionally gated by
``max_disp`` (drop detections that jump implausibly far from the previous
ball — cheap false-positive rejection).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from courtvision.logging import get_logger
from courtvision.schemas import Ball2DSequence, BallDetection2D

_log = get_logger(__name__)

_DEFAULT_WEIGHTS = "/artifacts/checkpoints/yolov8m.pt"

# A candidate ball: full-frame centre x, y, and detector confidence.
_Cand = tuple[float, float, float]


def _to_numpy(x: Any) -> Any:
    """Coerce a torch tensor (or numpy array) off the GPU into numpy."""
    import numpy as np

    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _select(
    cands: list[_Cand], prev_xy: tuple[float, float] | None, max_disp: float | None
) -> _Cand | None:
    """Pick one ball per frame: highest-confidence candidate, preferring those
    within ``max_disp`` of the previous frame's ball; fall back to the global
    best (re-acquisition after a gap)."""
    if not cands:
        return None
    cands.sort(key=lambda c: c[2], reverse=True)
    if max_disp is not None and prev_xy is not None:
        for cx, cy, sc in cands:
            if math.hypot(cx - prev_xy[0], cy - prev_xy[1]) <= max_disp:
                return (cx, cy, sc)
    return cands[0]


def _tile_boxes(
    w: int, h: int, rows: int, cols: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Overlapping tile boxes ``(x0, y0, x1, y1)`` covering a w-by-h frame."""
    tw, th = w / cols, h / rows
    ox, oy = tw * overlap, th * overlap
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, round(c * tw - ox))
            y0 = max(0, round(r * th - oy))
            x1 = min(w, round((c + 1) * tw + ox))
            y1 = min(h, round((r + 1) * th + oy))
            boxes.append((x0, y0, x1, y1))
    return boxes


def _boxes_to_cands(boxes: Any, off_x: float = 0.0, off_y: float = 0.0) -> list[_Cand]:
    """Convert an ultralytics Boxes object to ``(cx, cy, score)`` candidates,
    shifted by a tile offset so coords are in the full frame."""
    if boxes is None or len(boxes) == 0:
        return []
    confs = _to_numpy(boxes.conf)
    xyxy = _to_numpy(boxes.xyxy)
    out: list[_Cand] = []
    for i in range(len(confs)):
        x1, y1, x2, y2 = xyxy[i]
        out.append(
            (float((x1 + x2) / 2.0 + off_x), float((y1 + y2) / 2.0 + off_y), float(confs[i]))
        )
    return out


def run_stage3_yolo(video_path: Path, config: dict[str, Any]) -> Ball2DSequence:
    """Per-camera ball detection via YOLOv8. Returns a :class:`Ball2DSequence`.

    ``config`` (see ``configs/stage3_ball/yolo_ball.yaml``):
    ``session_id``, ``camera_id``, ``model.{weights,classes,conf,iou,imgsz}``,
    optional ``max_disp`` (px motion gate), ``max_seconds``, and
    ``tile.{rows,cols,overlap}`` to enable tiled small-object inference.
    """
    import cv2
    from ultralytics import YOLO  # type: ignore[attr-defined]  # heavy import, deferred

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    session_id = str(config["session_id"])
    camera_id = str(config["camera_id"])
    model_cfg = config.get("model", {})
    weights = model_cfg.get("weights") or _DEFAULT_WEIGHTS
    classes = list(model_cfg.get("classes", [32]))  # 32 = COCO "sports ball"
    conf = float(model_cfg.get("conf", 0.25))
    iou = float(model_cfg.get("iou", 0.5))
    imgsz = int(model_cfg.get("imgsz", 1280))
    max_disp = config.get("max_disp")
    max_disp = float(max_disp) if max_disp is not None else None
    tile_cfg = config.get("tile")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(config["max_seconds"] * fps) if config.get("max_seconds") and fps > 0 else None

    _log.info(
        "stage3-yolo: %s cam=%s fps=%.2f n_frames=%d weights=%s classes=%s imgsz=%d tile=%s",
        session_id,
        camera_id,
        fps,
        n_frames,
        weights,
        classes,
        imgsz,
        tile_cfg,
    )

    model = YOLO(weights)
    predict_kw: dict[str, Any] = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "classes": classes,
        "verbose": False,
    }

    detections: list[BallDetection2D] = []
    prev_xy: tuple[float, float] | None = None

    if tile_cfg:
        rows = int(tile_cfg.get("rows", 2))
        cols = int(tile_cfg.get("cols", 2))
        overlap = float(tile_cfg.get("overlap", 0.2))
        tboxes: list[tuple[int, int, int, int]] | None = None
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or (max_frames is not None and frame_idx >= max_frames):
                    break
                h, w = frame.shape[:2]
                if tboxes is None:
                    tboxes = _tile_boxes(w, h, rows, cols, overlap)
                crops = [frame[y0:y1, x0:x1] for (x0, y0, x1, y1) in tboxes]
                results = model.predict(crops, **predict_kw)
                cands: list[_Cand] = []
                for (x0, y0, _, _), res in zip(tboxes, results, strict=False):
                    cands.extend(_boxes_to_cands(res.boxes, off_x=x0, off_y=y0))
                sel = _select(cands, prev_xy, max_disp)
                if sel is not None:
                    detections.append(
                        BallDetection2D(frame_idx=frame_idx, x=sel[0], y=sel[1], score=sel[2])
                    )
                    prev_xy = (sel[0], sel[1])
                frame_idx += 1
        finally:
            cap.release()
        observed = frame_idx
    else:
        cap.release()
        observed = 0
        for frame_idx, result in enumerate(
            model.predict(source=str(video_path), stream=True, **predict_kw)
        ):
            observed = frame_idx + 1
            if max_frames is not None and frame_idx >= max_frames:
                break
            sel = _select(_boxes_to_cands(result.boxes), prev_xy, max_disp)
            if sel is not None:
                detections.append(
                    BallDetection2D(frame_idx=frame_idx, x=sel[0], y=sel[1], score=sel[2])
                )
                prev_xy = (sel[0], sel[1])

    effective_n = observed if n_frames <= 0 else n_frames
    _log.info(
        "stage3-yolo: %s cam=%s detected ball in %d/%d frames",
        session_id,
        camera_id,
        len(detections),
        effective_n,
    )
    return Ball2DSequence(
        session_id=session_id,
        camera_id=camera_id,
        fps=fps if fps > 0 else 0.0,
        detections=detections,
    )


__all__ = ["run_stage3_yolo"]
