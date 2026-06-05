"""Stage 3 visualization — ball-detection overlay onto a video.

CPU-only. Reads a video + a ``Ball2DSequence`` and writes a new video with a
per-frame ball marker (circle + crosshair) plus a fading trajectory trail, so
you can eyeball whether the ball is tracked through dribble / shot / release
and where it drops out (occlusion behind the shooter, motion blur, etc.).

Mirrors ``stage1_detection/overlay.py`` for the Phase D qualitative user-test
step (PRD §10).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2

from courtvision.logging import get_logger
from courtvision.schemas import Ball2DSequence

_log = get_logger(__name__)

_BALL_BGR = (0, 165, 255)  # orange
_TRAIL_BGR = (0, 255, 255)  # yellow


def make_ball_overlay_video(
    video_path: Path,
    sequence: Ball2DSequence,
    output_path: Path,
    max_seconds: float | None = None,
    trail_len: int = 12,
) -> Path:
    """Render ``sequence``'s ball detections onto ``video_path``.

    ``max_seconds`` caps output length against the source fps; ``None``
    processes the whole clip. ``trail_len`` is how many recent detected
    positions to draw as a fading trail. Returns the output path; raises
    ``FileNotFoundError`` if the source can't be opened.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(sequence.fps) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps) if max_seconds is not None else None

    by_frame = {d.frame_idx: d for d in sequence.detections}

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open writer: {output_path}")

    trail: deque[tuple[int, int]] = deque(maxlen=trail_len)
    frame_idx = 0
    written = 0
    detected = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            det = by_frame.get(frame_idx)
            if det is not None:
                cx, cy = round(det.x), round(det.y)
                trail.append((cx, cy))
                detected += 1

            # Fading trail: older points smaller + dimmer.
            n = len(trail)
            for i, (tx, ty) in enumerate(trail):
                frac = (i + 1) / n
                cv2.circle(frame, (tx, ty), max(1, int(3 * frac)), _TRAIL_BGR, -1, cv2.LINE_AA)

            if det is not None:
                cx, cy = round(det.x), round(det.y)
                cv2.circle(frame, (cx, cy), 9, _BALL_BGR, 2, cv2.LINE_AA)
                cv2.drawMarker(frame, (cx, cy), _BALL_BGR, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"ball {det.score:.2f}",
                    (cx + 12, max(0, cy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    _BALL_BGR,
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_idx += 1
            written += 1
    finally:
        writer.release()
        cap.release()

    _log.info(
        "wrote ball overlay %s (%d frames, ball in %d, %.1f s @ %.2f fps)",
        output_path,
        written,
        detected,
        written / fps if fps else 0.0,
        fps,
    )
    return output_path


__all__ = ["make_ball_overlay_video"]
