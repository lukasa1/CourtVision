"""Stage 1 visualization — bbox + track-ID overlay onto a video.

CPU-only. Reads a video + a ``TrackSequence`` and writes a new video with
per-frame bounding boxes labelled by track id. Used by the Phase B
qualitative user-test step (PRD §10) to eyeball whether ids stick with
the same player across frames.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from courtvision.logging import get_logger
from courtvision.schemas import Track, TrackSequence

_log = get_logger(__name__)


def _palette(track_ids: list[int], seed: int = 0) -> dict[int, tuple[int, int, int]]:
    """Stable per-track-id BGR colour palette."""
    rng = np.random.default_rng(seed)
    out: dict[int, tuple[int, int, int]] = {}
    for tid in track_ids:
        r, g, b = (int(c) for c in rng.integers(60, 255, size=3))
        out[tid] = (r, g, b)
    return out


def make_overlay_video(
    video_path: Path,
    sequence: TrackSequence,
    output_path: Path,
    max_seconds: float | None = None,
) -> Path:
    """Render ``sequence``'s bboxes onto ``video_path`` and write ``output_path``.

    ``max_seconds`` caps the output video length (computed against the
    source clip's reported fps). ``None`` processes the whole clip.

    Returns the output path. Raises ``FileNotFoundError`` if the source
    video can't be opened.
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

    by_frame: dict[int, list[Track]] = defaultdict(list)
    for t in sequence.tracks:
        by_frame[t.frame_idx].append(t)

    palette = _palette(sorted({t.track_id for t in sequence.tracks}))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open writer: {output_path}")

    frame_idx = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            for t in by_frame.get(frame_idx, []):
                x, y, bw, bh = t.bbox
                x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
                colour = palette.get(t.track_id, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                label = f"id {t.track_id} ({t.confidence:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            frame_idx += 1
            written += 1
    finally:
        writer.release()
        cap.release()

    _log.info("wrote overlay %s (%d frames, %.1f s @ %.2f fps)", output_path, written, written / fps, fps)
    return output_path


__all__ = ["make_overlay_video"]
