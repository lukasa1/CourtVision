"""Stage 2 visualization — Halpe-26 skeleton overlay onto a video.

CPU-only (no mmpose). Reads a video + a ``Pose2DSequence`` and writes a new
video with per-track skeletons drawn on each frame. Wrists (the headline
keypoint for the Stage 7 contest metric) are highlighted. Used by the Phase C
qualitative user-test step (PRD §10) to eyeball whether wrists/elbows track
smoothly through shot motion.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.logging import get_logger
from courtvision.schemas import Pose2D, Pose2DSequence

_log = get_logger(__name__)

# Halpe-26 bone connections (index pairs). First 17 indices are COCO-17;
# 17 head, 18 neck, 19 hip-centre, 20-25 toes/heels.
_HALPE26_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # face: nose-eyes-ears
    (0, 17),
    (17, 18),
    (18, 19),  # nose-head-neck-hip(centre)
    (18, 5),
    (18, 6),
    (5, 6),  # neck-shoulders + shoulder link
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms (L: 5-7-9, R: 6-8-10)
    (19, 11),
    (19, 12),
    (11, 12),  # hip-centre-hips + hip link
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs (L: 11-13-15, R: 12-14-16)
    (15, 24),
    (15, 20),
    (20, 22),  # left foot: ankle-heel, ankle-bigtoe, -smalltoe
    (16, 25),
    (16, 21),
    (21, 23),  # right foot
)
_WRIST_IDX = (9, 10)  # L, R — highlighted
_SCORE_THRESH = 0.3  # don't draw low-confidence keypoints/bones


def _palette(track_ids: list[int], seed: int = 1) -> dict[int, tuple[int, int, int]]:
    """Stable per-track-id BGR colour palette."""
    rng = np.random.default_rng(seed)
    out: dict[int, tuple[int, int, int]] = {}
    for tid in track_ids:
        out[tid] = tuple(int(c) for c in rng.integers(60, 255, size=3))  # type: ignore[assignment]
    return out


def _draw_pose(frame: np.ndarray[Any, Any], pose: Pose2D, colour: tuple[int, int, int]) -> None:
    kps = pose.keypoints
    pts = [(int(k.x), int(k.y), k.score) for k in kps]
    # Bones (only when both endpoints are confident).
    for a, b in _HALPE26_SKELETON:
        if (
            a < len(pts)
            and b < len(pts)
            and pts[a][2] >= _SCORE_THRESH
            and pts[b][2] >= _SCORE_THRESH
        ):
            cv2.line(frame, pts[a][:2], pts[b][:2], colour, 2, cv2.LINE_AA)
    # Keypoints.
    for i, (x, y, s) in enumerate(pts):
        if s < _SCORE_THRESH:
            continue
        if i in _WRIST_IDX:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)  # wrist: red, highlighted
            cv2.circle(frame, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 3, colour, -1, cv2.LINE_AA)
    # Track-id label near the nose / first confident keypoint.
    anchor = next(((x, y) for (x, y, s) in pts if s >= _SCORE_THRESH), None)
    if anchor is not None:
        cv2.putText(
            frame,
            f"id {pose.track_id}",
            (anchor[0], max(0, anchor[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA,
        )


def make_pose_overlay_video(
    video_path: Path,
    sequence: Pose2DSequence,
    output_path: Path,
    max_seconds: float | None = None,
) -> Path:
    """Render ``sequence``'s skeletons onto ``video_path`` → ``output_path``.

    Wrists are drawn as large red dots. ``max_seconds`` caps output length
    against the source fps. Returns the output path; raises if the video can't
    be opened.
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

    by_frame: dict[int, list[Pose2D]] = defaultdict(list)
    for p in sequence.poses:
        by_frame[p.frame_idx].append(p)
    palette = _palette(sorted({p.track_id for p in sequence.poses}))

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
            for pose in by_frame.get(frame_idx, []):
                _draw_pose(frame, pose, palette.get(pose.track_id, (0, 255, 0)))
            writer.write(frame)
            frame_idx += 1
            written += 1
    finally:
        writer.release()
        cap.release()

    _log.info("wrote pose overlay %s (%d frames @ %.2f fps)", output_path, written, fps)
    return output_path


__all__ = ["make_pose_overlay_video"]
