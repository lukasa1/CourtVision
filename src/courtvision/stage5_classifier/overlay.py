"""Stage 5 qualitative overlay — shot classifier on a clip. PRD §9.5.3 / E5.

Draws, per frame: the COCO-17 pose skeleton the classifier reads, a live
class-probability readout (dribble / layup / shoot bars), the predicted top
class, the ground-truth label (if known), and a flash when a shot event fires.
Lets you *watch* whether the classifier's probabilities track the action.

CPU/cv2 only (no torch/mmaction), so it imports anywhere; the per-window
probabilities come from ``infer.classify_windows``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from courtvision.logging import get_logger

_log = get_logger(__name__)

# COCO-17 skeleton connectivity.
_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (11, 12),
]
_BAR_BGR = {"dribble": (255, 160, 0), "layup": (0, 200, 255), "shoot": (40, 40, 255)}


def make_shot_overlay_video(
    video_path: Path,
    keypoint: Any,  # [T, 17, 2] pixel coords
    keypoint_score: Any,  # [T, 17]
    centers: list[int],
    probs: list[Any],  # per-window softmax over class_order
    output_path: Path,
    *,
    class_order: tuple[str, ...] = ("dribble", "layup", "shoot", "background"),
    true_label: int | None = None,
    event_frames: list[int] | None = None,
    kp_thresh: float = 0.2,
) -> Path:
    """Render the skeleton + class-probability readout onto ``video_path``."""
    import numpy as np

    video_path, output_path = Path(video_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # type: ignore[attr-defined]

    centers_arr = np.asarray(centers)
    event_frames = event_frames or []
    fs = max(0.6, w / 900.0)  # font scale relative to width
    bar_w = int(w * 0.34)
    x0 = w - bar_w - int(w * 0.03)

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # skeleton
        if f < len(keypoint):
            kp, ks = keypoint[f], keypoint_score[f]
            for a, b in _EDGES:
                if ks[a] > kp_thresh and ks[b] > kp_thresh:
                    cv2.line(
                        frame,
                        (int(kp[a][0]), int(kp[a][1])),
                        (int(kp[b][0]), int(kp[b][1])),
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            for j in range(len(kp)):
                if ks[j] > kp_thresh:
                    cv2.circle(
                        frame, (int(kp[j][0]), int(kp[j][1])), 4, (0, 200, 255), -1, cv2.LINE_AA
                    )

        # nearest-window probabilities for this frame
        p = probs[int(np.argmin(np.abs(centers_arr - f)))] if len(centers) else None
        if p is not None:
            y = int(h * 0.04)
            row_h = int(h * 0.035)
            for k, cls in enumerate(class_order[:3]):  # dribble/layup/shoot bars
                yy = y + k * (row_h + 8)
                cv2.rectangle(
                    frame, (x0, yy), (x0 + int(bar_w * float(p[k])), yy + row_h), _BAR_BGR[cls], -1
                )
                cv2.rectangle(frame, (x0, yy), (x0 + bar_w, yy + row_h), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    f"{cls} {float(p[k]):.2f}",
                    (x0 + 4, yy + row_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            top = int(np.argmax(p))
            cv2.putText(
                frame,
                f"PRED: {class_order[top]} {float(p[top]):.2f}",
                (x0, y + 3 * (row_h + 8) + int(row_h * 0.8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs * 0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if true_label is not None:
                cv2.putText(
                    frame,
                    f"TRUE: {class_order[true_label]}",
                    (x0, y + 3 * (row_h + 8) + int(row_h * 1.8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.6,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

        if any(abs(f - ef) <= 4 for ef in event_frames):
            cv2.putText(
                frame,
                "SHOT EVENT",
                (int(w * 0.03), int(h * 0.06)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                (40, 40, 255),
                3,
                cv2.LINE_AA,
            )

        writer.write(frame)
        f += 1
    writer.release()
    cap.release()
    _log.info("wrote shot overlay %s (%d frames)", output_path, f)
    return output_path


_ACTION_BGR = {
    "shoot": (0, 0, 255),
    "layup": (0, 165, 255),
    "dribble": (210, 210, 210),
    "background": (130, 130, 130),
}


def make_action_overlay_video(
    video_path: Path,
    sequence: Any,  # Pose2DSequence (halpe26), multi-track
    track_actions: dict[int, dict[int, tuple[str, float]]],  # track_id -> {frame_idx -> (action, conf)}
    output_path: Path,
    *,
    track_events: dict[int, list[tuple[int, float]]] | None = None,  # track_id -> [(peak_frame, conf)]
    max_seconds: float | None = None,
    shoot_thresh: float = 0.6,
    flash_radius: int = 12,
    score_thresh: float = 0.3,
) -> Path:
    """Render every tracked player's Halpe-26 skeleton + a rolling action tag.

    ``track_actions`` is the per-frame ``(action, confidence)`` rolling label
    (always shown, muted). The bright "SHOOT" flash is decoupled: it fires only
    within ``flash_radius`` frames of a detected shot-event peak (``track_events``,
    from the consolidated ``run_shot_inference``) whose conf >= ``shoot_thresh``,
    so the indicator is a tight burst at the shot rather than a smear.
    """
    from collections import defaultdict

    from courtvision.stage2_pose.overlay import _HALPE26_SKELETON, _palette

    video_path, output_path = Path(video_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps) if max_seconds else None
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # type: ignore[attr-defined]

    by_frame: dict[int, list[Any]] = defaultdict(list)
    for p in sequence.poses:
        by_frame[p.frame_idx].append(p)
    palette = _palette(sorted({p.track_id for p in sequence.poses}))
    fs = max(0.5, w / 1600.0)

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and f >= max_frames):
            break
        for pose in by_frame.get(f, []):
            tid = pose.track_id
            col = palette.get(tid, (0, 255, 0))
            pts = [(int(k.x), int(k.y), k.score) for k in pose.keypoints]
            for a, b in _HALPE26_SKELETON:
                if (
                    a < len(pts)
                    and b < len(pts)
                    and pts[a][2] >= score_thresh
                    and pts[b][2] >= score_thresh
                ):
                    cv2.line(frame, pts[a][:2], pts[b][:2], col, 2, cv2.LINE_AA)
            for x, y, s in pts:
                if s >= score_thresh:
                    cv2.circle(frame, (x, y), 2, col, -1, cv2.LINE_AA)
            anchor = next(((x, y) for (x, y, s) in pts if s >= score_thresh), None)
            if anchor is None:
                continue
            tx, ty = anchor[0] - 10, max(16, anchor[1] - 14)
            # SHOOT flash: only in a tight band around a consolidated event peak.
            near = [
                c
                for pf, c in (track_events or {}).get(tid, [])
                if abs(f - pf) <= flash_radius and c >= shoot_thresh
            ]
            if near:
                cv2.circle(frame, anchor, 18, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(
                    frame, f"P{tid} SHOOT {max(near):.2f}", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (0, 0, 255), 3, cv2.LINE_AA,
                )
                continue
            act = track_actions.get(tid, {}).get(f)
            if act is not None:
                label, conf = act
                tag_col = _ACTION_BGR.get(label, (210, 210, 210))
                if conf < 0.5:
                    tag_col = (130, 130, 130)
                cv2.putText(
                    frame, f"P{tid} {label} {conf:.2f}", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, tag_col, 1, cv2.LINE_AA,
                )
        cv2.putText(
            frame,
            "Stage 5 action recognizer (no idle class: labels are forced every frame)",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (0, 255, 255), 1, cv2.LINE_AA,
        )
        writer.write(frame)
        f += 1
    writer.release()
    cap.release()
    _log.info("wrote action overlay %s (%d frames)", output_path, f)
    return output_path


__all__ = ["make_action_overlay_video", "make_shot_overlay_video"]
