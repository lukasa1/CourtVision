"""Stage 6 — release-detection overlay (qualitative review).

Draws the shooter's COCO-17 skeleton, a bottom strip plotting the dominant
wrist's height over time, and ticks for the two predicted release frames
(``pose_apex`` = red, ``pose_forward`` = orange) so you can eyeball which lands
on the true release. CPU/cv2 only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from courtvision.logging import get_logger

_log = get_logger(__name__)

# COCO-17 skeleton connectivity.
_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12),
]
_APEX_BGR = (0, 0, 255)  # red
_FWD_BGR = (0, 165, 255)  # orange


def make_release_overlay_video(
    video_path: Path,
    keypoint: Any,  # [T, 17, 2]
    keypoint_score: Any,  # [T, 17]
    analysis: dict[str, Any],  # from analyze_release
    output_path: Path,
    *,
    max_seconds: float | None = None,
    kp_thresh: float = 0.2,
    flash_radius: int = 2,
) -> Path:
    """Render the skeleton + wrist-height strip + the two predicted releases."""
    import numpy as np

    video_path, output_path = Path(video_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps) if max_seconds else None
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # type: ignore[attr-defined]

    apex_f = int(analysis["apex"].release_frame_idx)
    fwd_f = int(analysis["forward"].release_frame_idx)
    apex_flag = analysis["apex"].confidence_flag
    fwd_flag = analysis["forward"].confidence_flag
    side = analysis["side"]
    wy = np.asarray(analysis["wrist_y"], dtype=np.float64)
    wsc = np.asarray(analysis["wrist_score"], dtype=np.float64)
    lo, hi = analysis["window"]
    total = len(wy)
    fs = max(0.5, w / 1400.0)

    # Wrist-height strip geometry (bottom band). Normalise height within window.
    band_h = int(h * 0.12)
    y0 = h - band_h - 4
    valid = wsc[lo:hi] >= kp_thresh
    if valid.any():
        ymin, ymax = float(wy[lo:hi][valid].min()), float(wy[lo:hi][valid].max())
    else:
        ymin, ymax = 0.0, 1.0
    span = max(ymax - ymin, 1.0)

    def strip_x(f: int) -> int:
        return int(8 + (f / max(total - 1, 1)) * (w - 16))

    def strip_y(yval: float) -> int:  # invert so a higher hand plots higher
        return int(y0 + band_h - 6 - ((ymax - yval) / span) * (band_h - 12))

    kp = np.asarray(keypoint, dtype=np.float64)
    ks = np.asarray(keypoint_score, dtype=np.float64)

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and f >= max_frames):
            break
        if f < len(kp):
            pt, sc = kp[f], ks[f]
            for a, b in _EDGES:
                if sc[a] > kp_thresh and sc[b] > kp_thresh:
                    cv2.line(
                        frame, (int(pt[a][0]), int(pt[a][1])), (int(pt[b][0]), int(pt[b][1])),
                        (0, 255, 0), 2, cv2.LINE_AA,
                    )
            for j in range(len(pt)):
                if sc[j] > kp_thresh:
                    cv2.circle(frame, (int(pt[j][0]), int(pt[j][1])), 3, (0, 200, 255), -1, cv2.LINE_AA)

        # header
        cv2.putText(
            frame,
            f"side={side}  apex f={apex_f}({apex_flag})  fwd f={fwd_f}({fwd_flag})",
            (10, int(h * 0.04)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (255, 255, 0), 2, cv2.LINE_AA,
        )

        # wrist-height strip
        cv2.rectangle(frame, (4, y0), (w - 4, h - 4), (40, 40, 40), -1)
        # shade the forward-active interval (arm extended overhead) as a band
        interval = analysis.get("forward_interval")
        if interval is not None:
            bx0, bx1 = strip_x(int(interval[0])), strip_x(int(interval[1]) - 1)
            sub = frame[y0 : h - 4, bx0 : bx1 + 1]
            tint = sub.copy()
            tint[:] = (0, 90, 140)  # forward-active band (dim orange)
            frame[y0 : h - 4, bx0 : bx1 + 1] = cv2.addWeighted(sub, 0.5, tint, 0.5, 0)
        prev = None
        for ff in range(lo, hi):
            if wsc[ff] < kp_thresh:
                prev = None
                continue
            pxy = (strip_x(ff), strip_y(float(wy[ff])))
            if prev is not None:
                cv2.line(frame, prev, pxy, (200, 200, 200), 1, cv2.LINE_AA)
            prev = pxy
        for mf, col in ((apex_f, _APEX_BGR), (fwd_f, _FWD_BGR)):
            cv2.line(frame, (strip_x(mf), y0), (strip_x(mf), h - 4), col, 2, cv2.LINE_AA)
        cv2.line(frame, (strip_x(f), y0), (strip_x(f), h - 4), (255, 255, 255), 1, cv2.LINE_AA)

        # release flashes on the player
        wr_idx = 9 if side == "L" else 10
        anchor = (int(kp[f, wr_idx, 0]), int(kp[f, wr_idx, 1])) if f < len(kp) else (w // 2, h // 2)
        # Markers appear AT the detected frame and persist forward (never
        # before), so the circle's first appearance == the detected frame ==
        # the trace bar (previously a symmetric ±flash made it appear early).
        if 0 <= f - apex_f <= flash_radius:
            cv2.circle(frame, anchor, 20, _APEX_BGR, 3, cv2.LINE_AA)
            cv2.putText(frame, "APEX", (anchor[0] + 12, anchor[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, _APEX_BGR, 2, cv2.LINE_AA)
        if 0 <= f - fwd_f <= flash_radius:
            cv2.circle(frame, anchor, 28, _FWD_BGR, 3, cv2.LINE_AA)
            cv2.putText(frame, "FWD", (anchor[0] + 12, anchor[1] + int(fs * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, _FWD_BGR, 2, cv2.LINE_AA)

        writer.write(frame)
        f += 1
    writer.release()
    cap.release()
    _log.info("wrote release overlay %s (%d frames)", output_path, f)
    return output_path


__all__ = ["make_release_overlay_video"]
