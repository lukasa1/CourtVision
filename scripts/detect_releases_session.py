"""Stage 6 — detect the exact release frame per Stage 5 event, using 3D pose.

Approach (3D adaptation of the existing rule_forward.py logic):
    Per event window (default ±32 frames around the Stage 5 detection), find
    the SHOOTER track as the player whose dominant wrist reaches the highest
    world-Z in the window. Then on that player's 3D pose:
        - "forward active" mask = wrist_z > shoulder_z  AND  elbow_angle >= 160°
        - first sustained interval (≥ min_run consecutive frames)
        - release_frame = interval_start + release_offset (default -1)
        - fallback if no sustained interval: wrist_apex (frame of max wrist Z)

Output:
    outputs/sessions/<session>/release_frames.json  — ReleaseFrameSequence

Evaluation:
    For each detected event whose Stage 5 detection matched a GT shot, compute
    Δframes = release_frame_predicted - release_frame_gt. Report mean / median /
    P90 / max plus per-shot detail.

Usage:
    uv run python scripts/detect_releases_session.py --session 2026-06-02_1on1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from courtvision.io_utils import read_json_as, write_json
from courtvision.schemas import (
    Pose3DSequence,
    ReleaseFrame,
    ReleaseFrameSequence,
)


# Halpe-26 indices (matches the rest of the codebase)
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10


def _angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at B between rays BA and BC, in degrees."""
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return float("nan")
    cos = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _track_window(pose3d_seq: Pose3DSequence, track_id: int,
                  frame_lo: int, frame_hi: int) -> dict[int, Any]:
    return {p.frame_idx: p for p in pose3d_seq.poses
            if p.track_id == track_id and frame_lo <= p.frame_idx < frame_hi}


def _identify_shooter(pose3d_seq: Pose3DSequence, center_frame: int,
                      half_window: int = 32) -> tuple[int | None, str | None]:
    """At each frame in the window, find which player's dominant wrist is
    HIGHEST (peak world-Z). The shooter is the player with the global max.

    Returns (track_id, side) where side ∈ {'L','R'}.
    """
    lo = max(0, center_frame - half_window)
    hi = center_frame + half_window
    # candidate tracks present in window
    poses_in_window = [p for p in pose3d_seq.poses if lo <= p.frame_idx < hi]
    if not poses_in_window:
        return None, None
    best: tuple[float, int, str] | None = None
    for p in poses_in_window:
        for w_idx, side in [(_L_WRIST, "L"), (_R_WRIST, "R")]:
            kp = p.keypoints[w_idx]
            if kp.n_cameras < 2:
                continue
            if best is None or kp.z > best[0]:
                best = (kp.z, p.track_id, side)
    if best is None:
        return None, None
    return best[1], best[2]


def _build_arrays(pose3d_seq: Pose3DSequence, track_id: int,
                  frame_lo: int, frame_hi: int, side: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame arrays (length N = hi - lo) for the chosen side:
    wrist_xyz, elbow_xyz, shoulder_xyz, valid_mask, present_mask."""
    n = frame_hi - frame_lo
    by_frame = _track_window(pose3d_seq, track_id, frame_lo, frame_hi)
    sh_idx = _R_SHOULDER if side == "R" else _L_SHOULDER
    el_idx = _R_ELBOW if side == "R" else _L_ELBOW
    wr_idx = _R_WRIST if side == "R" else _L_WRIST
    wrist = np.full((n, 3), np.nan)
    elbow = np.full((n, 3), np.nan)
    shoulder = np.full((n, 3), np.nan)
    valid = np.zeros(n, dtype=bool)
    present = np.zeros(n, dtype=bool)
    for i in range(n):
        p = by_frame.get(frame_lo + i)
        if p is None: continue
        present[i] = True
        wkp = p.keypoints[wr_idx]
        ekp = p.keypoints[el_idx]
        skp = p.keypoints[sh_idx]
        if min(wkp.n_cameras, ekp.n_cameras, skp.n_cameras) >= 2:
            wrist[i] = (wkp.x, wkp.y, wkp.z)
            elbow[i] = (ekp.x, ekp.y, ekp.z)
            shoulder[i] = (skp.x, skp.y, skp.z)
            valid[i] = True
    return wrist, elbow, shoulder, valid, present


def detect_release_3d(
    pose3d_seq: Pose3DSequence,
    event_frame: int,
    half_window: int = 32,
    elbow_extend_deg: float = 160.0,
    release_offset: int = -1,
    min_run: int = 2,
) -> dict[str, Any] | None:
    """For one Stage 5 event, find the release frame using 3D wrist + shoulder + elbow.

    Returns dict with keys: event_frame, release_frame, method, shooter_track_id,
    side, wrist_z_at_release, confidence_flag.
    """
    shooter_tid, side = _identify_shooter(pose3d_seq, event_frame, half_window)
    if shooter_tid is None or side is None:
        return None

    frame_lo = max(0, event_frame - half_window)
    frame_hi = event_frame + half_window
    wrist, elbow, shoulder, valid, present = _build_arrays(
        pose3d_seq, shooter_tid, frame_lo, frame_hi, side
    )

    # Forward-active mask: wrist_z > shoulder_z AND elbow angle >= threshold
    n = len(wrist)
    forward = np.zeros(n, dtype=bool)
    for i in range(n):
        if not valid[i]: continue
        if wrist[i, 2] <= shoulder[i, 2]: continue
        ang = _angle_3d(shoulder[i], elbow[i], wrist[i])
        if not np.isfinite(ang): continue
        if ang >= elbow_extend_deg:
            forward[i] = True

    # First sustained interval (≥ min_run consecutive True frames)
    interval_start: int | None = None
    run = 0
    for i in range(n):
        if forward[i]:
            run += 1
            if run == min_run:
                interval_start = i - min_run + 1
                break
        else:
            run = 0

    if interval_start is not None:
        rel_local = max(0, min(interval_start + release_offset, n - 1))
        method = "pose_forward_3d"
        confidence = "high"
    else:
        # Fallback: apex of wrist_z
        valid_z = np.where(valid, wrist[:, 2], -np.inf)
        if not np.isfinite(valid_z).any():
            return None
        rel_local = int(np.argmax(valid_z))
        method = "pose_apex_3d_fallback"
        confidence = "low"

    release_frame = frame_lo + rel_local
    return {
        "event_frame": event_frame,
        "release_frame": int(release_frame),
        "method": method,
        "shooter_track_id": int(shooter_tid),
        "side": side,
        "wrist_z_at_release_m": float(wrist[rel_local, 2]) if valid[rel_local] else None,
        "confidence": confidence,
        "interval_start_local": int(interval_start) if interval_start is not None else None,
        "half_window": half_window,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="2026-06-02_1on1")
    p.add_argument("--half-window", type=int, default=32)
    args = p.parse_args()

    out_dir = Path(f"outputs/sessions/{args.session}")
    pose3d = read_json_as(Pose3DSequence, out_dir / "pose3d.json")
    eval_data = json.loads((out_dir / "evaluation.json").read_text())
    gt = json.loads(Path(f"data/sessions/{args.session}/shots_gt.json").read_text())
    print(f"Loaded {len(pose3d.poses)} 3D poses, {gt['n_shots']} GT shots")

    # Stage 5 fused predictions
    s5 = eval_data["stage5_shot_detection"]
    preds = s5["predictions"]
    print(f"Stage 5: {len(preds)} fused events ({s5['summary']['true_positives']} TP)")

    # Per-event release detection
    detections: list[dict[str, Any]] = []
    for pe in preds:
        d = detect_release_3d(pose3d, pe["frame_idx_a"], half_window=args.half_window)
        if d is None:
            d = {"event_frame": pe["frame_idx_a"], "release_frame": None,
                 "method": "no_shooter_identified"}
        d["matched_gt_shot_id"] = pe.get("matched_gt_shot_id")
        d["is_tp"] = pe["is_tp"]
        detections.append(d)

    # Persist as a JSON (we don't fully populate the ReleaseFrameSequence schema
    # since it requires fields like calibration_ref + 2D click that don't apply here).
    (out_dir / "release_frames.json").write_text(json.dumps({
        "session": args.session,
        "n_events": len(detections),
        "detections": detections,
    }, indent=2))
    print(f"  Saved → {out_dir / 'release_frames.json'}")

    # Evaluate Δframes vs GT — only for TP events
    deltas: list[int] = []
    per_shot: list[dict[str, Any]] = []
    gt_by_id = {i: s for i, s in enumerate(gt["shots"])}
    for d in detections:
        if not d["is_tp"]: continue
        gt_shot = gt_by_id.get(d["matched_gt_shot_id"])
        if gt_shot is None: continue
        if d["release_frame"] is None: continue
        delta = d["release_frame"] - gt_shot["release_frame_a"]
        deltas.append(delta)
        per_shot.append({
            "shot_id": d["matched_gt_shot_id"],
            "gt_release_frame_a": gt_shot["release_frame_a"],
            "predicted_release_frame": d["release_frame"],
            "delta_frames": delta,
            "method": d["method"],
            "confidence": d.get("confidence"),
            "side": d.get("side"),
            "wrist_z_m": d.get("wrist_z_at_release_m"),
        })
    per_shot.sort(key=lambda r: r["shot_id"])

    print("\n" + "=" * 78)
    print(f" STAGE 6 — Release frame Δframes vs GT  (n={len(deltas)} TP events)")
    print("=" * 78)
    if deltas:
        arr = np.array(deltas)
        print(f"  mean   Δframes = {arr.mean():+6.2f}")
        print(f"  median Δframes = {np.median(arr):+6.2f}")
        print(f"  mean   |Δframes| = {np.abs(arr).mean():.2f}")
        print(f"  P90    |Δframes| = {np.percentile(np.abs(arr), 90):.2f}")
        print(f"  max    |Δframes| = {np.abs(arr).max()}")
        print()
        print(f"  {'shot':>4}  {'gt':>5}  {'pred':>5}  {'Δ':>5}  {'method':>22}  {'conf':>5}  side  wrist_z")
        for r in per_shot:
            print(f"  {r['shot_id']:>4}  {r['gt_release_frame_a']:>5}  {r['predicted_release_frame']:>5}  "
                  f"{r['delta_frames']:>+5}  {r['method']:>22}  {r['confidence']:>5}  {r['side']:>4}  "
                  f"{r['wrist_z_m']:.2f}" if r['wrist_z_m'] is not None else "  —")

    # Augment evaluation.json
    eval_data["stage6_release_frame"] = {
        "summary": {
            "n_tp_events": len(deltas),
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "median_delta": float(np.median(deltas)) if deltas else None,
            "mean_abs_delta": float(np.mean(np.abs(deltas))) if deltas else None,
            "p90_abs_delta": float(np.percentile(np.abs(deltas), 90)) if deltas else None,
            "max_abs_delta": int(np.max(np.abs(deltas))) if deltas else None,
            "half_window": args.half_window,
        },
        "per_shot": per_shot,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(eval_data, indent=2))
    print(f"\n✓ Stage 6 results saved to evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
