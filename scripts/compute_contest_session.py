"""Stage 7 — Full 5-measurement pose-only contest metric.

Implements the spec from docs/decisions.md Decision 9 + the 2026-06-02 pose-only
redesign block. Per shot, at the predicted release frame:

    (1) wrist_to_wrist_distance_m  — 3D dist, closest defender wrist → shooter shooting wrist
    (2) wrist_height_advantage_m   — defender's HIGHER wrist Z − shooter's shooting wrist Z
    (3) shooter_fade_mps           — sqrt(max(0, -v_forward)^2 + v_lateral^2) in court frame
    (4) floor_distance_m           — 2D pelvis-to-pelvis (NBA SportVU baseline)
    (5) composite_score            — z-scored sum of (1)+(2)+(3), sign-corrected so bigger = more contested

Selection rules (pose-only redesign):
    - Closest defender = defender pelvis closest to shooter pelvis (3D)
    - Shooting hand = shooter's higher wrist at release (Z)

Reporting:
    - Each measurement computed both INSTANTANEOUS at the release frame AND
      SMOOTHED (5-frame mean centered on release).
    - `uncontested = True` if floor_distance > 3.0 m.
    - GT comparison for (1) and (2) only (we have wrist GT but not pelvis GT).

Usage:
    uv run python scripts/compute_contest_session.py --session 2026-06-02_1on1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.io_utils import read_json_as
from courtvision.schemas import Pose3DSequence


_L_WRIST, _R_WRIST = 9, 10
_PELVIS = 19  # Halpe-26 "hip center"
RIM_3D = np.array([0.0, 0.38, 3.05])
UNCONTESTED_THRESH_M = 3.0
SMOOTH_HALF_WIN = 2  # ±2 frames → 5-frame mean
VEL_HALF_WIN = 3     # central difference over ±3 frames


# ----- calibration / triangulation helpers (for GT triangulation) -----

def _load_calib(intrinsics_path: Path, extrinsics_path: Path) -> dict[str, np.ndarray]:
    intr = json.loads(intrinsics_path.read_text())
    extr = json.loads(extrinsics_path.read_text())
    K = np.asarray(intr["K"], dtype=np.float64)
    dist = np.asarray(intr["dist"], dtype=np.float64).ravel()
    R = np.asarray(extr["R"], dtype=np.float64)
    t = np.asarray(extr["tvec"], dtype=np.float64).reshape(3, 1)
    P = K @ np.hstack([R, t])
    return {"K": K, "dist": dist, "R": R, "t": t, "P": P}


def _triangulate_2d(xa, ya, Pa, Ka, dist_a, xb, yb, Pb, Kb, dist_b) -> np.ndarray:
    ua = cv2.undistortPoints(np.array([[[xa, ya]]], dtype=np.float64), Ka, dist_a, P=Ka).reshape(2, 1)
    ub = cv2.undistortPoints(np.array([[[xb, yb]]], dtype=np.float64), Kb, dist_b, P=Kb).reshape(2, 1)
    X_h = cv2.triangulatePoints(Pa, Pb, ua, ub)
    return (X_h[:3] / X_h[3]).ravel()


# ----- 3D-pose helpers -----

def _pose3d_keypoint(pose, idx: int) -> np.ndarray | None:
    if idx >= len(pose.keypoints):
        return None
    kp = pose.keypoints[idx]
    if kp.n_cameras < 2:
        return None
    return np.array([kp.x, kp.y, kp.z])


def _frame_poses(pose3d_seq: Pose3DSequence) -> dict[int, list]:
    out: dict[int, list] = {}
    for p in pose3d_seq.poses:
        out.setdefault(p.frame_idx, []).append(p)
    return out


def _smoothed_keypoint(by_frame, track_id: int, kp_idx: int, frame: int, half_win: int = SMOOTH_HALF_WIN) -> np.ndarray | None:
    """Average the keypoint over [frame-half_win, frame+half_win] for the same track_id."""
    pts: list[np.ndarray] = []
    for f in range(frame - half_win, frame + half_win + 1):
        poses_at = by_frame.get(f, [])
        for p in poses_at:
            if p.track_id != track_id:
                continue
            pt = _pose3d_keypoint(p, kp_idx)
            if pt is not None:
                pts.append(pt)
            break
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _pelvis_velocity(by_frame, track_id: int, frame: int, fps: float,
                     half_win: int = VEL_HALF_WIN) -> np.ndarray | None:
    """Central-difference pelvis velocity at `frame`. Returns m/s in world coords."""
    p_before = _smoothed_keypoint(by_frame, track_id, _PELVIS, frame - half_win, half_win=0)
    p_after = _smoothed_keypoint(by_frame, track_id, _PELVIS, frame + half_win, half_win=0)
    if p_before is None or p_after is None:
        return None
    dt = (2 * half_win) / fps
    return (p_after - p_before) / dt


def _identify_shooter(by_frame, center_frame: int, half_window: int) -> tuple[int | None, str | None]:
    """Find shooter = the player whose dominant wrist reaches highest Z in the window."""
    best: tuple[float, int, str] | None = None
    for f in range(center_frame - half_window, center_frame + half_window):
        for p in by_frame.get(f, []):
            for w_idx, side in [(_L_WRIST, "L"), (_R_WRIST, "R")]:
                w = _pose3d_keypoint(p, w_idx)
                if w is None:
                    continue
                if best is None or w[2] > best[0]:
                    best = (w[2], p.track_id, side)
    return (best[1], best[2]) if best else (None, None)


def _shooting_side_at(pose, score_thr: float = 0.0) -> str | None:
    """Higher-Z wrist of this pose is the shooting hand."""
    lw = _pose3d_keypoint(pose, _L_WRIST)
    rw = _pose3d_keypoint(pose, _R_WRIST)
    if lw is None and rw is None:
        return None
    if lw is None:
        return "R"
    if rw is None:
        return "L"
    return "R" if rw[2] > lw[2] else "L"


def _defender_pose(by_frame, frame: int, shooter_pose) -> Any:
    """Closest-pelvis-to-shooter (excluding shooter track) at this frame."""
    shooter_pelvis = _pose3d_keypoint(shooter_pose, _PELVIS)
    if shooter_pelvis is None:
        return None
    candidates: list[tuple[float, Any]] = []
    for p in by_frame.get(frame, []):
        if p.track_id == shooter_pose.track_id:
            continue
        d_pelvis = _pose3d_keypoint(p, _PELVIS)
        if d_pelvis is None:
            continue
        candidates.append((float(np.linalg.norm(d_pelvis - shooter_pelvis)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _instant_measurements(shooter_pose, defender_pose, shooter_side: str,
                          pelvis_velocity: np.ndarray | None) -> dict[str, float | None]:
    """Compute the 4 measurements (1, 2, 3, 4) from a pair of poses + velocity."""
    out: dict[str, float | None] = {}
    # Shooter shooting wrist
    w_idx = _R_WRIST if shooter_side == "R" else _L_WRIST
    shooter_wrist = _pose3d_keypoint(shooter_pose, w_idx)
    shooter_pelvis = _pose3d_keypoint(shooter_pose, _PELVIS)

    # (1) wrist-to-wrist: defender's closest wrist to shooter wrist
    out["wrist_to_wrist_distance_m"] = None
    out["wrist_height_advantage_m"] = None
    out["floor_distance_m"] = None
    out["shooter_fade_mps"] = None

    if defender_pose is not None and shooter_wrist is not None:
        def_lw = _pose3d_keypoint(defender_pose, _L_WRIST)
        def_rw = _pose3d_keypoint(defender_pose, _R_WRIST)
        def_wrists = [w for w in (def_lw, def_rw) if w is not None]
        if def_wrists:
            d2w = [float(np.linalg.norm(w - shooter_wrist)) for w in def_wrists]
            out["wrist_to_wrist_distance_m"] = round(min(d2w), 4)
            # (2) wrist height advantage: defender's HIGHER wrist z − shooter wrist z
            def_higher = max(w[2] for w in def_wrists)
            out["wrist_height_advantage_m"] = round(float(def_higher - shooter_wrist[2]), 4)

    # (4) floor distance: 2D pelvis-to-pelvis
    if defender_pose is not None and shooter_pelvis is not None:
        def_pelvis = _pose3d_keypoint(defender_pose, _PELVIS)
        if def_pelvis is not None:
            d_xy = float(np.linalg.norm(def_pelvis[:2] - shooter_pelvis[:2]))
            out["floor_distance_m"] = round(d_xy, 4)

    # (3) shooter fade — sqrt(max(0, -v_forward)^2 + v_lateral^2)
    if pelvis_velocity is not None and shooter_pelvis is not None:
        # forward = unit vector from shooter pelvis to rim (xy only)
        f_xy = (RIM_3D[:2] - shooter_pelvis[:2])
        n = np.linalg.norm(f_xy)
        if n > 1e-6:
            forward = f_xy / n
            # lateral is the perpendicular in xy (rotate 90° CCW)
            lateral = np.array([-forward[1], forward[0]])
            v_xy = pelvis_velocity[:2]
            v_forward = float(np.dot(v_xy, forward))
            v_lateral = float(np.dot(v_xy, lateral))
            fade = float(np.sqrt(max(0.0, -v_forward) ** 2 + v_lateral ** 2))
            out["shooter_fade_mps"] = round(fade, 4)
            out["_v_forward_mps"] = round(v_forward, 4)
            out["_v_lateral_mps"] = round(v_lateral, 4)
    return out


def _pred_contest_full(
    pose3d_seq: Pose3DSequence, by_frame: dict[int, list],
    release_frame: int, shooter_track_id: int, fps: float,
) -> dict[str, Any]:
    """Compute the full 5-measurement contest metric at a release frame."""
    out: dict[str, Any] = {"release_frame": release_frame, "shooter_track_id": shooter_track_id}
    shooter_pose = next((p for p in by_frame.get(release_frame, []) if p.track_id == shooter_track_id), None)
    if shooter_pose is None:
        # fallback — find by highest wrist Z
        tid, side = _identify_shooter(by_frame, release_frame, 1)
        if tid is None:
            out["status"] = "no shooter found"; return out
        shooter_pose = next((p for p in by_frame.get(release_frame, []) if p.track_id == tid), None)
        out["shooter_track_id"] = tid

    shooter_side = _shooting_side_at(shooter_pose) or "R"
    out["shooter_side"] = shooter_side

    defender_pose = _defender_pose(by_frame, release_frame, shooter_pose)
    out["defender_track_id"] = defender_pose.track_id if defender_pose is not None else None

    pelvis_v = _pelvis_velocity(by_frame, shooter_pose.track_id, release_frame, fps)

    # Instantaneous
    inst = _instant_measurements(shooter_pose, defender_pose, shooter_side, pelvis_v)

    # Smoothed (±2 frames). Construct synthetic averaged "shooter_pose" and "defender_pose"
    # by averaging the relevant keypoints across the window.
    class _AvgPose:
        def __init__(self): self.keypoints = []
        def __repr__(self): return "<AvgPose>"

    def _avg_pose(track_id: int, indices: tuple[int, ...]) -> Any:
        from courtvision.schemas import Keypoint3D
        ap = _AvgPose()
        # Pad with NaN-confidence zeros for any keypoint we don't average; only fill the ones we use.
        ap.keypoints = [Keypoint3D(x=0, y=0, z=0, confidence=0, n_cameras=0)] * 26
        for i in indices:
            p_avg = _smoothed_keypoint(by_frame, track_id, i, release_frame)
            if p_avg is not None:
                ap.keypoints[i] = Keypoint3D(x=float(p_avg[0]), y=float(p_avg[1]), z=float(p_avg[2]),
                                              confidence=1.0, n_cameras=2)
        return ap

    needed = (_L_WRIST, _R_WRIST, _PELVIS)
    smoothed_shooter = _avg_pose(shooter_pose.track_id, needed)
    smoothed_defender = _avg_pose(defender_pose.track_id, needed) if defender_pose is not None else None
    smoothed = _instant_measurements(smoothed_shooter, smoothed_defender, shooter_side, pelvis_v)

    out["instantaneous"] = inst
    out["smoothed_5frame"] = smoothed
    # Uncontested check (use instantaneous floor distance)
    fd = inst.get("floor_distance_m")
    out["uncontested"] = (fd is not None and fd > UNCONTESTED_THRESH_M)
    out["status"] = "ok"
    return out


def _gt_for_shot(shot: dict, calib_a: dict, calib_b: dict) -> dict[str, Any]:
    """Triangulate GT clicks → GT shooter wrist 3D, GT defender wrist(s) 3D, then compute (1) and (2)."""
    out: dict[str, Any] = {"shot_id": shot["shot_id"]}
    sa = shot.get("shooter_wrist_a"); sb = shot.get("shooter_wrist_b")
    if sa is None or sb is None:
        out["status"] = "missing shooter GT"; return out
    shooter_3d = _triangulate_2d(
        sa[0], sa[1], calib_a["P"], calib_a["K"], calib_a["dist"],
        sb[0], sb[1], calib_b["P"], calib_b["K"], calib_b["dist"],
    )
    out["gt_shooter_wrist_m"] = [round(float(v), 4) for v in shooter_3d]

    def_wrists_3d: list[np.ndarray] = []
    for ka, kb in [("defender_wrist1_a", "defender_wrist1_b"),
                   ("defender_wrist2_a", "defender_wrist2_b")]:
        pa = shot.get(ka); pb = shot.get(kb)
        if pa is None or pb is None: continue
        def_wrists_3d.append(_triangulate_2d(
            pa[0], pa[1], calib_a["P"], calib_a["K"], calib_a["dist"],
            pb[0], pb[1], calib_b["P"], calib_b["K"], calib_b["dist"],
        ))
    if not def_wrists_3d:
        out["status"] = "no defender GT wrist with both cameras"; return out
    out["gt_defender_wrists_m"] = [[round(float(v), 4) for v in w] for w in def_wrists_3d]
    # (1) wrist-to-wrist
    dists = [float(np.linalg.norm(w - shooter_3d)) for w in def_wrists_3d]
    out["gt_wrist_to_wrist_distance_m"] = round(min(dists), 4)
    # (2) wrist height advantage (using the defender wrist with the highest z, not the closest one)
    def_higher = float(max(w[2] for w in def_wrists_3d))
    out["gt_wrist_height_advantage_m"] = round(def_higher - float(shooter_3d[2]), 4)
    out["status"] = "ok"
    return out


def _zscore(values: list[float | None]) -> list[float | None]:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(arr) < 2:
        return [None] * len(values)
    mu, sd = float(arr.mean()), float(arr.std(ddof=0))
    if sd < 1e-9:
        return [0.0 if v is not None else None for v in values]
    return [round((v - mu) / sd, 4) if v is not None else None for v in values]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="2026-06-02_1on1")
    args = p.parse_args()

    out_dir = Path(f"outputs/sessions/{args.session}")
    pose3d = read_json_as(Pose3DSequence, out_dir / "pose3d.json")
    eval_data = json.loads((out_dir / "evaluation.json").read_text())
    rel_data = json.loads((out_dir / "release_frames.json").read_text())
    gt = json.loads(Path(f"data/sessions/{args.session}/shots_gt.json").read_text())
    sync = json.loads(Path(f"data/sessions/{args.session}/sync.json").read_text())
    fps = sync["fps_a"]
    by_frame = _frame_poses(pose3d)

    calib_a = _load_calib(
        Path("data/calibration/intrinsics_A_checker.json"),
        Path(f"data/sessions/{args.session}/extrinsics_A.json"),
    )
    calib_b = _load_calib(
        Path("data/calibration/intrinsics_B_checker.json"),
        Path(f"data/sessions/{args.session}/extrinsics_B.json"),
    )
    rel_by_shot = {d["matched_gt_shot_id"]: d for d in rel_data["detections"]
                   if d.get("is_tp") and d.get("release_frame") is not None}

    per_shot: list[dict[str, Any]] = []
    for shot in gt["shots"]:
        sid = shot["shot_id"]
        row: dict[str, Any] = {
            "shot_id": sid, "make": shot["make"], "shot_type": shot["shot_type"],
            "uncontested_gt_flag": shot["uncontested"],
        }
        row["gt"] = _gt_for_shot(shot, calib_a, calib_b)
        rel = rel_by_shot.get(sid)
        if rel is None:
            row["pred"] = {"status": "no Stage 6 prediction"}
        else:
            row["pred"] = _pred_contest_full(pose3d, by_frame,
                                             rel["release_frame"], rel["shooter_track_id"], fps)
        per_shot.append(row)

    # ---- Composite z-score across the 15 shots ----
    # Sign convention: bigger composite = MORE contested.
    #   (1) wrist-to-wrist distance:   smaller = more contested  →  invert sign
    #   (2) wrist height advantage:    larger  = more contested  →  keep sign
    #   (3) shooter fade:              larger  = more contested  →  keep sign
    def _series(field: str, source: str = "instantaneous"):
        return [(r["pred"].get(source) or {}).get(field) if isinstance(r["pred"].get(source), dict) else None
                for r in per_shot]
    w_to_w = _series("wrist_to_wrist_distance_m")
    wh_adv = _series("wrist_height_advantage_m")
    fade = _series("shooter_fade_mps")
    z_w = _zscore(w_to_w)
    z_h = _zscore(wh_adv)
    z_f = _zscore(fade)
    composites: list[float | None] = []
    for zw, zh, zf in zip(z_w, z_h, z_f):
        parts = []
        if zw is not None: parts.append(-zw)   # invert (smaller distance = more contested)
        if zh is not None: parts.append(zh)
        if zf is not None: parts.append(zf)
        composites.append(round(sum(parts), 4) if parts else None)
    for row, c in zip(per_shot, composites):
        if isinstance(row["pred"].get("instantaneous"), dict):
            row["pred"]["instantaneous"]["composite_z"] = c
        else:
            row["pred"]["composite_z"] = c

    # ---- Errors for (1) and (2) — the two measurements with GT ----
    for row in per_shot:
        gt_w = row["gt"].get("gt_wrist_to_wrist_distance_m")
        pr_w = (row["pred"].get("instantaneous") or {}).get("wrist_to_wrist_distance_m")
        if gt_w is not None and pr_w is not None:
            row["err_wrist_to_wrist_m"] = round(abs(gt_w - pr_w), 4)
        gt_h = row["gt"].get("gt_wrist_height_advantage_m")
        pr_h = (row["pred"].get("instantaneous") or {}).get("wrist_height_advantage_m")
        if gt_h is not None and pr_h is not None:
            row["err_wrist_height_advantage_m"] = round(abs(gt_h - pr_h), 4)

    # ---- Summary statistics ----
    def _stats(field: str, source: str = "instantaneous"):
        vals = []
        for r in per_shot:
            v = (r["pred"].get(source) or {}).get(field) if isinstance(r["pred"].get(source), dict) else None
            if v is not None: vals.append(v)
        if not vals: return None
        return {
            "n": len(vals),
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        }
    def _make_miss_split(field: str):
        m, x = [], []
        for r in per_shot:
            v = (r["pred"].get("instantaneous") or {}).get(field) if isinstance(r["pred"].get("instantaneous"), dict) else None
            if v is None: continue
            (m if r["make"] else x).append(v)
        return {
            "makes_mean": round(float(np.mean(m)), 4) if m else None,
            "misses_mean": round(float(np.mean(x)), 4) if x else None,
            "makes_n": len(m), "misses_n": len(x),
        }

    summary = {
        "wrist_to_wrist_distance_m": {**(_stats("wrist_to_wrist_distance_m") or {}),
                                       "make_miss": _make_miss_split("wrist_to_wrist_distance_m")},
        "wrist_height_advantage_m":  {**(_stats("wrist_height_advantage_m") or {}),
                                       "make_miss": _make_miss_split("wrist_height_advantage_m")},
        "shooter_fade_mps":          {**(_stats("shooter_fade_mps") or {}),
                                       "make_miss": _make_miss_split("shooter_fade_mps")},
        "floor_distance_m":          {**(_stats("floor_distance_m") or {}),
                                       "make_miss": _make_miss_split("floor_distance_m")},
        "composite_z":               {**(_stats("composite_z") or {}),
                                       "make_miss": _make_miss_split("composite_z")},
    }

    # Error stats for (1) and (2) — the two with GT
    errs_w = [r.get("err_wrist_to_wrist_m") for r in per_shot if r.get("err_wrist_to_wrist_m") is not None]
    errs_h = [r.get("err_wrist_height_advantage_m") for r in per_shot if r.get("err_wrist_height_advantage_m") is not None]
    summary["wrist_to_wrist_distance_m_error_vs_gt_m"] = {
        "n": len(errs_w),
        "mean_abs": round(float(np.mean(errs_w)), 4) if errs_w else None,
        "median_abs": round(float(np.median(errs_w)), 4) if errs_w else None,
        "max_abs": round(float(np.max(errs_w)), 4) if errs_w else None,
    } if errs_w else None
    summary["wrist_height_advantage_m_error_vs_gt_m"] = {
        "n": len(errs_h),
        "mean_abs": round(float(np.mean(errs_h)), 4) if errs_h else None,
        "median_abs": round(float(np.median(errs_h)), 4) if errs_h else None,
        "max_abs": round(float(np.max(errs_h)), 4) if errs_h else None,
    } if errs_h else None
    summary["uncontested_count"] = sum(1 for r in per_shot
                                       if isinstance(r["pred"].get("instantaneous"), dict)
                                       and r["pred"].get("uncontested"))

    # ---- Print ----
    print("=" * 88)
    print(" STAGE 7 — Full 5-measurement pose-only contest metric (Decision 9 pose-only redesign)")
    print("=" * 88)
    for field, label, unit in [
        ("wrist_to_wrist_distance_m", "Wrist-to-wrist distance", "m"),
        ("wrist_height_advantage_m",  "Wrist height advantage  ", "m"),
        ("shooter_fade_mps",          "Shooter fade            ", "m/s"),
        ("floor_distance_m",          "Floor distance (pelvis) ", "m"),
        ("composite_z",               "Composite z-score       ", ""),
    ]:
        s = summary[field]
        if s.get("mean") is None: continue
        mm = s["make_miss"]
        print(f"\n  {label}  (n={s['n']})")
        print(f"    mean {s['mean']:+7.3f} {unit:3s}   median {s['median']:+7.3f}   "
              f"range [{s['min']:+.3f}, {s['max']:+.3f}]   std {s['std']:.3f}")
        if mm.get("makes_n", 0) and mm.get("misses_n", 0):
            print(f"    makes (n={mm['makes_n']}): {mm['makes_mean']:+.3f}   "
                  f"misses (n={mm['misses_n']}): {mm['misses_mean']:+.3f}")

    if summary.get("wrist_to_wrist_distance_m_error_vs_gt_m"):
        e = summary["wrist_to_wrist_distance_m_error_vs_gt_m"]
        print(f"\n  Error vs GT — wrist-to-wrist:  n={e['n']}, "
              f"mean |err| {e['mean_abs']*100:.1f} cm, median {e['median_abs']*100:.1f}, max {e['max_abs']*100:.1f}")
    if summary.get("wrist_height_advantage_m_error_vs_gt_m"):
        e = summary["wrist_height_advantage_m_error_vs_gt_m"]
        print(f"  Error vs GT — height advantage: n={e['n']}, "
              f"mean |err| {e['mean_abs']*100:.1f} cm, median {e['median_abs']*100:.1f}, max {e['max_abs']*100:.1f}")
    print(f"\n  Uncontested shots (floor dist > {UNCONTESTED_THRESH_M} m): {summary['uncontested_count']}")

    # Per-shot table
    print("\n  Per-shot (instantaneous):")
    print(f"  {'shot':>4}  {'make':>5}  {'w-w (m)':>8}  {'hgt (m)':>8}  {'fade':>6}  {'floor':>7}  {'z':>6}  {'err_w_cm':>8}  {'unc':>3}")
    for r in per_shot:
        inst = r["pred"].get("instantaneous") if isinstance(r["pred"].get("instantaneous"), dict) else {}
        def _f(x, fmt="{:+.3f}"):
            return fmt.format(x) if x is not None else "--"
        w = inst.get("wrist_to_wrist_distance_m")
        h = inst.get("wrist_height_advantage_m")
        fade = inst.get("shooter_fade_mps")
        fd = inst.get("floor_distance_m")
        z = inst.get("composite_z")
        err = r.get("err_wrist_to_wrist_m")
        unc = "Y" if r["pred"].get("uncontested") else "."
        print(f"  {r['shot_id']:>4}  {'make' if r['make'] else 'miss':>5}  "
              f"{_f(w):>8}  {_f(h):>8}  {_f(fade,fmt='{:.2f}'):>6}  {_f(fd):>7}  {_f(z,fmt='{:+.2f}'):>6}  "
              f"{(f'{err*100:.1f}' if err is not None else '--'):>8}  {unc:>3}")

    # Save
    out = {"session": args.session, "summary": summary, "per_shot": per_shot}
    (out_dir / "contest_metrics.json").write_text(json.dumps(out, indent=2))
    eval_data["stage7_contest_metric"] = out
    (out_dir / "evaluation.json").write_text(json.dumps(eval_data, indent=2))
    print(f"\n✓ Saved → contest_metrics.json + augmented evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
