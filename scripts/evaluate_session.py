"""Compute headline pipeline-vs-GT metrics for a session.

Inputs (all in their canonical locations under outputs/ + data/):
    outputs/sessions/<session>/pose2d_cam_a.json     (Stage 2)
    outputs/sessions/<session>/pose2d_cam_b.json
    outputs/sessions/<session>/pose3d.json            (Stage 4 triangulation)
    data/sessions/<session>/shots_gt.json             (hand-labeled GT)
    data/calibration/intrinsics_A_checker.json + extrinsics_A.json
    (same for B)

Metrics produced:

    Stage 2 (per camera) — wrist PCK
        For each shot, match the GT-clicked shooter / defender wrists to the
        closest predicted wrist on a tracked player at the release frame.
        Report:
          - mean pixel error
          - PCK@0.05 (correct if err < 5% of GT-matched-player bbox diagonal)
          - per-shot table

    Stage 4 — 3D wrist accuracy (cm)
        Triangulate the GT clicks in A + B to get a GT 3D wrist position.
        Compare to the predicted 3D wrist from pose3d.json at the same release
        frame. Report mean / median / P90 error in centimetres.

    Future (gated on existence of inputs):
        Stage 5 shot precision / recall, Stage 6 Δframes, Stage 7 contest.

Usage:
    uv run python scripts/evaluate_session.py --session 2026-06-02_1on1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.io_utils import read_json_as
from courtvision.schemas import Pose2DSequence, Pose3DSequence


_WRIST_LEFT_HALPE = 9   # Halpe-26 left wrist
_WRIST_RIGHT_HALPE = 10
_SCORE_THRESHOLD = 0.3


def _load_calib(intrinsics_path: Path, extrinsics_path: Path) -> dict[str, np.ndarray]:
    intr = json.loads(intrinsics_path.read_text())
    extr = json.loads(extrinsics_path.read_text())
    K = np.asarray(intr["K"], dtype=np.float64)
    dist = np.asarray(intr["dist"], dtype=np.float64).ravel()
    R = np.asarray(extr["R"], dtype=np.float64)
    t = np.asarray(extr["tvec"], dtype=np.float64).reshape(3, 1)
    P = K @ np.hstack([R, t])
    return {"K": K, "dist": dist, "R": R, "t": t, "P": P}


def _pose2d_by_frame(seq: Pose2DSequence) -> dict[int, list]:
    by_frame: dict[int, list] = {}
    for p in seq.poses:
        by_frame.setdefault(p.frame_idx, []).append(p)
    return by_frame


def _pose3d_by_frame(seq: Pose3DSequence) -> dict[int, list]:
    by_frame: dict[int, list] = {}
    for p in seq.poses:
        by_frame.setdefault(p.frame_idx, []).append(p)
    return by_frame


def _bbox_from_keypoints(kps: list, score_thr: float = _SCORE_THRESHOLD) -> tuple[float, float, float, float] | None:
    """Compute (xmin, ymin, xmax, ymax) from confident keypoints."""
    xs = [kp.x for kp in kps if kp.score >= score_thr]
    ys = [kp.y for kp in kps if kp.score >= score_thr]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _match_wrist_pose(
    poses_at_frame: list,
    gt_x: float, gt_y: float,
    score_thr: float = _SCORE_THRESHOLD,
) -> tuple[Any, int, float] | None:
    """Find the pose whose closest wrist is nearest to (gt_x, gt_y).
    Returns (matched_pose, which_wrist_index, pixel_error_to_that_wrist)."""
    best: tuple[Any, int, float] | None = None
    for pose in poses_at_frame:
        for w_idx in (_WRIST_LEFT_HALPE, _WRIST_RIGHT_HALPE):
            if w_idx >= len(pose.keypoints):
                continue
            kp = pose.keypoints[w_idx]
            if kp.score < score_thr:
                continue
            d = float(np.hypot(kp.x - gt_x, kp.y - gt_y))
            if best is None or d < best[2]:
                best = (pose, w_idx, d)
    return best


def evaluate_stage2_wrist(
    pose_a: Pose2DSequence,
    pose_b: Pose2DSequence,
    shots_gt: dict[str, Any],
    pck_threshold_frac: float = 0.05,
) -> dict[str, Any]:
    """For each labeled shot, match each GT-clicked wrist to a predicted wrist
    and record pixel error + PCK@thr (normalised by GT-matched-pose bbox diagonal)."""
    a_by_frame = _pose2d_by_frame(pose_a)
    b_by_frame = _pose2d_by_frame(pose_b)

    per_shot: list[dict[str, Any]] = []
    errs_a: list[float] = []
    errs_b: list[float] = []
    pck_correct_a, pck_total_a = 0, 0
    pck_correct_b, pck_total_b = 0, 0

    fields = [
        ("shooter_wrist", "shooter_wrist_a", "shooter_wrist_b"),
        ("defender_wrist1", "defender_wrist1_a", "defender_wrist1_b"),
        ("defender_wrist2", "defender_wrist2_a", "defender_wrist2_b"),
    ]

    for shot in shots_gt["shots"]:
        fa, fb = shot["release_frame_a"], shot["release_frame_b"]
        a_poses = a_by_frame.get(fa, [])
        b_poses = b_by_frame.get(fb, [])
        row: dict[str, Any] = {"shot_id": shot["shot_id"]}

        for label, gt_a_key, gt_b_key in fields:
            for cam, poses, gt_pt, err_list in [
                ("a", a_poses, shot.get(gt_a_key), errs_a),
                ("b", b_poses, shot.get(gt_b_key), errs_b),
            ]:
                if gt_pt is None:
                    row[f"{label}_{cam}"] = None
                    continue
                gx, gy = gt_pt
                matched = _match_wrist_pose(poses, gx, gy)
                if matched is None:
                    row[f"{label}_{cam}"] = {"err_px": None, "track_id": None, "wrist": None, "pck_pass": None}
                    continue
                pose, w_idx, err = matched
                bbox = _bbox_from_keypoints(pose.keypoints)
                diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])) if bbox else 0.0
                pck_pass = (err <= pck_threshold_frac * diag) if diag > 0 else False
                row[f"{label}_{cam}"] = {
                    "err_px": round(err, 2),
                    "track_id": pose.track_id,
                    "wrist": "L" if w_idx == _WRIST_LEFT_HALPE else "R",
                    "pose_bbox_diag_px": round(diag, 1),
                    "pck_pass": bool(pck_pass),
                }
                err_list.append(err)
                if cam == "a":
                    pck_total_a += 1; pck_correct_a += int(pck_pass)
                else:
                    pck_total_b += 1; pck_correct_b += int(pck_pass)
        per_shot.append(row)

    summary = {
        "n_shots": len(shots_gt["shots"]),
        "pck_threshold_frac": pck_threshold_frac,
        "cam_a": {
            "n_clicks": pck_total_a,
            "wrist_pck": round(pck_correct_a / pck_total_a, 4) if pck_total_a else None,
            "mean_err_px": round(float(np.mean(errs_a)), 2) if errs_a else None,
            "median_err_px": round(float(np.median(errs_a)), 2) if errs_a else None,
            "max_err_px": round(float(np.max(errs_a)), 2) if errs_a else None,
        },
        "cam_b": {
            "n_clicks": pck_total_b,
            "wrist_pck": round(pck_correct_b / pck_total_b, 4) if pck_total_b else None,
            "mean_err_px": round(float(np.mean(errs_b)), 2) if errs_b else None,
            "median_err_px": round(float(np.median(errs_b)), 2) if errs_b else None,
            "max_err_px": round(float(np.max(errs_b)), 2) if errs_b else None,
        },
    }
    return {"summary": summary, "per_shot": per_shot}


def _undistort(x: float, y: float, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pt = np.array([[[x, y]]], dtype=np.float64)
    und = cv2.undistortPoints(pt, K, dist, P=K)
    return np.array([float(und[0, 0, 0]), float(und[0, 0, 1])]).reshape(2, 1)


def _triangulate_one(
    xa: float, ya: float, P_a: np.ndarray, K_a: np.ndarray, dist_a: np.ndarray,
    xb: float, yb: float, P_b: np.ndarray, K_b: np.ndarray, dist_b: np.ndarray,
) -> np.ndarray:
    ua = _undistort(xa, ya, K_a, dist_a)
    ub = _undistort(xb, yb, K_b, dist_b)
    X_h = cv2.triangulatePoints(P_a, P_b, ua, ub)
    return (X_h[:3] / X_h[3]).ravel()


def evaluate_stage4_wrist_3d(
    pose3d: Pose3DSequence,
    shots_gt: dict[str, Any],
    calib_a: dict[str, np.ndarray],
    calib_b: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Triangulate GT clicks into 3D wrists, compare to predicted 3D wrists."""
    p3_by_frame = _pose3d_by_frame(pose3d)
    per_shot: list[dict[str, Any]] = []
    errs_cm: list[float] = []

    for shot in shots_gt["shots"]:
        fa = shot["release_frame_a"]
        # GT wrist (triangulated from A+B click)
        ga = shot.get("shooter_wrist_a")
        gb = shot.get("shooter_wrist_b")
        row: dict[str, Any] = {"shot_id": shot["shot_id"]}
        if ga is None or gb is None:
            row["gt_3d_wrist_m"] = None
            row["err_cm"] = None
            per_shot.append(row)
            continue
        gt3d = _triangulate_one(
            ga[0], ga[1], calib_a["P"], calib_a["K"], calib_a["dist"],
            gb[0], gb[1], calib_b["P"], calib_b["K"], calib_b["dist"],
        )
        row["gt_3d_wrist_m"] = [round(float(v), 4) for v in gt3d]
        # Predicted 3D wrist — pick the predicted pose whose closest wrist (L or R) is nearest to gt3d
        preds = p3_by_frame.get(fa, [])
        best: tuple[Any, int, float] | None = None
        for p3 in preds:
            for w_idx in (_WRIST_LEFT_HALPE, _WRIST_RIGHT_HALPE):
                if w_idx >= len(p3.keypoints):
                    continue
                kp = p3.keypoints[w_idx]
                if kp.n_cameras < 2:  # only consider triangulated keypoints
                    continue
                pred = np.array([kp.x, kp.y, kp.z])
                d = float(np.linalg.norm(pred - gt3d))
                if best is None or d < best[2]:
                    best = (p3, w_idx, d)
        if best is None:
            row["pred_3d_wrist_m"] = None
            row["err_cm"] = None
            per_shot.append(row)
            continue
        p3, w_idx, err = best
        kp = p3.keypoints[w_idx]
        row["pred_3d_wrist_m"] = [round(float(kp.x), 4), round(float(kp.y), 4), round(float(kp.z), 4)]
        row["matched_track_id"] = p3.track_id
        row["matched_wrist"] = "L" if w_idx == _WRIST_LEFT_HALPE else "R"
        row["err_cm"] = round(err * 100, 2)
        errs_cm.append(err * 100)
        per_shot.append(row)

    summary = {
        "n_evaluated": len(errs_cm),
        "n_skipped_missing_gt": len(per_shot) - len(errs_cm),
        "mean_err_cm": round(float(np.mean(errs_cm)), 2) if errs_cm else None,
        "median_err_cm": round(float(np.median(errs_cm)), 2) if errs_cm else None,
        "p90_err_cm": round(float(np.percentile(errs_cm, 90)), 2) if errs_cm else None,
        "max_err_cm": round(float(np.max(errs_cm)), 2) if errs_cm else None,
    }
    return {"summary": summary, "per_shot": per_shot}


def _dedup_within_camera(events: list[dict], frame_window: int = 30) -> list[dict]:
    """Merge events within ``frame_window`` frames into the highest-confidence one."""
    events = sorted(events, key=lambda e: e["frame_idx"])
    out: list[dict] = []
    for e in events:
        if out and (e["frame_idx"] - out[-1]["frame_idx"]) <= frame_window:
            if e["p_shoot"] > out[-1]["p_shoot"]:
                # Replace with the higher-confidence event (keep this frame as the canonical one)
                merged = {**e, "merged_from": out[-1].get("merged_from", []) + [out[-1]["frame_idx"]]}
                out[-1] = merged
            else:
                out[-1].setdefault("merged_from", []).append(e["frame_idx"])
            continue
        out.append({**e, "merged_from": []})
    return out


def fuse_stage5_events(
    events_a: list[dict], events_b: list[dict],
    drift_slope_b_per_a: float,
    intra_window: int = 30,
    cross_window: int = 30,
) -> list[dict]:
    """Within-camera dedup + cross-camera fusion. Output uses A-clock frame indices."""
    a_dedup = _dedup_within_camera(events_a, intra_window)
    # Translate B's events to A's clock
    b_a_clock = [{**e, "frame_idx": int(round(e["frame_idx"] / drift_slope_b_per_a))} for e in events_b]
    b_dedup = _dedup_within_camera(b_a_clock, intra_window)

    # Cross-camera fusion — greedy nearest within cross_window
    used_b: set[int] = set()
    fused: list[dict] = []
    for a in a_dedup:
        best_i, best_d = -1, cross_window + 1
        for i, b in enumerate(b_dedup):
            if i in used_b: continue
            d = abs(a["frame_idx"] - b["frame_idx"])
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            used_b.add(best_i)
            b = b_dedup[best_i]
            fused.append({
                "frame_idx_a": a["frame_idx"],
                "frame_idx_b_in_a_clock": b["frame_idx"],
                "cams": ["a", "b"],
                "p_shoot_a": a["p_shoot"],
                "p_shoot_b": b["p_shoot"],
                "p_shoot_avg": round((a["p_shoot"] + b["p_shoot"]) / 2, 4),
            })
        else:
            fused.append({
                "frame_idx_a": a["frame_idx"],
                "cams": ["a"],
                "p_shoot_a": a["p_shoot"],
                "p_shoot_avg": a["p_shoot"],
            })
    for i, b in enumerate(b_dedup):
        if i in used_b: continue
        fused.append({
            "frame_idx_a": b["frame_idx"],  # already in A clock
            "cams": ["b"],
            "p_shoot_b": b["p_shoot"],
            "p_shoot_avg": b["p_shoot"],
        })
    fused.sort(key=lambda e: e["frame_idx_a"])
    return fused


def evaluate_stage5(
    shotevents_a: dict, shotevents_b: dict, drift_slope_b_per_a: float,
    shots_gt: dict, tolerance_frames: int = 30,
) -> dict:
    """Late-fuse two-camera events, score precision/recall vs GT release frames."""
    fused = fuse_stage5_events(
        shotevents_a["events"], shotevents_b["events"], drift_slope_b_per_a,
    )
    gt_frames = [s["release_frame_a"] for s in shots_gt["shots"]]
    gt_used: set[int] = set()
    pred_status: list[dict] = []
    for e in fused:
        best_gi, best_d = -1, tolerance_frames + 1
        for gi, gf in enumerate(gt_frames):
            if gi in gt_used: continue
            d = abs(e["frame_idx_a"] - gf)
            if d < best_d:
                best_d, best_gi = d, gi
        if best_gi >= 0:
            gt_used.add(best_gi)
            pred_status.append({**e, "matched_gt_shot_id": best_gi, "delta_frames": e["frame_idx_a"] - gt_frames[best_gi], "is_tp": True})
        else:
            pred_status.append({**e, "matched_gt_shot_id": None, "is_tp": False})

    per_shot_status: list[dict] = []
    for gi, gf in enumerate(gt_frames):
        matched = next((p for p in pred_status if p.get("matched_gt_shot_id") == gi), None)
        per_shot_status.append({
            "shot_id": gi,
            "gt_release_frame_a": gf,
            "detected": matched is not None,
            "predicted_frame_a": matched["frame_idx_a"] if matched else None,
            "delta_frames": matched["delta_frames"] if matched else None,
            "p_shoot_avg": matched["p_shoot_avg"] if matched else None,
            "cams_matched": matched["cams"] if matched else None,
        })

    tp = sum(1 for p in pred_status if p["is_tp"])
    fp = sum(1 for p in pred_status if not p["is_tp"])
    fn = len(gt_frames) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "summary": {
            "shoot_threshold": shotevents_a.get("shoot_threshold"),
            "tolerance_frames": tolerance_frames,
            "n_gt_shots": len(gt_frames),
            "n_predicted_events_after_fusion": len(fused),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        },
        "per_shot": per_shot_status,
        "predictions": pred_status,
    }


def _print_summary(stage2: dict, stage4: dict | None, stage5: dict | None = None) -> None:
    print("\n" + "=" * 78)
    print(" EVALUATION SUMMARY")
    print("=" * 78)
    s2 = stage2["summary"]
    print(f"\nStage 2 — wrist PCK@{s2['pck_threshold_frac']:.2f}  (n={s2['n_shots']} shots)")
    for cam in ("a", "b"):
        c = s2[f"cam_{cam}"]
        if c["wrist_pck"] is None:
            print(f"  cam_{cam}: no GT clicks"); continue
        print(f"  cam_{cam}: PCK = {c['wrist_pck']:.3f}  "
              f"({int(c['wrist_pck']*c['n_clicks'])}/{c['n_clicks']})  "
              f"mean {c['mean_err_px']:.1f} px  median {c['median_err_px']:.1f}  max {c['max_err_px']:.1f}")
    if stage4 is not None:
        s4 = stage4["summary"]
        print(f"\nStage 4 — 3D wrist accuracy  (n={s4['n_evaluated']}/{s4['n_evaluated']+s4['n_skipped_missing_gt']} shots)")
        if s4["mean_err_cm"] is not None:
            print(f"  mean {s4['mean_err_cm']:.2f} cm  median {s4['median_err_cm']:.2f}  "
                  f"P90 {s4['p90_err_cm']:.2f}  max {s4['max_err_cm']:.2f}")
    if stage5 is not None:
        s5 = stage5["summary"]
        print(f"\nStage 5 — shot detection  (thr={s5['shoot_threshold']}, ±{s5['tolerance_frames']} frame tolerance)")
        print(f"  Predicted events (after fusion): {s5['n_predicted_events_after_fusion']}  |  GT shots: {s5['n_gt_shots']}")
        print(f"  TP={s5['true_positives']}  FP={s5['false_positives']}  FN={s5['false_negatives']}")
        print(f"  precision={s5['precision']:.3f}  recall={s5['recall']:.3f}  F1={s5['f1']:.3f}")
        # Per-shot detection table
        print(f"  Per-shot:")
        for ps in stage5["per_shot"]:
            tag = "✓" if ps["detected"] else "✗"
            if ps["detected"]:
                print(f"    [{tag}] shot {ps['shot_id']:>2}: GT frame {ps['gt_release_frame_a']:>5}  "
                      f"pred frame {ps['predicted_frame_a']:>5}  Δ{ps['delta_frames']:+4d}  "
                      f"p={ps['p_shoot_avg']:.2f}  cams={ps['cams_matched']}")
            else:
                print(f"    [{tag}] shot {ps['shot_id']:>2}: GT frame {ps['gt_release_frame_a']:>5}  MISSED")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="2026-06-02_1on1")
    args = p.parse_args()

    out_dir = Path(f"outputs/sessions/{args.session}")
    gt = json.loads(Path(f"data/sessions/{args.session}/shots_gt.json").read_text())

    pose_a = read_json_as(Pose2DSequence, out_dir / "pose2d_cam_a.json")
    pose_b = read_json_as(Pose2DSequence, out_dir / "pose2d_cam_b.json")
    print(f"Loaded {len(pose_a.poses)} A poses, {len(pose_b.poses)} B poses, {gt['n_shots']} GT shots")

    stage2 = evaluate_stage2_wrist(pose_a, pose_b, gt)
    stage4 = None
    pose3d_path = out_dir / "pose3d.json"
    if pose3d_path.exists():
        pose3d = read_json_as(Pose3DSequence, pose3d_path)
        calib_a = _load_calib(
            Path("data/calibration/intrinsics_A_checker.json"),
            Path(f"data/sessions/{args.session}/extrinsics_A.json"),
        )
        calib_b = _load_calib(
            Path("data/calibration/intrinsics_B_checker.json"),
            Path(f"data/sessions/{args.session}/extrinsics_B.json"),
        )
        stage4 = evaluate_stage4_wrist_3d(pose3d, gt, calib_a, calib_b)
    else:
        print(f"⚠ {pose3d_path} not found — skipping Stage 4 evaluation")

    # Stage 5 — late-fusion + GT match (if Stage 5 outputs exist)
    stage5 = None
    sev_a = out_dir / "shotevents_cam_a.json"
    sev_b = out_dir / "shotevents_cam_b.json"
    if sev_a.exists() and sev_b.exists():
        sync = json.loads(Path(f"data/sessions/{args.session}/sync.json").read_text())
        events_a = json.loads(sev_a.read_text())
        events_b = json.loads(sev_b.read_text())
        stage5 = evaluate_stage5(
            events_a, events_b, sync["drift_slope_b_per_a"], gt,
        )
    else:
        print(f"⚠ Stage 5 events not found — skipping Stage 5 evaluation")

    out = {"session": args.session, "stage2_wrist_pck": stage2, "stage4_wrist_3d": stage4, "stage5_shot_detection": stage5}
    out_path = out_dir / "evaluation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n✓ Saved → {out_path}")
    _print_summary(stage2, stage4, stage5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
