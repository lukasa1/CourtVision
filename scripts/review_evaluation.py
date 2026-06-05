"""Render per-shot evaluation review PNGs.

For each labeled shot, produces a side-by-side image (camera A | camera B) at
the release frame with:
    - GT-clicked wrists drawn as red circles + label
    - Predicted wrists (from Stage 2) drawn as green circles
    - Yellow line between each GT/pred pair, with the pixel error printed

Plus a caption strip showing:
    shot id, shot type, make/miss, Stage 4 3D wrist error in cm

Run after scripts/evaluate_session.py has produced evaluation.json.

Usage:
    uv run python scripts/review_evaluation.py --session 2026-06-02_1on1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLOR_GT = (0, 0, 255)        # red (BGR) — clicked GT
COLOR_PRED = (0, 200, 0)      # green — predicted
COLOR_LINE = (0, 255, 255)    # yellow — connector


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    return img if ok else None


def find_pred_wrist(poses_at_frame: list, gt_x: float, gt_y: float, score_thr: float = 0.3) -> tuple[float, float] | None:
    """Return the (predicted_x, predicted_y) of the closest wrist (L or R) on any pose."""
    best: tuple[float, float, float] | None = None
    for pose in poses_at_frame:
        for w_idx in (9, 10):
            if w_idx >= len(pose["keypoints"]):
                continue
            kp = pose["keypoints"][w_idx]
            if kp["score"] < score_thr:
                continue
            d = float(np.hypot(kp["x"] - gt_x, kp["y"] - gt_y))
            if best is None or d < best[2]:
                best = (kp["x"], kp["y"], d)
    if best is None:
        return None
    return (best[0], best[1])


def draw_pair(img, gt_pt, pred_pt, label):
    if gt_pt is None:
        return
    gx, gy = int(round(gt_pt[0])), int(round(gt_pt[1]))
    cv2.circle(img, (gx, gy), 14, (255, 255, 255), 2)
    cv2.circle(img, (gx, gy), 9, COLOR_GT, -1)
    if pred_pt is not None:
        px, py = int(round(pred_pt[0])), int(round(pred_pt[1]))
        cv2.line(img, (gx, gy), (px, py), COLOR_LINE, 2)
        cv2.circle(img, (px, py), 12, (255, 255, 255), 2)
        cv2.circle(img, (px, py), 7, COLOR_PRED, -1)
        err = float(np.hypot(gx - px, gy - py))
        # Label text near GT point
        tx, ty = gx + 18, gy - 14
        text = f"{label}: {err:.1f}px"
    else:
        tx, ty = gx + 18, gy - 14
        text = f"{label}: NO PRED"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GT, 2)


def header(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def render_shot(shot, pose_a_by_frame, pose_b_by_frame, eval_per_shot, stage5_per_shot, stage6_per_shot, stage7_per_shot, cap_a, cap_b, out_path):
    fa, fb = shot["release_frame_a"], shot["release_frame_b"]
    img_a = read_frame(cap_a, fa)
    img_b = read_frame(cap_b, fb)
    if img_a is None or img_b is None:
        print(f"  ⚠ shot {shot['shot_id']}: could not read frame"); return False

    a_poses = pose_a_by_frame.get(fa, [])
    b_poses = pose_b_by_frame.get(fb, [])

    # Shooter + defender wrist clicks (GT) + predictions (closest predicted wrist)
    pairs = [
        ("S", shot.get("shooter_wrist_a"), shot.get("shooter_wrist_b")),
        ("D1", shot.get("defender_wrist1_a"), shot.get("defender_wrist1_b")),
        ("D2", shot.get("defender_wrist2_a"), shot.get("defender_wrist2_b")),
    ]
    for label, gt_a, gt_b in pairs:
        if gt_a is not None:
            pred_a = find_pred_wrist(a_poses, gt_a[0], gt_a[1])
            draw_pair(img_a, gt_a, pred_a, label)
        if gt_b is not None:
            pred_b = find_pred_wrist(b_poses, gt_b[0], gt_b[1])
            draw_pair(img_b, gt_b, pred_b, label)

    header(img_a, f"A  frame {fa}")
    header(img_b, f"B  frame {fb}")

    # Pad heights
    h = max(img_a.shape[0], img_b.shape[0])
    def pad(im, th):
        if im.shape[0] == th: return im
        pad_arr = np.zeros((th - im.shape[0], im.shape[1], 3), dtype=im.dtype)
        return np.vstack([im, pad_arr])
    img_a = pad(img_a, h); img_b = pad(img_b, h)
    side = np.hstack([img_a, img_b])

    # Caption strip — 6 lines: header, Stage 4, Stage 5/6, Stage 7 line A, Stage 7 line B, legend
    cap_h = 205
    caption = np.zeros((cap_h, side.shape[1], 3), dtype=np.uint8)
    miss_str = "MAKE" if shot["make"] else "MISS"
    miss_color = (0, 255, 0) if shot["make"] else (0, 0, 255)

    # Line 1 — shot header (well-spaced; MISS/MAKE in its own column)
    header_text = f"Shot {shot['shot_id']}   {shot['shot_type']}   {'UNCONTESTED' if shot['uncontested'] else 'contested'}"
    cv2.putText(caption, header_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2)
    cv2.putText(caption, miss_str, (560, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, miss_color, 2)

    # Line 2 — Stage 4 (3D wrist accuracy)
    s4 = eval_per_shot.get(shot["shot_id"], {})
    err_cm = s4.get("err_cm")
    if err_cm is not None:
        col = (0, 255, 0) if err_cm < 10 else ((0, 200, 255) if err_cm < 50 else (0, 0, 255))
        cv2.putText(caption, f"Stage 4 (3D wrist): {err_cm:.1f} cm", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2)
    else:
        cv2.putText(caption, "Stage 4 (3D wrist): -", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (160, 160, 160), 1)

    # Line 3 — Stage 5 (shot detection) + Stage 6 (release frame)
    s5 = stage5_per_shot.get(shot["shot_id"], {})
    if s5:
        if s5.get("detected"):
            df = s5.get("delta_frames", 0)
            ps = s5.get("p_shoot_avg", 0.0)
            cams = "+".join(s5.get("cams_matched") or [])
            cv2.putText(caption, f"Stage 5: DET  d={df:+d}f  p={ps:.2f}  cams={cams}",
                        (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
        else:
            cv2.putText(caption, "Stage 5: MISSED",
                        (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

    s6 = stage6_per_shot.get(shot["shot_id"], {})
    if s6:
        df6 = s6.get("delta_frames", 0)
        method = s6.get("method", "?")
        conf = s6.get("confidence", "?")
        # Color by magnitude of error (frames)
        abs_d = abs(df6) if df6 is not None else 999
        col6 = (0, 255, 0) if abs_d <= 5 else ((0, 200, 255) if abs_d <= 15 else (0, 0, 255))
        cv2.putText(caption, f"Stage 6: d={df6:+d}f  method={method}  conf={conf}",
                    (660, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.70, col6, 2)
    elif s5 and s5.get("detected"):
        cv2.putText(caption, "Stage 6: -",
                    (660, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (160, 160, 160), 1)

    # Line 4 — Stage 7 (5-measurement contest metric, main values)
    s7 = stage7_per_shot.get(shot["shot_id"], {})
    inst = (s7.get("pred") or {}).get("instantaneous") or {}
    has_inst = bool(s7) and any(inst.get(k) is not None for k in
                                ("wrist_to_wrist_distance_m", "wrist_height_advantage_m",
                                 "shooter_fade_mps", "floor_distance_m", "composite_z"))
    if has_inst:
        ww = inst.get("wrist_to_wrist_distance_m")
        ha = inst.get("wrist_height_advantage_m")
        fa = inst.get("shooter_fade_mps")
        fl = inst.get("floor_distance_m")
        cz = inst.get("composite_z")
        def _fmt(v, unit="cm", scale=100, dec=1):
            return f"{v*scale:+.{dec}f}{unit}" if v is not None else "--"
        text7a = (f"Stage 7: "
                  f"w-w={(ww*100):.1f}cm  " if ww is not None else "Stage 7: w-w=--  ")
        text7a += (f"hgt={ha*100:+.1f}cm  " if ha is not None else "hgt=--  ")
        text7a += (f"fade={fa:.2f}m/s  " if fa is not None else "fade=--  ")
        text7a += (f"floor={fl*100:.1f}cm  " if fl is not None else "floor=--  ")
        text7a += (f"comp={cz:+.2f}s" if cz is not None else "comp=--")
        # Color by composite magnitude — bigger = more contested
        col7 = (200, 200, 200)
        if cz is not None:
            if cz < -1.0:
                col7 = (0, 255, 0)   # uncontested → green
            elif cz < 1.0:
                col7 = (0, 200, 255) # medium → amber
            else:
                col7 = (0, 0, 255)   # heavily contested → red
        cv2.putText(caption, text7a, (12, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, col7, 2)
    else:
        cv2.putText(caption, "Stage 7: -", (12, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (160, 160, 160), 1)

    # Line 5 — Stage 7 errors vs GT + uncontested flag
    if s7:
        err_w = s7.get("err_wrist_to_wrist_m")
        err_h = s7.get("err_wrist_height_advantage_m")
        unc = (s7.get("pred") or {}).get("uncontested")
        parts = []
        if err_w is not None: parts.append(f"|err_w-w|={err_w*100:.1f}cm")
        if err_h is not None: parts.append(f"|err_hgt|={err_h*100:.1f}cm")
        if unc:
            parts.append("UNCONTESTED")
        if parts:
            col_err = (160, 160, 160)
            worst = max([e for e in (err_w, err_h) if e is not None], default=None)
            if worst is not None:
                col_err = (0, 255, 0) if worst <= 0.15 else ((0, 200, 255) if worst <= 0.30 else (0, 0, 255))
            cv2.putText(caption, f"vs GT:   {'   '.join(parts)}",
                        (12, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col_err, 2)

    # Line 6 — legend
    cv2.putText(caption, "red = GT click,  green = predicted wrist,  yellow line = error",
                (12, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    final = np.vstack([side, caption])
    if final.shape[1] > 2400:
        scale = 2400 / final.shape[1]
        final = cv2.resize(final, (2400, int(final.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="2026-06-02_1on1")
    p.add_argument("--out-subdir", default="eval_review",
                   help="Subdirectory under outputs/sessions/<session>/ to write PNGs into.")
    args = p.parse_args()

    out_dir = Path(f"outputs/sessions/{args.session}")
    gt = json.loads(Path(f"data/sessions/{args.session}/shots_gt.json").read_text())
    eval_data = json.loads((out_dir / "evaluation.json").read_text())

    # Build a per-shot index of Stage 4 results for the caption
    stage4_idx: dict[int, dict] = {}
    s4 = eval_data.get("stage4_wrist_3d")
    if s4 is not None:
        for row in s4["per_shot"]:
            stage4_idx[row["shot_id"]] = row
    # Per-shot Stage 5 detection status
    stage5_idx: dict[int, dict] = {}
    s5 = eval_data.get("stage5_shot_detection")
    if s5 is not None:
        for row in s5["per_shot"]:
            stage5_idx[row["shot_id"]] = row
    # Per-shot Stage 6 release-frame detection
    stage6_idx: dict[int, dict] = {}
    s6 = eval_data.get("stage6_release_frame")
    if s6 is not None:
        for row in s6["per_shot"]:
            stage6_idx[row["shot_id"]] = row
    # Per-shot Stage 7 contest metric
    stage7_idx: dict[int, dict] = {}
    s7 = eval_data.get("stage7_contest_metric")
    if s7 is not None:
        for row in s7.get("per_shot", []):
            stage7_idx[row["shot_id"]] = row

    pose_a = json.loads((out_dir / "pose2d_cam_a.json").read_text())
    pose_b = json.loads((out_dir / "pose2d_cam_b.json").read_text())
    pose_a_by_frame: dict[int, list] = {}
    for po in pose_a["poses"]:
        pose_a_by_frame.setdefault(po["frame_idx"], []).append(po)
    pose_b_by_frame: dict[int, list] = {}
    for po in pose_b["poses"]:
        pose_b_by_frame.setdefault(po["frame_idx"], []).append(po)

    video_a = Path(f"data/sessions/{args.session}/trimmed/A_trimmed.mp4")
    video_b = Path(f"data/sessions/{args.session}/trimmed/B_trimmed.mp4")
    cap_a = cv2.VideoCapture(str(video_a))
    cap_b = cv2.VideoCapture(str(video_b))

    review_dir = out_dir / args.out_subdir
    review_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(gt['shots'])} per-shot evaluation reviews → {review_dir}")
    for shot in gt["shots"]:
        ok = render_shot(shot, pose_a_by_frame, pose_b_by_frame, stage4_idx, stage5_idx, stage6_idx, stage7_idx,
                         cap_a, cap_b, review_dir / f"shot_{shot['shot_id']:03d}.png")
        if ok:
            print(f"  ✓ shot {shot['shot_id']}")
    cap_a.release(); cap_b.release()

    print(f"\n✓ Open one to inspect:")
    print(f"    open '{review_dir}/shot_000.png'")
    print(f"  or browse all:")
    print(f"    open '{review_dir}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
