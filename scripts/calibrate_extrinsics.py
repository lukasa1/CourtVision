"""Interactive extrinsic-calibration script — click backboard + floor landmarks.

World frame (right-handed):
    Origin: directly below rim center, on the floor.
    +x:     to the LEFT when facing the basket from the court (right-hand rule).
    +y:     INTO the court, away from the baseline.
    +z:     UP.

Standard NCAA college dimensions baked in:
    Backboard face is at y = -0.38 m (rim center is 0.38 m in front of it).
    Baseline is at y = -1.60 m (backboard is 1.22 m in front of baseline).
    Lane is 3.66 m wide (lines at x = ±1.83 m).
    Free-throw line is 4.572 m from backboard face → y_world = 4.192 m.
    Free-throw semicircle radius = 1.83 m (centered on FT line center).
    3-point arc radius = 6.75 m from rim center floor projection.
    Backboard is 1.83 m × 1.07 m (top at z = 3.96, bottom at z = 2.90).
    Rim is 0.46 m (18 in) inner diameter at z = 3.05 m.
    Inner ("shooter's") box: 0.61 m × 0.46 m, bottom at z = 3.05 (rim level).

Landmarks (click in order shown; you can skip any you can't see clearly):
    -- BACKBOARD (in the y = -0.38 m plane) --
    L01  BB_TL    backboard outer top-left corner (image-position)
    L02  BB_TR    backboard outer top-right corner
    L03  BB_BR    backboard outer bottom-right corner
    L04  BB_BL    backboard outer bottom-left corner
    L05  ISQ_TL   inner shooter's-box top-left
    L06  ISQ_TR   inner shooter's-box top-right
    L07  ISQ_BR   inner shooter's-box bottom-right (at rim height)
    L08  ISQ_BL   inner shooter's-box bottom-left (at rim height)
    L09  RIM      rim center (geometric center of the orange hoop's ellipse)
    -- FLOOR (z = 0 plane; far from backboard plane → great depth diversity) --
    L10  FT_LANE_TL  Top-left corner where the FT line meets the lane line
    L11  FT_LANE_TR  Top-right corner where the FT line meets the lane line
    L12  FT_CENTER   Center of the FT line (midpoint between L10 and L11)
    L13  SEMI_TOP    Top of the FT semicircle (court-side apex, 1.83 m past FT line)
    L14  ARC_TOP     Top of the 3-point arc (where arc crosses the y-axis on the court side)

Image-relative ordering for left/right: click them as you SEE them in the image
(image-top-left of the backboard is L01, regardless of which side of the court
the camera is on). The script auto-determines which physical world side each
clicked image-corner maps to by trying both sides and picking the lower-error
fit with a plausible camera position.

Usage:
    uv run python scripts/calibrate_extrinsics.py \
        --frame data/sessions/2026-06-02_1on1/frames/A_t040s.png \
        --intrinsics data/calibration/intrinsics_A_checker.json \
        --output data/sessions/2026-06-02_1on1/extrinsics_A.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Universal backboard + rim
BB_W_HALF, BB_TOP_Z, BB_BOT_Z = 0.915, 3.96, 2.90
ISQ_W_HALF, ISQ_TOP_Z, ISQ_BOT_Z = 0.305, 3.51, 3.05
RIM_Y, RIM_Z = 0.38, 3.05    # NBA spec: 6" front-edge + 9" rim radius = 15" = 0.38 m
BB_Y = -0.38                 # backboard face is BEHIND the rim center in our frame

# NCAA college dimensions
LANE_HALF = 1.83             # half of 12 ft (3.66 m)
FT_LINE_Y = 4.192            # 15 ft from backboard face = 4.572 m, minus 0.38 = 4.192
SEMI_RADIUS = 1.83           # FT semicircle radius
ARC_RADIUS = 6.75            # 3-point arc radius from rim center floor projection
BASELINE_Y = -1.60           # 4 ft from baseline to backboard face → baseline behind rim


def world_for_side(side: str) -> dict[str, np.ndarray]:
    """Map clicked image-position labels to 3D world coords given camera side.

    side='left':  image-left-of-board maps to +x world corner (camera at +x).
    side='right': image-left-of-board maps to -x world corner (camera at -x).
    """
    sx = +1.0 if side == "left" else -1.0
    return {
        # Backboard (8 coplanar at y = BB_Y)
        "BB_TL":  np.array([sx * BB_W_HALF, BB_Y, BB_TOP_Z]),
        "BB_TR":  np.array([-sx * BB_W_HALF, BB_Y, BB_TOP_Z]),
        "BB_BR":  np.array([-sx * BB_W_HALF, BB_Y, BB_BOT_Z]),
        "BB_BL":  np.array([sx * BB_W_HALF, BB_Y, BB_BOT_Z]),
        "ISQ_TL": np.array([sx * ISQ_W_HALF, BB_Y, ISQ_TOP_Z]),
        "ISQ_TR": np.array([-sx * ISQ_W_HALF, BB_Y, ISQ_TOP_Z]),
        "ISQ_BR": np.array([-sx * ISQ_W_HALF, BB_Y, ISQ_BOT_Z]),
        "ISQ_BL": np.array([sx * ISQ_W_HALF, BB_Y, ISQ_BOT_Z]),
        # Rim (off-plane at y = 0.38, z = 3.05)
        "RIM":    np.array([0.0, RIM_Y, RIM_Z]),
        # Floor (z = 0, far from backboard plane)
        "FT_LANE_TL": np.array([sx * LANE_HALF, FT_LINE_Y, 0.0]),
        "FT_LANE_TR": np.array([-sx * LANE_HALF, FT_LINE_Y, 0.0]),
        "FT_CENTER":  np.array([0.0, FT_LINE_Y, 0.0]),
        "SEMI_TOP":   np.array([0.0, FT_LINE_Y + SEMI_RADIUS, 0.0]),
        "ARC_TOP":    np.array([0.0, ARC_RADIUS, 0.0]),
    }


LANDMARK_ORDER = [
    "BB_TL", "BB_TR", "BB_BR", "BB_BL",
    "ISQ_TL", "ISQ_TR", "ISQ_BR", "ISQ_BL",
    "RIM",
    "FT_LANE_TL", "FT_LANE_TR", "FT_CENTER", "SEMI_TOP", "ARC_TOP",
]

LANDMARK_HINTS = {
    "BB_TL":      "Backboard OUTER TOP-LEFT corner (as seen in image)",
    "BB_TR":      "Backboard OUTER TOP-RIGHT corner",
    "BB_BR":      "Backboard OUTER BOTTOM-RIGHT corner",
    "BB_BL":      "Backboard OUTER BOTTOM-LEFT corner",
    "ISQ_TL":     "Inner shooter's-box TOP-LEFT corner",
    "ISQ_TR":     "Inner shooter's-box TOP-RIGHT corner",
    "ISQ_BR":     "Inner shooter's-box BOTTOM-RIGHT corner (at rim level)",
    "ISQ_BL":     "Inner shooter's-box BOTTOM-LEFT corner (at rim level)",
    "RIM":        "Rim CENTER (middle of the orange hoop's ellipse)",
    "FT_LANE_TL": "FLOOR: where FT-line meets LEFT lane line (court-side corner of the key)",
    "FT_LANE_TR": "FLOOR: where FT-line meets RIGHT lane line",
    "FT_CENTER":  "FLOOR: midpoint of the FT-line (between lane lines)",
    "SEMI_TOP":   "FLOOR: top of the FT semicircle on the court side (the 'D' arc apex)",
    "ARC_TOP":    "FLOOR: top of the 3-point arc on the court side (where arc crosses straight out from basket)",
}


SOLVERS_FOR_BACKBOARD_ONLY = [
    (cv2.SOLVEPNP_IPPE,  "IPPE"),   # designed for coplanar
    (cv2.SOLVEPNP_SQPNP, "SQPNP"),
    (cv2.SOLVEPNP_EPNP,  "EPNP"),
]

SOLVERS_GENERAL = [
    (cv2.SOLVEPNP_SQPNP, "SQPNP"),
    (cv2.SOLVEPNP_EPNP,  "EPNP"),
    (cv2.SOLVEPNP_ITERATIVE, "ITER"),
]


class LandmarkClicker:
    def __init__(self, img: np.ndarray, label_order: list[str], hints: dict[str, str]):
        self.img = img
        self.label_order = label_order
        self.hints = hints
        self.clicks: dict[str, tuple[float, float]] = {}
        self.skipped: set[str] = set()
        self.fig, self.ax = plt.subplots(figsize=(8, 14))
        self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.update_title()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def next_label(self) -> str | None:
        for lbl in self.label_order:
            if lbl not in self.clicks and lbl not in self.skipped:
                return lbl
        return None

    def update_title(self) -> None:
        nxt = self.next_label()
        placed = len(self.clicks)
        if nxt is None:
            self.ax.set_title(
                f"DONE: {placed} placed, {len(self.skipped)} skipped. "
                "Press SPACE to compute solvePnP. R-click = undo last."
            )
        else:
            self.ax.set_title(
                f"#{placed+1}/{len(self.label_order)} — {nxt}: {self.hints[nxt]}\n"
                "L-click = place. R-click = undo. 's' = skip this one. SPACE = done (min 6)."
            )
        self.fig.canvas.draw_idle()

    def on_click(self, event: Any) -> None:
        if event.inaxes is not self.ax:
            return
        if event.button == 1:
            nxt = self.next_label()
            if nxt is None: return
            self.clicks[nxt] = (float(event.xdata), float(event.ydata))
            self.ax.plot(event.xdata, event.ydata, "ro", ms=6)
            self.ax.text(event.xdata + 8, event.ydata - 8, nxt, color="red",
                         fontsize=8, fontweight="bold")
            self.update_title()
        elif event.button == 3:
            if self.clicks:
                last = list(self.clicks.keys())[-1]
                del self.clicks[last]
                self._redraw()

    def on_key(self, event: Any) -> None:
        if event.key == "s":
            nxt = self.next_label()
            if nxt is not None:
                self.skipped.add(nxt)
                self.update_title()
        elif event.key in (" ", "enter"):
            plt.close(self.fig)

    def _redraw(self) -> None:
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        for lbl, (x, y) in self.clicks.items():
            self.ax.plot(x, y, "ro", ms=6)
            self.ax.text(x + 8, y - 8, lbl, color="red", fontsize=8, fontweight="bold")
        self.update_title()


def plausibility_score(cam_pos: np.ndarray) -> tuple[bool, str]:
    """Sanity-check a camera position. Returns (is_plausible, reason)."""
    x, y, z = cam_pos
    if not (0.0 < z < 5.0):
        return False, f"height z={z:.2f} m out of [0, 5]"
    if y < -3.0:
        return False, f"y={y:.2f} m — camera is too far behind the baseline"
    if y > 15.0:
        return False, f"y={y:.2f} m — camera is impossibly far into the court"
    if abs(x) > 15.0:
        return False, f"|x|={abs(x):.2f} m — camera is too far to the side"
    return True, ""


def fit_one(img_pts: np.ndarray, obj_pts: np.ndarray, K: np.ndarray, dist: np.ndarray,
            solvers: list[tuple[int, str]]) -> tuple[dict | None, list[dict]]:
    """Try every solver, return best by reprojection error + list of all attempts."""
    attempts = []
    best = None
    n_pts = len(img_pts)
    for flag, name in solvers:
        # IPPE needs ≥ 4 coplanar points; skip if we have non-coplanar set
        if flag == cv2.SOLVEPNP_IPPE and n_pts < 4:
            continue
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flag)
        except cv2.error:
            ok = False; rvec = tvec = None
        if not ok:
            attempts.append({"solver": name, "status": "failed"})
            continue
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        per_pt = np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)
        mean = float(per_pt.mean())
        R, _ = cv2.Rodrigues(rvec)
        cam_pos = (-R.T @ tvec).ravel()
        is_plausible, reason = plausibility_score(cam_pos)
        attempts.append({
            "solver": name, "status": "ok",
            "mean_err_px": mean, "max_err_px": float(per_pt.max()),
            "cam_pos": cam_pos.tolist(),
            "plausible": is_plausible,
            "implausibility_reason": reason,
        })
        if not is_plausible:
            continue
        cur = {
            "solver": name, "rvec": rvec, "tvec": tvec, "R": R,
            "cam_pos": cam_pos, "per_pt": per_pt, "mean": mean,
        }
        if best is None or mean < best["mean"]:
            best = cur
    return best, attempts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame", required=True, type=Path)
    p.add_argument("--intrinsics", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    intr = json.loads(args.intrinsics.read_text())
    K = np.array(intr["K"], dtype=np.float64)
    dist = np.array(intr["dist"], dtype=np.float64).ravel()

    img = cv2.imread(str(args.frame))
    if img is None:
        print(f"⚠ Cannot read {args.frame}"); return 1
    h, w = img.shape[:2]

    print(f"Opening clicker for {args.frame.name} ({w}x{h}, intrinsics RMS {intr.get('rms_reprojection_error_px', '?')} px)")
    print("Controls: left-click=place, right-click=undo, 's'=skip current landmark, SPACE/ENTER=done")
    print("Click whichever landmarks are CLEARLY visible. Skip the rest. Minimum 6 total.")

    clicker = LandmarkClicker(img, LANDMARK_ORDER, LANDMARK_HINTS)
    plt.show()

    if len(clicker.clicks) < 6:
        print(f"⚠ Only {len(clicker.clicks)} clicked; need ≥ 6 for stable PnP.")
        return 1

    img_pts_list, labels_used = [], []
    for lbl in LANDMARK_ORDER:
        if lbl in clicker.clicks:
            img_pts_list.append(list(clicker.clicks[lbl]))
            labels_used.append(lbl)
    img_pts = np.array(img_pts_list, dtype=np.float64).reshape(-1, 1, 2)
    print(f"\nUsing {len(labels_used)} landmarks: {labels_used}")

    has_off_plane = any(lbl in {"RIM", "FT_LANE_TL", "FT_LANE_TR", "FT_CENTER", "SEMI_TOP", "ARC_TOP"} for lbl in labels_used)
    has_floor = any(lbl.startswith("FT_") or lbl in {"SEMI_TOP", "ARC_TOP"} for lbl in labels_used)
    solvers = SOLVERS_GENERAL if has_off_plane else SOLVERS_FOR_BACKBOARD_ONLY
    if not has_off_plane:
        print("⚠ All clicked landmarks are coplanar (backboard only) — PnP will have 2-fold ambiguity.")
    if has_floor:
        print(f"✓ Floor landmarks included — depth conditioning is good.")

    # Try both sides; for each, try all solvers
    print(f"\nTrying both side conventions × {len(solvers)} solvers...")
    results = {}
    for side in ["left", "right"]:
        world_lut = world_for_side(side)
        obj_pts = np.array([world_lut[lbl] for lbl in labels_used], dtype=np.float64).reshape(-1, 1, 3)
        best, attempts = fit_one(img_pts, obj_pts, K, dist, solvers)
        results[side] = {"best": best, "attempts": attempts}
        if best is not None:
            print(f"  side={side}: best={best['solver']} mean {best['mean']:.2f} px, cam_pos=({best['cam_pos'][0]:+.2f}, {best['cam_pos'][1]:+.2f}, {best['cam_pos'][2]:+.2f})")
        else:
            print(f"  side={side}: no plausible fit found")
            for a in attempts:
                print(f"    {a}")

    plausible_options = [(side, r["best"]) for side, r in results.items() if r["best"] is not None]
    if not plausible_options:
        print("\n⚠ No plausible camera pose found from any solver/side combo!")
        print("    Either click more landmarks (especially floor ones), or check that your")
        print("    intrinsics were computed from the same rotated-frame orientation.")
        return 1
    chosen_side, best = min(plausible_options, key=lambda x: x[1]["mean"])
    world_lut = world_for_side(chosen_side)

    print(f"\n→ CHOSEN: side={chosen_side} solver={best['solver']}")
    print(f"  Mean reprojection error: {best['mean']:.2f} px")
    print(f"  Max reprojection error:  {float(best['per_pt'].max()):.2f} px")
    print(f"  Camera position (world m): ({best['cam_pos'][0]:+.2f}, {best['cam_pos'][1]:+.2f}, {best['cam_pos'][2]:+.2f})")
    print(f"  Per-landmark errors:")
    for lbl, err in zip(labels_used, best["per_pt"]):
        print(f"    {lbl:12s}: {err:6.2f} px")

    # Save
    out = {
        "frame": str(args.frame),
        "intrinsics_source": str(args.intrinsics),
        "side": chosen_side,
        "solver": best["solver"],
        "world_frame": "origin=floor-under-rim-center, +x=left-facing-basket, +y=into-court, +z=up",
        "world_dimensions_source": "NCAA college standard",
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "resolution": [w, h],
        "rvec": best["rvec"].ravel().tolist(),
        "tvec": best["tvec"].ravel().tolist(),
        "R": best["R"].tolist(),
        "camera_position_world_m": best["cam_pos"].tolist(),
        "labels": labels_used,
        "image_points": img_pts.reshape(-1, 2).tolist(),
        "world_points": [world_lut[lbl].tolist() for lbl in labels_used],
        "per_landmark_reprojection_error_px": best["per_pt"].tolist(),
        "mean_reprojection_error_px": best["mean"],
        "max_reprojection_error_px": float(best["per_pt"].max()),
        "skipped": sorted(clicker.skipped),
        "all_attempts": {s: r["attempts"] for s, r in results.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"  ✓ Saved → {args.output}")

    if best["mean"] < 3.0:
        print("  ✓✓ Mean < 3 px — excellent.")
    elif best["mean"] < 5.0:
        print("  ✓ Mean < 5 px — passes PRD §4 bar.")
    elif best["mean"] < 10.0:
        print("  ⚠ Mean in [5, 10] px — usable; consider re-clicking or adding floor landmarks.")
    else:
        print("  ⚠⚠ Mean ≥ 10 px — recheck clicks or court-dimension assumptions.")

    # Diagnostic image
    diag = img.copy()
    proj, _ = cv2.projectPoints(
        np.array([world_lut[lbl] for lbl in labels_used], dtype=np.float64).reshape(-1, 1, 3),
        best["rvec"], best["tvec"], K, dist,
    )
    img_pts_flat = img_pts.reshape(-1, 2)
    for (x, y), (px, py), lbl in zip(img_pts_flat, proj.reshape(-1, 2), labels_used):
        cv2.circle(diag, (int(x), int(y)), 8, (0, 0, 255), 2)
        cv2.circle(diag, (int(px), int(py)), 6, (0, 255, 0), 2)
        cv2.line(diag, (int(x), int(y)), (int(px), int(py)), (255, 255, 0), 1)
        cv2.putText(diag, lbl, (int(x) + 12, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    diag_path = args.output.with_suffix(".diag.png")
    cv2.imwrite(str(diag_path), diag)
    print(f"  ✓ Diagnostic → {diag_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
