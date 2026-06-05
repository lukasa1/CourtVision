"""3D skeletal reenactment with orbiting virtual camera, one clip per shot.

For each GT shot, renders a ~3-second 3D reenactment around the release frame:
    - Court reference (backboard, rim, lane, FT line, 3-pt arc) drawn from calibrated world coords
    - Both player skeletons (Halpe-26 wireframe) — shooter in ORANGE, defender in GRAY
    - "SHOT!" banner around the release frame (±5 frames)
    - Camera azimuth orbits 180° during the clip
    - Title plate with shot id, type, make/miss

Outputs:
    outputs/sessions/<session>/3d_reenactments/shot_NNN.mp4   (per-shot ~3s clips)
    outputs/sessions/<session>/3d_reenactments/highlight_reel.mp4   (all shots concatenated)

Usage:
    uv run python scripts/render_3d_reenactment.py --session 2026-06-02_1on1
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from courtvision.io_utils import read_json_as
from courtvision.schemas import Pose3DSequence


# Halpe-26 indices (consistent with the rest of the codebase)
_L_WRIST, _R_WRIST = 9, 10
_HALPE26_SKELETON: tuple[tuple[int, int], ...] = (
    # Face / head
    (0, 17), (17, 18), (18, 19),
    # Shoulders + arms
    (18, 5), (18, 6), (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    # Hips + torso
    (19, 11), (19, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16),
    # Feet
    (15, 20), (15, 22), (15, 24), (16, 21), (16, 23), (16, 25),
)


# --------- court geometry (high-school NCAA spec, matches our calibration) ----------

BB_W_HALF = 0.915
BB_TOP_Z, BB_BOT_Z = 3.96, 2.90
BB_Y = -0.38
RIM_Y, RIM_Z = 0.38, 3.05
RIM_R = 0.225
LANE_HALF = 1.83
FT_LINE_Y = 4.192
SEMI_R = 1.83
ARC_R = 6.02
BASELINE_Y = -1.60
SIDELINE_HALF = 7.62


def court_lines() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """List of (xs, ys, zs) polylines for court markings."""
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # Backboard rectangle (in y=BB_Y plane, vertical)
    bb_x = [BB_W_HALF, -BB_W_HALF, -BB_W_HALF, BB_W_HALF, BB_W_HALF]
    bb_y = [BB_Y] * 5
    bb_z = [BB_TOP_Z, BB_TOP_Z, BB_BOT_Z, BB_BOT_Z, BB_TOP_Z]
    out.append((np.array(bb_x), np.array(bb_y), np.array(bb_z)))

    # Rim (circle around (0, RIM_Y, RIM_Z), parallel to floor)
    ang = np.linspace(0, 2 * np.pi, 36)
    rim_x = RIM_R * np.cos(ang)
    rim_y = RIM_Y + RIM_R * np.sin(ang)
    rim_z = np.full_like(rim_x, RIM_Z)
    out.append((rim_x, rim_y, rim_z))

    # Rim attachment to backboard (small bar)
    out.append((np.array([0, 0]), np.array([BB_Y, RIM_Y - RIM_R]), np.array([RIM_Z, RIM_Z])))

    # Baseline
    out.append((np.array([-SIDELINE_HALF, SIDELINE_HALF]),
                np.array([BASELINE_Y, BASELINE_Y]), np.array([0.0, 0.0])))
    # Sidelines (only show short portion)
    out.append((np.array([-SIDELINE_HALF, -SIDELINE_HALF]),
                np.array([BASELINE_Y, BASELINE_Y + 14.33]), np.array([0.0, 0.0])))
    out.append((np.array([SIDELINE_HALF, SIDELINE_HALF]),
                np.array([BASELINE_Y, BASELINE_Y + 14.33]), np.array([0.0, 0.0])))
    # Half-court line
    out.append((np.array([-SIDELINE_HALF, SIDELINE_HALF]),
                np.array([BASELINE_Y + 14.33, BASELINE_Y + 14.33]), np.array([0.0, 0.0])))

    # Lane lines (the key)
    out.append((np.array([LANE_HALF, LANE_HALF]),
                np.array([BASELINE_Y, FT_LINE_Y]), np.array([0.0, 0.0])))
    out.append((np.array([-LANE_HALF, -LANE_HALF]),
                np.array([BASELINE_Y, FT_LINE_Y]), np.array([0.0, 0.0])))
    # FT line
    out.append((np.array([-LANE_HALF, LANE_HALF]),
                np.array([FT_LINE_Y, FT_LINE_Y]), np.array([0.0, 0.0])))

    # FT semicircle (on court side of FT line)
    semi_t = np.linspace(0, np.pi, 36)
    semi_x = SEMI_R * np.cos(semi_t)
    semi_y = FT_LINE_Y + SEMI_R * np.sin(semi_t)
    semi_z = np.zeros_like(semi_x)
    out.append((semi_x, semi_y, semi_z))

    # 3-point arc (on court side of baseline) — high-school spec radius 6.02
    arc_t = np.linspace(0, np.pi, 60)
    arc_x = ARC_R * np.cos(arc_t)
    arc_y = ARC_R * np.sin(arc_t)
    arc_z = np.zeros_like(arc_x)
    out.append((arc_x, arc_y, arc_z))

    return out


def draw_court(ax) -> None:
    for xs, ys, zs in court_lines():
        ax.plot(xs, ys, zs, color="#8a6f3c", lw=1.2, alpha=0.75)


def draw_skeleton(ax, pose, color: str, alpha: float = 1.0) -> None:
    """Draw one Halpe-26 skeleton in 3D. Skip points that aren't triangulated."""
    valid = [(kp.x, kp.y, kp.z) if kp.n_cameras >= 2 else None for kp in pose.keypoints]
    # Bones
    for a, b in _HALPE26_SKELETON:
        if a >= len(valid) or b >= len(valid): continue
        if valid[a] is None or valid[b] is None: continue
        xs = [valid[a][0], valid[b][0]]
        ys = [valid[a][1], valid[b][1]]
        zs = [valid[a][2], valid[b][2]]
        ax.plot(xs, ys, zs, color=color, lw=2.5, alpha=alpha, solid_capstyle="round")
    # Wrist dots (highlight)
    for w_idx in (_L_WRIST, _R_WRIST):
        if w_idx < len(valid) and valid[w_idx] is not None:
            ax.scatter([valid[w_idx][0]], [valid[w_idx][1]], [valid[w_idx][2]],
                       s=60, c="red", edgecolors="white", linewidth=0.8, zorder=10)


def setup_axes(ax, azim: float, elev: float = 22.0) -> None:
    """World-frame bounds. Tight on the shot area (basket + FT line region).
    True world aspect kept via box_aspect."""
    x_half = 4.5
    y_lo, y_hi = BASELINE_Y - 0.3, 6.5  # ~7 m of depth around the shot area
    z_hi = 4.2
    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(0, z_hi)
    ax.set_box_aspect((2 * x_half, y_hi - y_lo, z_hi))  # true metres
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)


def identify_shooter_in_window(pose3d_seq, center_frame, half_window):
    """Return shooter track_id by the player whose wrist peaks highest in Z."""
    lo, hi = center_frame - half_window, center_frame + half_window
    best = None
    for p in pose3d_seq.poses:
        if not (lo <= p.frame_idx < hi): continue
        for w in (_L_WRIST, _R_WRIST):
            kp = p.keypoints[w]
            if kp.n_cameras < 2: continue
            if best is None or kp.z > best[0]:
                best = (kp.z, p.track_id)
    return best[1] if best else None


def render_shot_clip(
    shot: dict, pose3d_seq, fps_a: float, out_path: Path,
    half_window_frames: int = 90, dpi: int = 110,
    metric_panel: dict | None = None,
) -> bool:
    center = shot["release_frame_a"]
    lo, hi = center - half_window_frames, center + half_window_frames
    shooter_tid = identify_shooter_in_window(pose3d_seq, center, half_window_frames)

    # Group poses by frame for fast lookup
    by_frame: dict[int, list] = {}
    for p in pose3d_seq.poses:
        if lo <= p.frame_idx < hi:
            by_frame.setdefault(p.frame_idx, []).append(p)
    if not by_frame:
        print(f"  ⚠ shot {shot['shot_id']}: no 3D pose data in window — skipping")
        return False

    fig = plt.figure(figsize=(10, 9), dpi=dpi, facecolor="#101010")
    ax = fig.add_subplot(111, projection="3d", facecolor="#101010")
    plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)

    writer = FFMpegWriter(fps=int(round(fps_a)), bitrate=4000, codec="libx264")
    with writer.saving(fig, str(out_path), dpi=dpi):
        for f in range(lo, hi):
            ax.clear()
            # Camera: azimuth orbits 180° over the clip
            t = (f - lo) / max(1, (hi - lo - 1))
            azim = 50.0 + t * 180.0
            setup_axes(ax, azim=azim)
            draw_court(ax)
            # Shoot indicator near the actual release frame
            shooting_now = abs(f - center) <= 5
            shooter_pose = None
            defender_pose = None
            for pose in by_frame.get(f, []):
                if pose.track_id == shooter_tid:
                    color = "#ffaa00" if shooting_now else "#ff7733"
                    alpha = 1.0
                    shooter_pose = pose
                else:
                    color = "#bfbfbf"
                    alpha = 0.85
                    defender_pose = pose
                draw_skeleton(ax, pose, color=color, alpha=alpha)

            # Contest-distance line at the shot moment — dashed line from shooter's
            # release wrist (higher-Z wrist) to defender's closest wrist + label.
            if shooting_now and shooter_pose is not None and defender_pose is not None:
                lw = shooter_pose.keypoints[_L_WRIST]
                rw = shooter_pose.keypoints[_R_WRIST]
                shooter_wrist = None
                if lw.n_cameras >= 2 and rw.n_cameras >= 2:
                    shooter_wrist = (rw.x, rw.y, rw.z) if rw.z > lw.z else (lw.x, lw.y, lw.z)
                elif lw.n_cameras >= 2:
                    shooter_wrist = (lw.x, lw.y, lw.z)
                elif rw.n_cameras >= 2:
                    shooter_wrist = (rw.x, rw.y, rw.z)
                if shooter_wrist is not None:
                    sx, sy, sz = shooter_wrist
                    best: tuple[float, tuple[float, float, float]] | None = None
                    for w_idx in (_L_WRIST, _R_WRIST):
                        kp = defender_pose.keypoints[w_idx]
                        if kp.n_cameras < 2:
                            continue
                        dist = float(np.linalg.norm(np.array([kp.x - sx, kp.y - sy, kp.z - sz])))
                        if best is None or dist < best[0]:
                            best = (dist, (kp.x, kp.y, kp.z))
                    if best is not None:
                        d_cm, (dx, dy, dz) = best[0] * 100, best[1]
                        ax.plot([sx, dx], [sy, dy], [sz, dz],
                                color="#ffd24d", linestyle="--", lw=2.4, alpha=0.95)
                        # Endpoint dots so the line is anchored visually
                        ax.scatter([sx, dx], [sy, dy], [sz, dz],
                                   s=40, c="#ffd24d", edgecolors="white", linewidth=0.6, zorder=11)
                        mx, my, mz = (sx + dx) / 2, (sy + dy) / 2, (sz + dz) / 2 + 0.18
                        ax.text(mx, my, mz, f"{d_cm:.0f} cm",
                                color="#ffd24d", fontsize=13, fontweight="bold",
                                ha="center", zorder=12,
                                bbox=dict(facecolor="black", alpha=0.55,
                                          edgecolor="#ffd24d", boxstyle="round,pad=0.2"))
            # Title + status overlay
            title = (
                f"Shot {shot['shot_id']}   {shot['shot_type']}   "
                f"{'MAKE' if shot['make'] else 'MISS'}   release frame {center}"
            )
            ax.set_title(title, color="white", fontsize=13, pad=8)
            if shooting_now:
                ax.text2D(0.5, 0.94, "SHOT!", transform=ax.transAxes,
                          ha="center", color="#ffd24d", fontsize=22, fontweight="bold")
            # Metric panel — display all 5 contest metrics during the shot moment
            if shooting_now and metric_panel is not None:
                lines = ["CONTEST METRICS"]
                def _fmt(label, val, unit, scale=1.0, decimals=2, signed=False):
                    if val is None:
                        return f"  {label:<14s} --"
                    v = val * scale
                    sign_fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
                    return f"  {label:<14s} {sign_fmt.format(v):>7s} {unit}"
                lines.append(_fmt("wrist-to-wrist", metric_panel.get("wrist_to_wrist_m"), "m",
                                  decimals=2, signed=False))
                lines.append(_fmt("height adv",     metric_panel.get("wrist_height_advantage_m"), "m",
                                  decimals=2, signed=True))
                lines.append(_fmt("shooter fade",   metric_panel.get("shooter_fade_mps"), "m/s",
                                  decimals=2, signed=False))
                lines.append(_fmt("floor dist",     metric_panel.get("floor_distance_m"), "m",
                                  decimals=2, signed=False))
                cz = metric_panel.get("composite_z")
                if cz is None:
                    lines.append("  composite       --")
                else:
                    interp = ("uncontested" if cz < -1.0 else
                              "medium     " if cz < 1.0 else
                              "CONTESTED  ")
                    lines.append(f"  composite      {cz:+5.2f} sigma  ({interp.strip()})")
                panel_text = "\n".join(lines)
                # Color-code by composite z-score
                if cz is None:
                    edge = "#aaaaaa"
                elif cz < -1.0:
                    edge = "#5ee06c"  # green — uncontested
                elif cz < 1.0:
                    edge = "#ffd24d"  # amber — medium
                else:
                    edge = "#e0625e"  # red — heavily contested
                ax.text2D(0.02, 0.30, panel_text, transform=ax.transAxes,
                          color="white", fontsize=10.5, family="monospace",
                          verticalalignment="bottom",
                          bbox=dict(facecolor="black", alpha=0.78,
                                    edgecolor=edge, linewidth=1.4, boxstyle="round,pad=0.5"))
            # Subtitle: which frame within the clip
            ax.text2D(0.02, 0.02, f"frame {f}", transform=ax.transAxes,
                      color="#999", fontsize=8)
            writer.grab_frame()
    plt.close(fig)
    return True


def render_title_plate(shot: dict, out_path: Path, duration_s: float = 1.0, fps: int = 60) -> None:
    """Black-background title card before each shot's reenactment."""
    fig = plt.figure(figsize=(11, 8), dpi=110, facecolor="#101010")
    ax = fig.add_subplot(111, facecolor="#101010")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    line1 = f"Shot {shot['shot_id']}"
    line2 = f"{shot['shot_type']}    {'MAKE' if shot['make'] else 'MISS'}    " \
            f"{'UNCONTESTED' if shot['uncontested'] else 'contested'}"
    line3 = f"release frame {shot['release_frame_a']}"
    ax.text(0.5, 0.6, line1, ha="center", va="center", transform=ax.transAxes,
            color="white", fontsize=46, fontweight="bold")
    color2 = "#5ee06c" if shot["make"] else "#e0625e"
    ax.text(0.5, 0.45, line2, ha="center", va="center", transform=ax.transAxes,
            color=color2, fontsize=22)
    ax.text(0.5, 0.32, line3, ha="center", va="center", transform=ax.transAxes,
            color="#888", fontsize=14)

    writer = FFMpegWriter(fps=fps, bitrate=2000, codec="libx264")
    with writer.saving(fig, str(out_path), dpi=110):
        for _ in range(int(round(duration_s * fps))):
            writer.grab_frame()
    plt.close(fig)


def concat_clips(clip_paths: list[Path], out_path: Path) -> None:
    """ffmpeg concat the list of clips into one MP4. Re-encodes with libx264 because
    title-plate and reenactment clips have differing codec params (which breaks -c copy)."""
    list_file = out_path.parent / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{p.name}'" for p in clip_paths))
    cmd = ["ffmpeg", "-y", "-v", "warning", "-f", "concat", "-safe", "0",
           "-i", list_file.name, "-c:v", "libx264", "-crf", "20", "-preset", "fast",
           "-pix_fmt", "yuv420p", out_path.name]
    subprocess.run(cmd, check=True, cwd=out_path.parent)


def _load_metric_panels(contest_path: Path) -> dict[int, dict]:
    """Build a {shot_id: metric_panel} dict from contest_metrics.json (instantaneous values)."""
    if not contest_path.exists():
        return {}
    data = json.loads(contest_path.read_text())
    out: dict[int, dict] = {}
    for row in data.get("per_shot", []):
        inst = (row.get("pred") or {}).get("instantaneous") or {}
        out[row["shot_id"]] = {
            "wrist_to_wrist_m": inst.get("wrist_to_wrist_distance_m"),
            "wrist_height_advantage_m": inst.get("wrist_height_advantage_m"),
            "shooter_fade_mps": inst.get("shooter_fade_mps"),
            "floor_distance_m": inst.get("floor_distance_m"),
            "composite_z": inst.get("composite_z"),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="2026-06-02_1on1")
    p.add_argument("--half-window-frames", type=int, default=90, help="frames before/after release")
    p.add_argument("--no-title-plates", action="store_true")
    p.add_argument("--no-concat", action="store_true")
    p.add_argument("--shot-ids", default=None,
                   help="Comma-separated shot IDs to render (default: all).")
    p.add_argument("--out-subdir", default="3d_reenactments",
                   help="Subdirectory under outputs/sessions/<session>/ for output clips.")
    args = p.parse_args()

    out_dir = Path(f"outputs/sessions/{args.session}")
    pose3d = read_json_as(Pose3DSequence, out_dir / "pose3d.json")
    gt = json.loads(Path(f"data/sessions/{args.session}/shots_gt.json").read_text())
    sync = json.loads(Path(f"data/sessions/{args.session}/sync.json").read_text())
    fps_a = sync["fps_a"]
    print(f"Loaded {len(pose3d.poses)} 3D poses, {gt['n_shots']} GT shots; fps_a={fps_a:.3f}")

    metric_panels = _load_metric_panels(out_dir / "contest_metrics.json")
    if metric_panels:
        print(f"Loaded contest metrics for {len(metric_panels)} shots")

    if args.shot_ids:
        wanted = {int(s) for s in args.shot_ids.split(",")}
        shots = [s for s in gt["shots"] if s["shot_id"] in wanted]
        print(f"Filtering to {len(shots)} shots: {sorted(wanted)}")
    else:
        shots = gt["shots"]

    reenact_dir = out_dir / args.out_subdir
    reenact_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []
    for shot in shots:
        sid = shot["shot_id"]
        if not args.no_title_plates:
            tp_path = reenact_dir / f"shot_{sid:03d}_title.mp4"
            render_title_plate(shot, tp_path, fps=int(round(fps_a)))
            clip_paths.append(tp_path)
        clip_path = reenact_dir / f"shot_{sid:03d}.mp4"
        ok = render_shot_clip(shot, pose3d, fps_a, clip_path,
                              half_window_frames=args.half_window_frames,
                              metric_panel=metric_panels.get(sid))
        if ok:
            print(f"  ✓ rendered shot {sid:>2}")
            clip_paths.append(clip_path)
        else:
            print(f"  ⚠ skipped shot {sid:>2}")

    if not args.no_concat and clip_paths:
        reel = reenact_dir / "highlight_reel.mp4"
        print(f"\nConcatenating {len(clip_paths)} clips → {reel.name}")
        try:
            concat_clips(clip_paths, reel)
            print(f"  ✓ {reel}")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠ concat failed: {e}")

    print(f"\n✓ Output dir: {reenact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
