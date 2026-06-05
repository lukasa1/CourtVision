"""Run 2-cam DLT triangulation on a session's Stage 2 outputs.

Loads:
    outputs/sessions/<session>/pose2d_cam_a.json
    outputs/sessions/<session>/pose2d_cam_b.json
    data/calibration/intrinsics_A_checker.json     (K, dist)
    data/calibration/intrinsics_B_checker.json
    data/sessions/<session>/extrinsics_A.json      (R, t)
    data/sessions/<session>/extrinsics_B.json
    data/sessions/<session>/sync.json              (drift_slope_b_per_a)

Writes:
    outputs/sessions/<session>/pose3d.json
    outputs/sessions/<session>/stage4_overlay.mp4  (side-by-side reprojection check)

Usage:
    uv run python scripts/triangulate_session.py --session 2026-06-02_1on1
    uv run python scripts/triangulate_session.py --session 2026-06-02_1on1 --no-overlay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from courtvision.io_utils import read_json_as, write_json
from courtvision.schemas import Pose2D, Pose2DSequence, Pose3D, Pose3DSequence, Keypoint2D
from courtvision.stage2_pose.overlay import _HALPE26_SKELETON, _palette, _draw_pose
from courtvision.stage4_triangulation.dlt_2cam import (
    _projection_matrix,
    triangulate_session,
)


def load_calib(intrinsics_path: Path, extrinsics_path: Path) -> dict:
    intr = json.loads(intrinsics_path.read_text())
    extr = json.loads(extrinsics_path.read_text())
    return {
        "K": np.asarray(intr["K"], dtype=np.float64),
        "dist": np.asarray(intr["dist"], dtype=np.float64).ravel(),
        "R": np.asarray(extr["R"], dtype=np.float64),
        "tvec": np.asarray(extr["tvec"], dtype=np.float64).reshape(3),
    }


def reproject_pose3d_to_pose2d(pose3d: Pose3D, K: np.ndarray, dist: np.ndarray,
                                R: np.ndarray, t: np.ndarray) -> Pose2D:
    """Project a 3D pose into one camera (with distortion) → Pose2D for overlay rendering."""
    rvec, _ = cv2.Rodrigues(R)
    obj_pts = np.array([[kp.x, kp.y, kp.z] for kp in pose3d.keypoints], dtype=np.float64).reshape(-1, 1, 3)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, t.reshape(3, 1), K, dist)
    img_pts = img_pts.reshape(-1, 2)

    kps2d = [
        Keypoint2D(x=float(p[0]), y=float(p[1]),
                   score=float(kp.confidence) if kp.n_cameras > 0 else 0.0)
        for p, kp in zip(img_pts, pose3d.keypoints)
    ]
    return Pose2D(
        frame_idx=pose3d.frame_idx,
        track_id=pose3d.track_id,
        keypoints=kps2d,
        keypoint_format="halpe26",
    )


def render_reproj_overlay(
    pose3d_seq: Pose3DSequence,
    video_a: Path, video_b: Path,
    K_a: np.ndarray, dist_a: np.ndarray, R_a: np.ndarray, t_a: np.ndarray,
    K_b: np.ndarray, dist_b: np.ndarray, R_b: np.ndarray, t_b: np.ndarray,
    drift: float,
    out_path: Path,
    scale: float = 0.5,
) -> None:
    """Render a side-by-side overlay where each frame shows the reprojected 3D
    skeleton on top of the original camera-A and camera-B video frames."""
    cap_a = cv2.VideoCapture(str(video_a))
    cap_b = cv2.VideoCapture(str(video_b))
    n_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    n_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_a.get(cv2.CAP_PROP_FPS)
    h = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh, sw = int(h * scale), int(w * scale)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out_w, out_h = sw * 2, sh
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    # Group poses by frame
    by_frame: dict[int, list[Pose3D]] = {}
    for p in pose3d_seq.poses:
        by_frame.setdefault(p.frame_idx, []).append(p)
    track_ids = sorted({p.track_id for p in pose3d_seq.poses})
    palette = _palette(track_ids)

    for a_frame in range(n_a):
        ok_a, frame_a = cap_a.read()
        if not ok_a: break
        b_frame = max(0, min(int(round(a_frame * drift)), n_b - 1))
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, b_frame)
        ok_b, frame_b = cap_b.read()
        if not ok_b:
            frame_b = np.zeros_like(frame_a)

        poses_here = by_frame.get(a_frame, [])
        for p3d in poses_here:
            colour = palette.get(p3d.track_id, (0, 255, 0))
            pose_a_proj = reproject_pose3d_to_pose2d(p3d, K_a, dist_a, R_a, t_a)
            pose_b_proj = reproject_pose3d_to_pose2d(p3d, K_b, dist_b, R_b, t_b)
            _draw_pose(frame_a, pose_a_proj, colour)
            _draw_pose(frame_b, pose_b_proj, colour)

        # Banner with reprojection error of poses on this frame
        if poses_here:
            mean_err = float(np.mean([p.reprojection_error_px for p in poses_here]))
            cv2.putText(frame_a, f"3D->2D reproj err: {mean_err:.1f} px (Cam A)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_b, f"3D->2D reproj err: {mean_err:.1f} px (Cam B)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        a_small = cv2.resize(frame_a, (sw, sh))
        b_small = cv2.resize(frame_b, (sw, sh))
        writer.write(np.hstack([a_small, b_small]))

        if a_frame % 600 == 0:
            print(f"  rendered up to frame {a_frame}/{n_a}")

    writer.release()
    cap_a.release(); cap_b.release()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="2026-06-02_1on1")
    p.add_argument("--no-overlay", action="store_true", help="Skip the side-by-side reprojection video.")
    args = p.parse_args()

    out_dir = Path(f"outputs/sessions/{args.session}")
    pose_a_path = out_dir / "pose2d_cam_a.json"
    pose_b_path = out_dir / "pose2d_cam_b.json"
    if not pose_a_path.exists() or not pose_b_path.exists():
        print(f"⚠ Missing pose2d JSONs. Expected:\n  {pose_a_path}\n  {pose_b_path}")
        return 1

    pose2d_a = read_json_as(Pose2DSequence, pose_a_path)
    pose2d_b = read_json_as(Pose2DSequence, pose_b_path)
    print(f"Loaded poses: A={len(pose2d_a.poses)} entries, B={len(pose2d_b.poses)} entries")

    calib_a = load_calib(
        Path("data/calibration/intrinsics_A_checker.json"),
        Path(f"data/sessions/{args.session}/extrinsics_A.json"),
    )
    calib_b = load_calib(
        Path("data/calibration/intrinsics_B_checker.json"),
        Path(f"data/sessions/{args.session}/extrinsics_B.json"),
    )
    sync = json.loads(Path(f"data/sessions/{args.session}/sync.json").read_text())

    print(f"Triangulating...")
    pose3d_seq = triangulate_session(
        pose2d_a, pose2d_b,
        extrinsics_a={"K": calib_a["K"], "dist": calib_a["dist"], "R": calib_a["R"], "tvec": calib_a["tvec"]},
        extrinsics_b={"K": calib_b["K"], "dist": calib_b["dist"], "R": calib_b["R"], "tvec": calib_b["tvec"]},
        sync=sync,
        calibration_ref=f"data/sessions/{args.session}/extrinsics_*.json",
    )
    print(f"  {len(pose3d_seq.poses)} 3D pose entries")
    errs = [p.reprojection_error_px for p in pose3d_seq.poses]
    if errs:
        print(f"  reprojection error: mean {np.mean(errs):.2f} px, median {np.median(errs):.2f}, "
              f"P90 {np.percentile(errs, 90):.2f}, max {np.max(errs):.2f}")

    out_path = out_dir / "pose3d.json"
    write_json(pose3d_seq, out_path)
    print(f"  ✓ Saved → {out_path}")

    if not args.no_overlay:
        overlay_path = out_dir / "stage4_overlay.mp4"
        print(f"\nRendering 3D-reprojection overlay → {overlay_path}")
        render_reproj_overlay(
            pose3d_seq,
            video_a=Path(f"data/sessions/{args.session}/trimmed/A_trimmed.mp4"),
            video_b=Path(f"data/sessions/{args.session}/trimmed/B_trimmed.mp4"),
            K_a=calib_a["K"], dist_a=calib_a["dist"], R_a=calib_a["R"], t_a=calib_a["tvec"],
            K_b=calib_b["K"], dist_b=calib_b["dist"], R_b=calib_b["R"], t_b=calib_b["tvec"],
            drift=sync["drift_slope_b_per_a"],
            out_path=overlay_path,
        )
        print(f"  ✓ Saved → {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
