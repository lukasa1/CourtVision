"""2-camera DLT triangulation. Replaces Pose2Sim for the 2-camera case.

Given two ``Pose2DSequence`` (one per camera), their intrinsic K matrices +
distortion coefficients, their extrinsic R/t matrices (relative to a common
world frame), and the cross-camera frame-drift slope, produce a
``Pose3DSequence`` in world coordinates.

For 1-on-1 basketball with two players we have only two tracks per camera,
so cross-camera identity matching is solved by trying both possible mappings
on a few overlap frames and picking whichever gives lower mean reprojection
error after triangulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.schemas import (
    Keypoint3D,
    Pose2D,
    Pose2DSequence,
    Pose3D,
    Pose3DSequence,
)


def _projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """3x4 camera projection matrix P = K @ [R | t]."""
    return K @ np.hstack([R, t.reshape(3, 1)])


def _undistort_pt(x: float, y: float, K: np.ndarray, dist: np.ndarray) -> tuple[float, float]:
    """Undistort one (x, y) pixel point back into the linear camera model."""
    pt = np.array([[[x, y]]], dtype=np.float64)
    und = cv2.undistortPoints(pt, K, dist, P=K)
    return float(und[0, 0, 0]), float(und[0, 0, 1])


def _triangulate(
    x_a: float, y_a: float, K_a: np.ndarray, dist_a: np.ndarray, P_a: np.ndarray,
    x_b: float, y_b: float, K_b: np.ndarray, dist_b: np.ndarray, P_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Triangulate one point seen in two cameras. Returns (3D point, mean reproj err in px)."""
    ua = np.array(_undistort_pt(x_a, y_a, K_a, dist_a), dtype=np.float64).reshape(2, 1)
    ub = np.array(_undistort_pt(x_b, y_b, K_b, dist_b), dtype=np.float64).reshape(2, 1)
    X_h = cv2.triangulatePoints(P_a, P_b, ua, ub)
    X = (X_h[:3] / X_h[3]).ravel()

    # Reproject for sanity
    proj_a = P_a @ np.array([X[0], X[1], X[2], 1.0])
    proj_a = proj_a[:2] / proj_a[2]
    proj_b = P_b @ np.array([X[0], X[1], X[2], 1.0])
    proj_b = proj_b[:2] / proj_b[2]
    err_a = float(np.linalg.norm(proj_a - np.array([ua[0, 0], ua[1, 0]])))
    err_b = float(np.linalg.norm(proj_b - np.array([ub[0, 0], ub[1, 0]])))
    return X, 0.5 * (err_a + err_b)


def _b_frame_for(a_frame: int, drift_slope: float, n_frames_b: int) -> int:
    return max(0, min(int(round(a_frame * drift_slope)), n_frames_b - 1))


def _by_frame(poses: list[Pose2D]) -> dict[int, list[Pose2D]]:
    out: dict[int, list[Pose2D]] = {}
    for p in poses:
        out.setdefault(p.frame_idx, []).append(p)
    return out


def _triangulate_pose_pair(
    pose_a: Pose2D, pose_b: Pose2D,
    K_a: np.ndarray, dist_a: np.ndarray, P_a: np.ndarray,
    K_b: np.ndarray, dist_b: np.ndarray, P_b: np.ndarray,
    min_score: float = 0.30,
) -> tuple[list[Keypoint3D], float]:
    """Triangulate every keypoint in a pair of poses. Returns (3D keypoints, mean reproj err)."""
    kps_3d: list[Keypoint3D] = []
    errs: list[float] = []
    for i, (kp_a, kp_b) in enumerate(zip(pose_a.keypoints, pose_b.keypoints)):
        if kp_a.score < min_score or kp_b.score < min_score:
            kps_3d.append(Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.0, n_cameras=0))
            continue
        X, err = _triangulate(
            kp_a.x, kp_a.y, K_a, dist_a, P_a,
            kp_b.x, kp_b.y, K_b, dist_b, P_b,
        )
        conf = 0.5 * (kp_a.score + kp_b.score)
        kps_3d.append(Keypoint3D(x=float(X[0]), y=float(X[1]), z=float(X[2]),
                                 confidence=float(conf), n_cameras=2))
        errs.append(err)
    mean_err = float(np.mean(errs)) if errs else 0.0
    return kps_3d, mean_err


def find_track_mapping(
    poses_a: list[Pose2D], poses_b: list[Pose2D],
    K_a: np.ndarray, dist_a: np.ndarray, P_a: np.ndarray,
    K_b: np.ndarray, dist_b: np.ndarray, P_b: np.ndarray,
    drift_slope: float,
    n_frames_b: int,
    n_overlap_frames: int = 60,
) -> dict[int, int]:
    """Return {a_track_id: b_track_id} for the 1-on-1 case (exactly 2 tracks per cam).

    Strategy: take the first ``n_overlap_frames`` frames where both cameras have
    >= 2 tracks visible. For each such frame, test both possible mappings and
    compute the mean reprojection error of triangulating each pair's pelvis-ish
    keypoint. The mapping with the lower aggregate error wins.

    For Halpe-26 the pelvis is keypoint 19 (Hip); for COCO-17 we fall back to
    averaging hips (11, 12).
    """
    a_by_frame = _by_frame(poses_a)
    b_by_frame = _by_frame(poses_b)

    # Reference keypoint (pelvis-ish) per format
    def pelvis(pose: Pose2D) -> tuple[float, float, float] | None:
        if pose.keypoint_format == "halpe26" and len(pose.keypoints) >= 20:
            kp = pose.keypoints[19]  # Hip
            return (kp.x, kp.y, kp.score)
        # COCO-17: average hips (11=L_Hip, 12=R_Hip)
        if pose.keypoint_format == "coco17" and len(pose.keypoints) >= 13:
            lh, rh = pose.keypoints[11], pose.keypoints[12]
            score = 0.5 * (lh.score + rh.score)
            return ((lh.x + rh.x) * 0.5, (lh.y + rh.y) * 0.5, score)
        return None

    a_track_ids = sorted({p.track_id for p in poses_a})
    b_track_ids = sorted({p.track_id for p in poses_b})

    # Only handle the 2-player case cleanly; for >2 tracks we'd need Hungarian
    if len(a_track_ids) > 2 or len(b_track_ids) > 2:
        # Fall back: just match by ID order (1-on-1 assumption)
        return {a: b for a, b in zip(a_track_ids, b_track_ids)}
    if len(a_track_ids) < 2 or len(b_track_ids) < 2:
        return {a: b for a, b in zip(a_track_ids, b_track_ids)}

    # Build the two candidate mappings
    a_lo, a_hi = a_track_ids[0], a_track_ids[1]
    b_lo, b_hi = b_track_ids[0], b_track_ids[1]
    candidates = [
        {a_lo: b_lo, a_hi: b_hi},
        {a_lo: b_hi, a_hi: b_lo},
    ]
    candidate_errors = [0.0, 0.0]
    candidate_counts = [0, 0]

    sorted_a_frames = sorted(a_by_frame.keys())
    n_used = 0
    for a_frame in sorted_a_frames:
        if n_used >= n_overlap_frames: break
        b_frame = _b_frame_for(a_frame, drift_slope, n_frames_b)
        a_poses = a_by_frame[a_frame]
        b_poses = b_by_frame.get(b_frame, [])
        if len(a_poses) < 2 or len(b_poses) < 2: continue

        # Use pelvis points only for matching test
        a_pelvises = {p.track_id: pelvis(p) for p in a_poses}
        b_pelvises = {p.track_id: pelvis(p) for p in b_poses}
        if any(v is None for v in a_pelvises.values()): continue
        if any(v is None for v in b_pelvises.values()): continue
        if any(v[2] < 0.3 for v in a_pelvises.values()): continue
        if any(v[2] < 0.3 for v in b_pelvises.values()): continue

        for ci, mapping in enumerate(candidates):
            frame_err = 0.0
            for a_tid, b_tid in mapping.items():
                xa, ya, _ = a_pelvises[a_tid]
                xb, yb, _ = b_pelvises[b_tid]
                _, err = _triangulate(xa, ya, K_a, dist_a, P_a, xb, yb, K_b, dist_b, P_b)
                frame_err += err
            candidate_errors[ci] += frame_err
            candidate_counts[ci] += 1
        n_used += 1

    if candidate_counts[0] == 0:
        # Couldn't find any frame to test on — fall back to ID-order
        return {a_lo: b_lo, a_hi: b_hi}
    avg_errors = [e / max(c, 1) for e, c in zip(candidate_errors, candidate_counts)]
    chosen = int(np.argmin(avg_errors))
    return candidates[chosen]


def _pelvis_xy(pose: Pose2D, score_thr: float = 0.30) -> tuple[float, float] | None:
    """Pick the most reliable per-pose pelvis-ish pixel coordinate."""
    kps = pose.keypoints
    if pose.keypoint_format == "halpe26" and len(kps) >= 20:
        hip = kps[19]
        if hip.score >= score_thr:
            return (hip.x, hip.y)
    if len(kps) >= 13:
        lh, rh = kps[11], kps[12]
        if lh.score >= score_thr and rh.score >= score_thr:
            return (0.5 * (lh.x + rh.x), 0.5 * (lh.y + rh.y))
    return None


def triangulate_session(
    pose2d_a: Pose2DSequence,
    pose2d_b: Pose2DSequence,
    extrinsics_a: dict[str, Any],
    extrinsics_b: dict[str, Any],
    sync: dict[str, Any],
    calibration_ref: str = "",
    min_keypoint_score: float = 0.30,
    max_pelvis_match_err_px: float = 150.0,
) -> Pose3DSequence:
    """Fuse two Pose2DSequences into a Pose3DSequence via DLT triangulation.

    Cross-camera identity is solved **per frame** by triangulating each
    candidate (a_pose, b_pose) pair's pelvis and pairing by minimum
    reprojection error (Hungarian assignment). This is robust to per-camera
    ByteTrack ID switches over time — we don't rely on persistent IDs.
    """
    from scipy.optimize import linear_sum_assignment

    K_a = np.asarray(extrinsics_a["K"], dtype=np.float64)
    dist_a = np.asarray(extrinsics_a["dist"], dtype=np.float64).ravel()
    R_a = np.asarray(extrinsics_a["R"], dtype=np.float64)
    t_a = np.asarray(extrinsics_a["tvec"], dtype=np.float64).reshape(3)
    P_a = _projection_matrix(K_a, R_a, t_a)

    K_b = np.asarray(extrinsics_b["K"], dtype=np.float64)
    dist_b = np.asarray(extrinsics_b["dist"], dtype=np.float64).ravel()
    R_b = np.asarray(extrinsics_b["R"], dtype=np.float64)
    t_b = np.asarray(extrinsics_b["tvec"], dtype=np.float64).reshape(3)
    P_b = _projection_matrix(K_b, R_b, t_b)

    drift = float(sync.get("drift_slope_b_per_a", pose2d_b.fps / pose2d_a.fps))
    n_b = max((p.frame_idx for p in pose2d_b.poses), default=0) + 1

    a_by_frame = _by_frame(pose2d_a.poses)
    b_by_frame = _by_frame(pose2d_b.poses)

    out_poses: list[Pose3D] = []
    n_paired = 0
    n_dropped_high_err = 0
    n_frames_no_pair = 0

    for a_frame in sorted(a_by_frame.keys()):
        b_frame = _b_frame_for(a_frame, drift, n_b)
        a_poses = a_by_frame[a_frame]
        b_poses = b_by_frame.get(b_frame, [])
        if not a_poses or not b_poses:
            continue

        # Pairwise pelvis reprojection-error cost matrix.
        nA, nB = len(a_poses), len(b_poses)
        cost = np.full((nA, nB), 1e9)
        for i, pa in enumerate(a_poses):
            pa_pelvis = _pelvis_xy(pa, min_keypoint_score)
            if pa_pelvis is None:
                continue
            for j, pb in enumerate(b_poses):
                pb_pelvis = _pelvis_xy(pb, min_keypoint_score)
                if pb_pelvis is None:
                    continue
                _, err = _triangulate(
                    pa_pelvis[0], pa_pelvis[1], K_a, dist_a, P_a,
                    pb_pelvis[0], pb_pelvis[1], K_b, dist_b, P_b,
                )
                cost[i, j] = err

        if not np.isfinite(cost).any() or cost.min() >= 1e9:
            n_frames_no_pair += 1
            continue

        # Hungarian assignment — finds the global-min total-cost pairing.
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] > max_pelvis_match_err_px:
                n_dropped_high_err += 1
                continue
            kps3d, err = _triangulate_pose_pair(
                a_poses[i], b_poses[j],
                K_a, dist_a, P_a, K_b, dist_b, P_b,
                min_score=min_keypoint_score,
            )
            out_poses.append(Pose3D(
                frame_idx=a_frame,
                # Use A's track id as canonical (downstream uses this for ID lookups).
                track_id=a_poses[i].track_id,
                keypoints=kps3d,
                reprojection_error_px=err,
            ))
            n_paired += 1

    print(f"  per-frame matching: paired {n_paired}, dropped {n_dropped_high_err} "
          f"(pelvis err > {max_pelvis_match_err_px} px), {n_frames_no_pair} frames had no valid pelvis pair")

    return Pose3DSequence(
        session_id=pose2d_a.session_id,
        fps=pose2d_a.fps,
        poses=out_poses,
        calibration_ref=calibration_ref,
    )
