"""Intrinsic camera calibration from a checkerboard or ChArUco video.

iPhone-aware: handles `rotation=-90` metadata by rotating frames to natural
portrait orientation before detection, so K is valid for the same frames the
downstream pipeline (Stage 1 / 2 / 4) will consume.

Usage:
    uv run python scripts/calibrate_intrinsics.py \
        --video /path/to/A_checker.MOV \
        --board checker \
        --rotate -90 \
        --output data/calibration/intrinsics_A_checker.json

Board parameters HARDCODED to the PDFs in `Datasets/`:
- Checker: 9x6 INNER corners, 25 mm squares
- ChArUco: 11x8 squares, 20 mm square / 15 mm marker, DICT_4X4_50

Pipeline:
1. Sample 1 frame every `--sample-every` source frames.
2. Optionally rotate by `--rotate` degrees (counter-clockwise; -90 for our iPhone videos).
3. Detect board (checker or ChArUco).
4. Initial calibrateCamera.
5. Drop views with per-view reprojection error above the P90 of the distribution.
6. Re-calibrate on the kept views.
7. Save K, dist, RMS, n_views.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CHECKER_INNER = (9, 6)
CHECKER_SQUARE_M = 0.025

CHARUCO_SQUARES = (11, 8)
CHARUCO_SQ_M = 0.020
CHARUCO_MK_M = 0.015
CHARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def rotate_frame(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return img
    if degrees in (90, -270):
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees in (-90, 270):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if degrees in (180, -180):
        return cv2.rotate(img, cv2.ROTATE_180)
    raise ValueError(f"Unsupported rotation: {degrees}")


def _aruco_detector_permissive() -> Any:
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.005
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def collect_checkerboard_views(
    video_path: Path, sample_every: int, rotate_deg: int
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], int]:
    cols, rows = CHECKER_INNER
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * CHECKER_SQUARE_M

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture cannot open {video_path}")

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None
    frame_idx = 0
    n_attempts = 0

    while True:
        ok, img = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            n_attempts += 1
            img = rotate_frame(img, rotate_deg)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(
                gray,
                CHECKER_INNER,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRIT)
                obj_points.append(objp)
                img_points.append(corners)
        frame_idx += 1
    cap.release()
    assert img_size is not None
    return obj_points, img_points, img_size, n_attempts


def collect_charuco_views(
    video_path: Path, sample_every: int, rotate_deg: int
) -> tuple[list[np.ndarray], list[np.ndarray], Any, tuple[int, int], int]:
    cols, rows = CHARUCO_SQUARES
    aruco_dict = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard((cols, rows), CHARUCO_SQ_M, CHARUCO_MK_M, aruco_dict)
    detector_params = _aruco_detector_permissive()
    refine_params = cv2.aruco.RefineParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params, refine_params)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture cannot open {video_path}")

    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None
    frame_idx = 0
    n_attempts = 0

    while True:
        ok, img = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            n_attempts += 1
            img = rotate_frame(img, rotate_deg)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])
            ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
            if ch_corners is not None and ch_ids is not None and len(ch_ids) >= 6:
                all_corners.append(ch_corners)
                all_ids.append(ch_ids)
        frame_idx += 1
    cap.release()
    assert img_size is not None
    return all_corners, all_ids, board, img_size, n_attempts


def per_view_errors_checker(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
) -> np.ndarray:
    errs = []
    for obj, img, r, t in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))
        errs.append(err)
    return np.array(errs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--board", required=True, choices=["checker", "charuco"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--sample-every", type=int, default=15)
    p.add_argument("--rotate", type=int, default=0, choices=[-180, -90, 0, 90, 180])
    p.add_argument("--keep-percentile", type=float, default=90.0,
                   help="Drop views with per-view reprojection error above this percentile, then refit. Set 100 to disable.")
    p.add_argument("--rational-model", action="store_true",
                   help="Use CALIB_RATIONAL_MODEL (8-parameter distortion) for wide/short-focal lenses.")
    args = p.parse_args()

    print(f"Calibrating from {args.video.name} ({args.board}, rotate={args.rotate}°)")

    out: dict[str, Any] = {"video": str(args.video), "board_type": args.board, "rotate_deg": args.rotate}
    if args.board == "checker":
        obj_points, img_points, img_size, n_attempts = collect_checkerboard_views(
            args.video, args.sample_every, args.rotate
        )
        n_views = len(obj_points)
        print(f"  Attempted {n_attempts} frames; found checkerboard in {n_views}.")
        if n_views < 15:
            print(f"  ⚠ Need ≥ 15 successful detections. Try --sample-every {max(1, args.sample_every//2)}.")
            return 1

        flags = cv2.CALIB_RATIONAL_MODEL if args.rational_model else 0
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None, flags=flags
        )
        print(f"  Initial RMS: {ret:.4f} px on {n_views} views")

        errs = per_view_errors_checker(obj_points, img_points, K, dist, list(rvecs), list(tvecs))
        keep_thr = float(np.percentile(errs, args.keep_percentile))
        keep_mask = errs <= keep_thr
        n_kept = int(keep_mask.sum())
        if n_kept >= 15 and args.keep_percentile < 100:
            print(f"  Per-view error: median={np.median(errs):.3f} P90={np.percentile(errs, 90):.3f} max={errs.max():.3f}")
            print(f"  Dropping {n_views - n_kept} views with error > {keep_thr:.3f} px, refitting on {n_kept}...")
            obj_kept = [p for p, m in zip(obj_points, keep_mask) if m]
            img_kept = [p for p, m in zip(img_points, keep_mask) if m]
            ret, K, dist, _, _ = cv2.calibrateCamera(obj_kept, img_kept, img_size, None, None, flags=flags)
            n_views = n_kept

        out["board_params"] = {"inner_corners": list(CHECKER_INNER), "square_m": CHECKER_SQUARE_M}
        out.update({
            "resolution": list(img_size),
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "rms_reprojection_error_px": float(ret),
            "n_views": int(n_views),
            "n_attempts": int(n_attempts),
            "sample_every": args.sample_every,
            "rational_model": args.rational_model,
        })
    else:
        all_corners, all_ids, board, img_size, n_attempts = collect_charuco_views(
            args.video, args.sample_every, args.rotate
        )
        n_views = len(all_corners)
        print(f"  Attempted {n_attempts} frames; found ChArUco in {n_views}.")
        if n_views < 15:
            print(f"  ⚠ Need ≥ 15 successful detections; got {n_views}. ChArUco markers may be too small in frame. Use --board=checker instead.")
            return 1
        flags = cv2.CALIB_RATIONAL_MODEL if args.rational_model else 0
        ret, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, img_size, None, None, flags=flags
        )
        out["board_params"] = {
            "squares": list(CHARUCO_SQUARES), "square_m": CHARUCO_SQ_M,
            "marker_m": CHARUCO_MK_M, "dict": "DICT_4X4_50",
        }
        out.update({
            "resolution": list(img_size),
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "rms_reprojection_error_px": float(ret),
            "n_views": int(n_views),
            "n_attempts": int(n_attempts),
            "sample_every": args.sample_every,
            "rational_model": args.rational_model,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    K_arr = np.array(out["K"])
    print(f"  ✓ Final RMS: {ret:.4f} px on {n_views} views")
    print(f"  ✓ K: fx={K_arr[0,0]:.1f} fy={K_arr[1,1]:.1f} cx={K_arr[0,2]:.1f} cy={K_arr[1,2]:.1f}")
    print(f"  ✓ dist[0..4]: {[round(v,4) for v in out['dist'][:5]]}")
    print(f"  ✓ resolution: {out['resolution']}")
    print(f"  ✓ Saved → {args.output}")
    if ret < 0.5:
        print("  ✓ RMS < 0.5 px — passes acceptance bar.")
    elif ret < 1.0:
        print("  ✓ RMS < 1.0 px — acceptable; verify downstream reprojection.")
    elif ret < 2.0:
        print("  ⚠ RMS in [1, 2) px — usable but downstream 3D error will be elevated.")
    else:
        print("  ⚠⚠ RMS ≥ 2 px. Re-record with board held closer (≈ 50 cm) and held steady at varied angles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
