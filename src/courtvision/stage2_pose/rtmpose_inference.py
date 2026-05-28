"""Stage 2 — RTMPose-l Halpe-26 top-down inference (rtmlib / ONNX). PRD §9.2.

Top-down 2D pose on the player crops emitted by Stage 1. Public entrypoint
``run_stage2`` consumes a :class:`courtvision.schemas.TrackSequence` plus the
source clip and returns a :class:`courtvision.schemas.Pose2DSequence` with
26-keypoint Halpe poses, one per (frame, track).

Inference runs through **rtmlib** (ONNX / onnxruntime), not the MMPose PyTorch
stack: the OpenMMLab stack is unusable on the Modal image's bleeding-edge
torch 2.12 + CUDA 13 (see ``flags.md`` F20 / ``decisions.md`` Decision 2). This
is the *same* RTMPose-l Halpe-26 model + weights, just a different runtime.
rtmlib depends only on numpy/opencv/onnxruntime, so this module needs no torch
or mmcv; ``rtmlib`` is imported lazily so the file imports on the laptop for the
pure-Python callers (tests, schema round-trips) without it installed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.logging import get_logger
from courtvision.schemas import Keypoint2D, Pose2D, Pose2DSequence, Track, TrackSequence

_log = get_logger(__name__)

# RTMPose-l Halpe-26 ONNX (body7 pretrain, 256x192) — the ONNX export of the
# same checkpoint Stage 2 was specced on (hash 2abb7558). rtmlib downloads it
# on first use. Override via config['model']['onnx_model'].
_DEFAULT_ONNX = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip"
)


def _video_meta(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return fps, n_frames


def _tracks_by_frame(tracks: TrackSequence) -> dict[int, list[Track]]:
    by_frame: dict[int, list[Track]] = defaultdict(list)
    for t in tracks.tracks:
        by_frame[t.frame_idx].append(t)
    return by_frame


def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def run_stage2(tracks: TrackSequence, video_path: Path, config: dict[str, Any]) -> Pose2DSequence:
    """Top-down pose inference on Stage 1 crops. Returns ``Pose2DSequence``.

    RTMPose-l Halpe-26 via rtmlib (ONNX). For each frame that has tracks, runs
    the pose model on that frame's bboxes (one instance per track) and records
    a 26-keypoint :class:`Pose2D` per track; frames with no tracks produce no
    poses. ``config`` is shaped like
    ``configs/stage2_pose/rtmpose_l_halpe26.yaml``.
    """
    from rtmlib import RTMPose  # lazy: rtmlib/onnxruntime only on the inference image

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    model_cfg = config["model"]
    onnx_model = str(model_cfg.get("onnx_model") or _DEFAULT_ONNX)
    w, h = model_cfg.get("input_size", [192, 256])
    backend = str(model_cfg.get("backend", "onnxruntime"))
    device = str(model_cfg.get("device", "cpu"))
    num_keypoints = int(model_cfg.get("num_keypoints", 26))

    fps, n_frames = _video_meta(video_path)
    by_frame = _tracks_by_frame(tracks)
    _log.info(
        "stage2(rtmlib): %s cam=%s fps=%.2f n_frames=%d tracked_frames=%d device=%s",
        tracks.session_id,
        tracks.camera_id,
        fps,
        n_frames,
        len(by_frame),
        device,
    )

    pose_model = RTMPose(
        onnx_model=onnx_model,
        model_input_size=(int(w), int(h)),
        backend=backend,
        device=device,
    )

    poses: list[Pose2D] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_tracks = by_frame.get(frame_idx)
            if frame_tracks:
                bboxes = [_xywh_to_xyxy(t.bbox) for t in frame_tracks]
                # rtmlib: (N, K, 2) keypoints + (N, K) scores, order matches bboxes.
                kpts, scores = pose_model(frame, bboxes=bboxes)
                kpts = np.asarray(kpts)
                scores = np.asarray(scores)
                for i, t in enumerate(frame_tracks):
                    person_kpts = kpts[i]
                    person_scores = scores[i]
                    if person_kpts.shape[0] != num_keypoints:
                        raise ValueError(
                            f"expected {num_keypoints} keypoints from RTMPose, "
                            f"got {person_kpts.shape[0]} — wrong ONNX model?"
                        )
                    poses.append(
                        Pose2D(
                            frame_idx=frame_idx,
                            track_id=t.track_id,
                            keypoints=[
                                Keypoint2D(
                                    x=float(person_kpts[k][0]),
                                    y=float(person_kpts[k][1]),
                                    score=float(person_scores[k]),
                                )
                                for k in range(num_keypoints)
                            ],
                            keypoint_format="halpe26",
                        )
                    )
            frame_idx += 1
    finally:
        cap.release()

    _log.info("stage2: emitted %d poses across %d frames", len(poses), len(by_frame))
    return Pose2DSequence(
        session_id=tracks.session_id,
        camera_id=tracks.camera_id,
        fps=fps if fps > 0 else tracks.fps,
        poses=poses,
    )


__all__ = ["run_stage2"]
