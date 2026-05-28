"""Stage 2 — RTMPose-l Halpe-26 top-down inference. PRD §9.2.

Top-down 2D pose on the player crops emitted by Stage 1. Public entrypoint
``run_stage2`` consumes a :class:`courtvision.schemas.TrackSequence` plus the
source clip and returns a :class:`courtvision.schemas.Pose2DSequence` with
26-keypoint Halpe poses, one per (frame, track).

The OpenMMLab stack is Modal-only (it doesn't build on macOS arm64 — see
``flags.md`` F1/F17), so ``mmpose`` is imported lazily inside ``run_stage2``;
this module imports cleanly on the laptop for the pure-Python callers (tests,
schema round-trips) without the stack present.
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

# OpenMMLab ships its config tree inside the wheel under ``<pkg>/.mim/`` so
# ``mim``/downstream tools can find it. The checkpoint we pull
# (``rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192``) is a 26-keypoint
# Halpe head — it must be paired with a *halpe26* model config, NOT the
# coco-256x192 (17-kpt) config the Phase-A stub referenced. We try the known
# halpe26 config relpaths in order; ``config['model']['cfg_path']`` overrides.
_HALPE26_CONFIG_CANDIDATES = (
    "configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb512-700e_body8-halpe26-256x192.py",
    "configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb1024-700e_body8-halpe26-256x192.py",
)


def _resolve_model_config(model_cfg: dict[str, Any]) -> str:
    """Return a filesystem path to the RTMPose Halpe-26 model config.

    Honours an explicit ``cfg_path`` override; otherwise searches the
    installed ``mmpose`` package's bundled ``.mim/configs`` tree for a known
    halpe26 RTMPose-l config. Raises with actionable guidance if none is
    found (the exact filename can drift across mmpose releases).
    """
    override = model_cfg.get("cfg_path")
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"model.cfg_path does not exist: {p}")
        return str(p)

    import mmpose  # lazy: Modal-only

    mim_root = Path(mmpose.__file__).parent / ".mim"
    for rel in _HALPE26_CONFIG_CANDIDATES:
        cand = mim_root / rel
        if cand.exists():
            _log.info("resolved RTMPose Halpe-26 config: %s", cand)
            return str(cand)

    # Last resort: glob for any rtmpose-l halpe26 config under the tree.
    matches = sorted(mim_root.glob("configs/**/rtmpose-l_*halpe26*.py"))
    if matches:
        _log.info("resolved RTMPose Halpe-26 config (glob): %s", matches[0])
        return str(matches[0])

    raise FileNotFoundError(
        "could not locate an RTMPose-l Halpe-26 model config under "
        f"{mim_root}/configs. Set config['model']['cfg_path'] to the correct "
        "mmpose config .py for the rtmpose-l body8 halpe26 head."
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


def run_stage2(
    tracks: TrackSequence, video_path: Path, config: dict[str, Any]
) -> Pose2DSequence:
    """Top-down pose inference on Stage 1 crops. Returns ``Pose2DSequence``.

    Locked model: RTMPose-l, Halpe-26 keypoints (PRD §9.2). For each frame
    that has tracks, runs ``inference_topdown`` on that frame's bboxes (one
    instance per track) and records a 26-keypoint :class:`Pose2D` per track.
    Frames with no tracks produce no poses. ``config`` is shaped like
    ``configs/stage2_pose/rtmpose_l_halpe26.yaml``.
    """
    from mmpose.apis import inference_topdown, init_model  # lazy: Modal-only

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    model_cfg = config["model"]
    weights = str(model_cfg["weights"])
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"RTMPose checkpoint not found: {weights} (expected on the "
            "artifacts Volume at /artifacts/checkpoints/rtmpose_l_halpe26.pth)"
        )
    num_keypoints = int(model_cfg.get("num_keypoints", 26))
    device = str(model_cfg.get("device", "cuda:0"))
    cfg_path = _resolve_model_config(model_cfg)

    fps, n_frames = _video_meta(video_path)
    by_frame = _tracks_by_frame(tracks)
    _log.info(
        "stage2: %s cam=%s fps=%.2f n_frames=%d tracked_frames=%d device=%s",
        tracks.session_id,
        tracks.camera_id,
        fps,
        n_frames,
        len(by_frame),
        device,
    )

    model = init_model(cfg_path, weights, device=device)

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
                bboxes = np.array(
                    [_xywh_to_xyxy(t.bbox) for t in frame_tracks], dtype=np.float32
                )
                samples = inference_topdown(model, frame, bboxes, bbox_format="xyxy")
                for t, sample in zip(frame_tracks, samples, strict=True):
                    inst = sample.pred_instances
                    kpts = np.asarray(inst.keypoints)[0]  # (K, 2)
                    scores = np.asarray(inst.keypoint_scores)[0]  # (K,)
                    if kpts.shape[0] != num_keypoints:
                        raise ValueError(
                            f"expected {num_keypoints} keypoints from RTMPose, "
                            f"got {kpts.shape[0]} — model config likely not the "
                            "halpe26 head (see _resolve_model_config)"
                        )
                    poses.append(
                        Pose2D(
                            frame_idx=frame_idx,
                            track_id=t.track_id,
                            keypoints=[
                                Keypoint2D(
                                    x=float(kpts[k][0]),
                                    y=float(kpts[k][1]),
                                    score=float(scores[k]),
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
