"""Stage 3 — WASB basketball detection. PRD §9.3.

Off-the-shelf ball detection + tracking. Public entrypoint ``run_stage3``
returns a :class:`courtvision.schemas.Ball2DSequence`.

Model: WASB (BMVC 2023), basketball-pretrained, off-the-shelf. WASB is
research code (``nttcom/WASB-SBDT``) with no PyPI package — it's cloned into
``/opt/wasb`` inside the dedicated Modal image (PRD §13.3 / ``flags.md`` F9)
and imported off ``sys.path`` at call time. All WASB imports are therefore
deferred into :func:`run_stage3` so this module imports cleanly on a laptop
with no GPU and no WASB checkout.

Pipeline (faithful to WASB's own ``runners/eval.py``):

1. Read frames with OpenCV, BGR→RGB.
2. Slide a non-overlapping window of ``frames_in`` (=3) frames (WASB "Step=3",
   the paper's default/best config).
3. Affine-warp each frame to 512x288, ImageNet-normalize, concat on channel
   dim → ``(B, 3*frames_in, 288, 512)``.
4. HRNet → per-frame heatmaps → blob postprocessor → candidate ``(xy, score)``
   in *original* image coordinates (via the inverse affine).
5. ``OnlineTracker`` picks one ball per frame using motion-prediction quality
   gated by ``max_disp``.

Only frames where the tracker reports the ball *visible* become
``BallDetection2D`` rows — invisible frames are simply absent (sparse output),
matching the ``Ball2DSequence`` contract in PRD §7.3.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from courtvision.logging import get_logger
from courtvision.schemas import Ball2DSequence, BallDetection2D

if TYPE_CHECKING:
    import numpy as np

_log = get_logger(__name__)

_DEFAULT_WASB_SRC = "/opt/wasb/src"
_DEFAULT_WEIGHTS = "/artifacts/checkpoints/wasb_basketball.pth"

# WASB config fragments to merge into the runtime cfg. Loaded from the cloned
# repo so we stay faithful to upstream defaults rather than hardcoding them.
_CFG_FRAGMENTS = {
    "model": "configs/model/wasb.yaml",
    "detector": "configs/detector/tracknetv2.yaml",
    "tracker": "configs/tracker/online.yaml",
    "dataloader": "configs/dataloader/default.yaml",
    "runner": "configs/runner/eval.yaml",
}


def _build_cfg(wasb_src: Path, config: dict[str, Any]) -> Any:
    """Compose the WASB cfg from the cloned repo's YAMLs + our overrides.

    WASB's detector/model/postprocessor/tracker all read a single nested cfg
    (``cfg['model']['frames_in']``, ``cfg['detector']['postprocessor'][...]``,
    etc.). We rebuild that structure by merging the upstream config fragments
    and overriding only what inference needs: the checkpoint path, single-GPU
    device list, score threshold, and tracker displacement gate.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {key: OmegaConf.load(str(wasb_src / rel)) for key, rel in _CFG_FRAGMENTS.items()}
    )

    weights = config.get("model", {}).get("weights") or _DEFAULT_WEIGHTS
    # PRD config historically called this ``conf_threshold``; WASB calls it
    # ``score_threshold``. Accept either; fall back to WASB's default (0.5).
    score_threshold = config.get("score_threshold", config.get("model", {}).get("conf_threshold"))

    overrides: dict[str, Any] = {
        "runner": {
            "device": "cuda",
            "gpus": [0],
            "vis_result": False,
            "vis_hm": False,
            "vis_traj": False,
        },
        "detector": {"model_path": str(weights)},
        "output_dir": None,
    }
    if score_threshold is not None:
        overrides["detector"] = {
            **overrides["detector"],
            "postprocessor": {"score_threshold": float(score_threshold)},
        }
    if "max_disp" in config:
        overrides["tracker"] = {"max_disp": int(config["max_disp"])}

    return OmegaConf.merge(cfg, overrides)


def _video_meta(video_path: Path) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return fps, n_frames


def run_stage3(video_path: Path, config: dict[str, Any]) -> Ball2DSequence:
    """Per-camera ball detection. Returns a :class:`Ball2DSequence`.

    Expects ``config`` shaped like ``configs/stage3_ball/wasb.yaml``:
    ``session_id``, ``camera_id``, ``model.{weights,conf_threshold}``, and
    optionally ``score_threshold``, ``max_disp``, ``batch_size``,
    ``max_seconds``, ``wasb_src``.

    Requires CUDA — WASB's detector asserts a GPU. Runs inside ``run_wasb`` on
    Modal (T4); never on the laptop.
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    session_id = str(config["session_id"])
    camera_id = str(config["camera_id"])
    batch_size = int(config.get("batch_size", 8))

    wasb_src = Path(config.get("wasb_src", _DEFAULT_WASB_SRC))
    if not wasb_src.exists():
        raise FileNotFoundError(
            f"WASB source not found at {wasb_src}; expected the dedicated WASB "
            "Modal image to git-clone nttcom/WASB-SBDT into /opt/wasb"
        )
    # WASB modules use top-level imports (``from models import ...``), so its
    # ``src`` dir must be on sys.path.
    if str(wasb_src) not in sys.path:
        sys.path.insert(0, str(wasb_src))

    from dataloaders import build_img_transforms
    from dataloaders.dataset_loader import get_transform
    from detectors import build_detector
    from trackers import build_tracker

    cfg = _build_cfg(wasb_src, config)
    frames_in = int(cfg["model"]["frames_in"])
    inp_wh = (int(cfg["model"]["inp_width"]), int(cfg["model"]["inp_height"]))
    out_wh = (int(cfg["model"]["out_width"]), int(cfg["model"]["out_height"]))

    fps, n_frames = _video_meta(video_path)
    max_seconds = config.get("max_seconds")
    max_frames = int(max_seconds * fps) if max_seconds and fps > 0 else None
    _log.info(
        "stage3: %s cam=%s fps=%.2f n_frames=%d frames_in=%d weights=%s",
        session_id,
        camera_id,
        fps,
        n_frames,
        frames_in,
        cfg["detector"]["model_path"],
    )

    detector = build_detector(cfg)
    tracker = build_tracker(cfg)
    _, transform = build_img_transforms(cfg)

    # The affine warp (orig→512x288) and its inverse (heatmap→orig) depend only
    # on frame dimensions, which are constant within a clip — compute once from
    # the first frame.
    trans_input: np.ndarray[Any, Any] | None = None
    trans_out_inv_t: torch.Tensor | None = None

    def _prep(frame_bgr: np.ndarray[Any, Any]) -> torch.Tensor:
        nonlocal trans_input, trans_out_inv_t
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if trans_input is None:
            trans_input = get_transform(rgb, inp_wh, inv=0)
            trans_out_inv = get_transform(rgb, out_wh, inv=1)
            trans_out_inv_t = torch.from_numpy(np.asarray(trans_out_inv, dtype=np.float32))
        warped = cv2.warpAffine(rgb, trans_input, inp_wh, flags=cv2.INTER_LINEAR)
        out_t: torch.Tensor = transform(Image.fromarray(warped))  # (3,H,W) ImageNet-norm
        return out_t

    # Detections per original frame index. Step=3 windowing visits each frame
    # exactly once, so each frame maps to a single candidate list.
    det_by_frame: dict[int, list[dict[str, Any]]] = {}
    frame_order: list[int] = []

    # Pending batch of windows: parallel lists of stacked tensors + the real
    # frame indices each output element maps to (None = padding frame).
    batch_tensors: list[torch.Tensor] = []
    batch_frame_ids: list[list[int | None]] = []

    @torch.no_grad()
    def _flush() -> None:
        if not batch_tensors:
            return
        assert trans_out_inv_t is not None
        imgs = torch.stack(batch_tensors, dim=0)  # (B, 3*frames_in, H, W)
        b = imgs.shape[0]
        affine_mats = {0: trans_out_inv_t.unsqueeze(0).repeat(b, 1, 1)}
        results, _ = detector.run_tensor(imgs, affine_mats)
        for bid in range(b):
            for eid, fidx in enumerate(batch_frame_ids[bid]):
                if fidx is None:
                    continue
                det_by_frame[fidx] = results[bid][eid]
        batch_tensors.clear()
        batch_frame_ids.clear()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    window_frames: list[torch.Tensor] = []
    window_ids: list[int | None] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            window_frames.append(_prep(frame))
            window_ids.append(frame_idx)
            frame_order.append(frame_idx)
            frame_idx += 1
            if len(window_frames) == frames_in:
                batch_tensors.append(torch.cat(window_frames, dim=0))
                batch_frame_ids.append(list(window_ids))
                window_frames, window_ids = [], []
                if len(batch_tensors) >= batch_size:
                    _flush()
        # Tail window: pad with the last real frame to reach frames_in.
        if window_frames:
            pad = window_frames[-1]
            while len(window_frames) < frames_in:
                window_frames.append(pad)
                window_ids.append(None)
            batch_tensors.append(torch.cat(window_frames, dim=0))
            batch_frame_ids.append(list(window_ids))
        _flush()
    finally:
        cap.release()

    # Online tracking over frames in temporal order → one ball per frame.
    tracker.refresh()
    detections: list[BallDetection2D] = []
    for fidx in frame_order:
        result = tracker.update(det_by_frame.get(fidx, []))
        if result["visi"]:
            detections.append(
                BallDetection2D(
                    frame_idx=fidx,
                    x=float(result["x"]),
                    y=float(result["y"]),
                    score=float(result["score"]),
                )
            )

    _log.info(
        "stage3: %s cam=%s detected ball in %d/%d frames",
        session_id,
        camera_id,
        len(detections),
        len(frame_order),
    )
    return Ball2DSequence(
        session_id=session_id,
        camera_id=camera_id,
        fps=fps if fps > 0 else 0.0,
        detections=detections,
    )


def diagnose_stage3(video_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Heatmap diagnostic for WASB ball detection (no thresholding bias).

    Runs the model exactly like :func:`run_stage3` but reads the *raw* sigmoid
    heatmap per frame instead of the postprocessor's thresholded blobs. Reports
    the per-frame heatmap-max distribution (a threshold sweep) and — if
    ``config['heatmap_video_path']`` is set — writes a heatmap-overlay video
    (per-frame-normalized JET colormap + the argmax marker + the absolute max
    value printed). This answers the only question that matters before tuning:
    does the model fire *on the ball* (and just weakly → lower the threshold)
    or not at all (→ domain gap, needs fine-tuning)?

    Returns a stats dict; does not write a ``Ball2DSequence``.
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    session_id = str(config.get("session_id", "clip"))
    camera_id = str(config.get("camera_id", "cam"))
    batch_size = int(config.get("batch_size", 8))

    wasb_src = Path(config.get("wasb_src", _DEFAULT_WASB_SRC))
    if not wasb_src.exists():
        raise FileNotFoundError(f"WASB source not found at {wasb_src}")
    if str(wasb_src) not in sys.path:
        sys.path.insert(0, str(wasb_src))

    from dataloaders import build_img_transforms
    from dataloaders.dataset_loader import get_transform
    from detectors import build_detector
    from utils.image import affine_transform

    cfg = _build_cfg(wasb_src, config)
    frames_in = int(cfg["model"]["frames_in"])
    inp_wh = (int(cfg["model"]["inp_width"]), int(cfg["model"]["inp_height"]))
    out_wh = (int(cfg["model"]["out_width"]), int(cfg["model"]["out_height"]))

    fps, _ = _video_meta(video_path)
    max_seconds = config.get("max_seconds")
    max_frames = int(max_seconds * fps) if max_seconds and fps > 0 else None

    detector = build_detector(cfg)
    _, transform = build_img_transforms(cfg)

    trans_input: np.ndarray[Any, Any] | None = None
    trans_out_inv: np.ndarray[Any, Any] | None = None
    trans_out_inv_t: torch.Tensor | None = None

    def _prep(frame_bgr: np.ndarray[Any, Any]) -> torch.Tensor:
        nonlocal trans_input, trans_out_inv
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if trans_input is None:
            trans_input = get_transform(rgb, inp_wh, inv=0)
            trans_out_inv = np.asarray(get_transform(rgb, out_wh, inv=1), dtype=np.float32)
        warped = cv2.warpAffine(rgb, trans_input, inp_wh, flags=cv2.INTER_LINEAR)
        out_t: torch.Tensor = transform(Image.fromarray(warped))
        return out_t

    hm_max_by_frame: dict[int, float] = {}
    peak_by_frame: dict[int, tuple[float, float]] = {}
    heatmaps: dict[int, np.ndarray[Any, Any]] = {}
    batch_tensors: list[torch.Tensor] = []
    batch_frame_ids: list[list[int | None]] = []

    @torch.no_grad()
    def _flush() -> None:
        nonlocal trans_out_inv_t
        if not batch_tensors:
            return
        assert trans_out_inv is not None
        if trans_out_inv_t is None:
            trans_out_inv_t = torch.from_numpy(trans_out_inv)
        imgs = torch.stack(batch_tensors, dim=0)
        b = imgs.shape[0]
        affine_mats = {0: trans_out_inv_t.unsqueeze(0).repeat(b, 1, 1)}
        _, hms_vis = detector.run_tensor(imgs, affine_mats)
        for bid in range(b):
            for eid, fidx in enumerate(batch_frame_ids[bid]):
                if fidx is None:
                    continue
                hm = hms_vis[bid][eid][0]["hm"]  # (out_h, out_w) sigmoid
                py, px = np.unravel_index(int(hm.argmax()), hm.shape)
                ox, oy = affine_transform(
                    np.array([float(px), float(py)], dtype=np.float32), trans_out_inv
                )
                hm_max_by_frame[fidx] = float(hm.max())
                peak_by_frame[fidx] = (float(ox), float(oy))
                heatmaps[fidx] = hm.astype(np.float16)
        batch_tensors.clear()
        batch_frame_ids.clear()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    window_frames: list[torch.Tensor] = []
    window_ids: list[int | None] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            window_frames.append(_prep(frame))
            window_ids.append(frame_idx)
            frame_idx += 1
            if len(window_frames) == frames_in:
                batch_tensors.append(torch.cat(window_frames, dim=0))
                batch_frame_ids.append(list(window_ids))
                window_frames, window_ids = [], []
                if len(batch_tensors) >= batch_size:
                    _flush()
        if window_frames:
            pad = window_frames[-1]
            while len(window_frames) < frames_in:
                window_frames.append(pad)
                window_ids.append(None)
            batch_tensors.append(torch.cat(window_frames, dim=0))
            batch_frame_ids.append(list(window_ids))
        _flush()
    finally:
        cap.release()

    maxes = [hm_max_by_frame[f] for f in sorted(hm_max_by_frame)]
    thresholds = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    frames_ge = {f"ge_{t}": int(sum(m >= t for m in maxes)) for t in thresholds}
    sorted_maxes = sorted(maxes)
    n = len(sorted_maxes)
    stats: dict[str, Any] = {
        "session_id": session_id,
        "camera_id": camera_id,
        "n_frames": n,
        "hm_max_min": float(sorted_maxes[0]) if n else 0.0,
        "hm_max_median": float(sorted_maxes[n // 2]) if n else 0.0,
        "hm_max_p90": float(sorted_maxes[int(0.9 * n)]) if n else 0.0,
        "hm_max_max": float(sorted_maxes[-1]) if n else 0.0,
        "hm_max_mean": float(sum(maxes) / n) if n else 0.0,
        "frames_ge_threshold": frames_ge,
    }

    hv = config.get("heatmap_video_path")
    if hv:
        out_path = Path(hv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            fps if fps > 0 else 30.0,
            (width, height),
        )
        fi = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and fi >= max_frames:
                    break
                hm = heatmaps.get(fi)
                if hm is not None:
                    hmf = hm.astype(np.float32)
                    # Per-frame normalized so the spatial pattern is visible
                    # regardless of absolute confidence; absolute max printed.
                    norm = hmf / max(float(hmf.max()), 1e-6)
                    hm_big = cv2.resize((norm * 255).astype(np.uint8), (width, height))
                    cm = cv2.applyColorMap(hm_big, cv2.COLORMAP_JET)
                    frame = cv2.addWeighted(frame, 0.6, cm, 0.4, 0)
                    px, py = peak_by_frame.get(fi, (0.0, 0.0))
                    cv2.drawMarker(
                        frame,
                        (int(px), int(py)),
                        (255, 255, 255),
                        cv2.MARKER_CROSS,
                        18,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"max {hm_max_by_frame.get(fi, 0.0):.3f}",
                        (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(frame)
                fi += 1
        finally:
            writer.release()
            cap.release()
        stats["heatmap_video"] = str(out_path)

    _log.info("stage3-diag: %s cam=%s %s", session_id, camera_id, stats)
    return stats


__all__ = ["diagnose_stage3", "run_stage3"]
