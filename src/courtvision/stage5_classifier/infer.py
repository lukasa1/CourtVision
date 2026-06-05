"""Sliding-window shot inference for Stage 5. PRD §9.5.3 / E5.

Slides a 32-frame window (stride 4) over a per-player pose sequence, renders
each window to a keypoint-heatmap volume (the same posec3d preprocessing used
in training), runs the trained PoseConv3D classifier, and NMS-peaks the
``P(shoot)+P(layup)`` signal into shot events → :class:`ShotEventSequence`.

Per ``flags.md`` F19 the **ball-ascent gate is OFF** (no reliable ball
detection); shot/no-shot is decided purely by the pose classifier's
probability peaks. Multi-camera late fusion is structured (``per_camera_probs``)
but exercised single-camera until own footage exists.

MMAction2 + torch imports are deferred (run inside ``poseconv3d_image``).
"""

from __future__ import annotations

from typing import Any

from courtvision.logging import get_logger
from courtvision.schemas import ShotEvent, ShotEventSequence

_log = get_logger(__name__)

# Head class order (matches training: MultiSubjects labels 0/1/2 + background).
_CLASS_ORDER = ("dribble", "layup", "shoot", "background")


def _nms_peaks(scores: list[float], centers: list[int], threshold: float, radius: int) -> list[int]:
    """Greedy 1-D NMS: keep the highest-scoring window, suppress others within
    ``radius`` frames of it, repeat. Returns kept indices in temporal order."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    suppressed: set[int] = set()
    for i in order:
        if i in suppressed or scores[i] < threshold:
            continue
        kept.append(i)
        for j in range(len(centers)):
            if abs(centers[j] - centers[i]) <= radius:
                suppressed.add(j)
    return sorted(kept, key=lambda i: centers[i])


def _segment_peaks(
    scores: list[float], centers: list[int], threshold: float, merge_gap: int
) -> list[int]:
    """Consolidate a bumpy detection signal into one event per shot.

    Groups windows with ``score >= threshold`` into segments — two above-
    threshold windows join the same segment if their centres are within
    ``merge_gap`` frames (so a short dip doesn't split one shot) — and keeps the
    single highest-scoring window per segment. More robust than greedy peak-NMS
    when a shot's wind-up makes a wide, bumpy plateau: it collapses the whole
    run into one event at the peak instead of emitting shoulder duplicates.
    """
    order = [i for i in sorted(range(len(centers)), key=lambda i: centers[i]) if scores[i] >= threshold]
    segments: list[list[int]] = []
    for i in order:
        if segments and centers[i] - centers[segments[-1][-1]] <= merge_gap:
            segments[-1].append(i)
        else:
            segments.append([i])
    keep = [max(seg, key=lambda i: scores[i]) for seg in segments]
    return sorted(keep, key=lambda i: centers[i])


def classify_windows(
    model: Any,
    keypoint: Any,  # np.ndarray [T, V=17, 2]
    keypoint_score: Any,  # np.ndarray [T, V=17]
    img_shape: tuple[int, int],
    *,
    clip_len: int = 64,
    stride: int = 8,
    frame_interval: int = 1,
) -> tuple[list[int], list[Any]]:
    """Run the classifier on each sliding window. Returns ``(centers, probs)``
    where ``centers[i]`` is the window's centre frame and ``probs[i]`` is the
    softmax over ``_CLASS_ORDER``. Shared by inference + overlay rendering.

    ``frame_interval`` subsamples within each window so each window covers
    ``clip_len * frame_interval`` raw frames but still feeds ``clip_len`` samples
    to the model. Required for deployment fps ≠ training fps (MultiSubjects was
    25 fps; iPhone footage at 60 fps needs ``frame_interval=2`` so each 64-sample
    window covers ~2.13 s rather than 1.07 s — see flags.md F27).
    """
    import numpy as np
    import torch

    # NB: mmaction2's Resize (pose-aware: scales keypoints + img_shape) — NOT
    # mmcv's image-only Resize, which leaves img_shape at the original frame
    # size and makes GeneratePoseTarget render full-res heatmaps (OOM).
    from mmaction.datasets.transforms import FormatShape, GeneratePoseTarget, PoseCompact, Resize

    pipeline = [
        PoseCompact(hw_ratio=1.0, allow_imgpad=True),
        Resize(scale=(56, 56), keep_ratio=False),
        GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False),
        FormatShape(input_format="NCTHW_Heatmap"),
    ]
    keypoint = np.asarray(keypoint, dtype=np.float32)
    keypoint_score = np.asarray(keypoint_score, dtype=np.float32)
    total = keypoint.shape[0]
    model.eval()
    device = next(model.parameters()).device

    # Each window spans `raw_span` source frames but feeds `clip_len` samples.
    raw_span = clip_len * frame_interval
    starts = list(range(0, max(1, total - raw_span + 1), stride)) or [0]
    win_probs: list[Any] = []
    centers: list[int] = []
    for s in starts:
        e = min(s + raw_span, total)
        # Stride-subsample to clip_len samples; .copy() is load-bearing because
        # PoseCompact (crop-shift) and Resize (scale) mutate results["keypoint"]
        # in place, which aliases this slice of the caller's array. Without the
        # copy, callers that reuse the array after inference (e.g. the overlay
        # renderer) get crop-shifted garbage coords.
        kp = keypoint[s:e:frame_interval].copy()
        ks = keypoint_score[s:e:frame_interval].copy()
        if kp.shape[0] < clip_len:  # pad short tail by repeating the last sample
            pad = clip_len - kp.shape[0]
            kp = np.concatenate([kp, np.repeat(kp[-1:], pad, axis=0)], axis=0)
            ks = np.concatenate([ks, np.repeat(ks[-1:], pad, axis=0)], axis=0)
        results: dict[str, Any] = {
            "keypoint": kp[None],  # [M=1, T, V, 2]
            "keypoint_score": ks[None],  # [M=1, T, V]
            "img_shape": img_shape,
            "modality": "Pose",
            "total_frames": clip_len,
            "start_index": 0,
            "num_clips": 1,  # we feed exact windows (no UniformSampleFrames), so set it
            "clip_len": clip_len,
        }
        for t in pipeline:
            results = t(results)
        x = torch.as_tensor(np.asarray(results["imgs"]), dtype=torch.float32, device=device)
        if x.ndim == 5 and x.shape[0] > 1:
            x = x[:1]
        with torch.inference_mode():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        win_probs.append(probs)
        centers.append((s + e) // 2)
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return centers, win_probs


def run_shot_inference(
    model: Any,
    keypoint: Any,  # np.ndarray [T, V=17, 2]
    keypoint_score: Any,  # np.ndarray [T, V=17]
    img_shape: tuple[int, int],
    *,
    session_id: str = "clip",
    camera_id: str = "cam",
    shooter_track_id: int = 0,
    clip_len: int = 64,
    stride: int = 8,
    frame_interval: int = 1,
    # Composed pipeline (Stage 5 → Stage 6 verification, flags.md F24): Stage 5
    # alone wants 0.75 to drop a confirmed dribble-drive FP at 0.65, but that
    # also drops the real S4T6 shot at 0.66. Lowering to 0.6 recovers the real
    # shot; Stage 6's `forward_conf=="high"` gate then filters the FPs that
    # come along (same pose input, different method → cross-method check).
    score_threshold: float = 0.6,
    merge_gap: int = 24,
) -> ShotEventSequence:
    """Sliding-window inference over one player's pose sequence.

    ``model`` is a built PoseConv3D (callable on a ``[B,17,T,56,56]`` heatmap
    tensor → ``[B,4]`` logits, e.g. from ``build_poseconv3d``).
    """
    centers, win_probs = classify_windows(
        model, keypoint, keypoint_score, img_shape,
        clip_len=clip_len, stride=stride, frame_interval=frame_interval,
    )
    total = len(keypoint)
    raw_span = clip_len * frame_interval
    starts = list(range(0, max(1, total - raw_span + 1), stride)) or [0]

    # NMS on P(shoot) only (index 2 of _CLASS_ORDER) — the target is normal
    # jump shots; layups are deprioritised, so they no longer drive shot events.
    # Ball-ascent gate OFF (no reliable ball detection; flags.md F19).
    shot_score = [float(p[2]) for p in win_probs]
    events: list[ShotEvent] = []
    for rank, i in enumerate(_segment_peaks(shot_score, centers, score_threshold, merge_gap)):
        p = win_probs[i]
        events.append(
            ShotEvent(
                event_id=f"{session_id}_{rank}",
                shooter_track_id=shooter_track_id,
                window_start_frame=int(starts[i]),
                window_end_frame=int(min(starts[i] + raw_span, total)),
                peak_frame=int(centers[i]),
                p_shoot=float(p[2]),
                p_layup=float(p[1]),
                p_dribble=float(p[0]),
                p_background=float(p[3]),
                per_camera_probs={camera_id: {c: float(p[k]) for k, c in enumerate(_CLASS_ORDER)}},
            )
        )
    _log.info(
        "stage5-infer: %s %d windows -> %d shot events", session_id, len(win_probs), len(events)
    )
    return ShotEventSequence(session_id=session_id, events=events)


__all__ = ["classify_windows", "run_shot_inference"]
