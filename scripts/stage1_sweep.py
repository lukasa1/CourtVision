"""Stage 1 ByteTrack coordinate-descent sweep.

Usage::

    # Baseline + all 5 axes on the dev split:
    uv run modal run scripts/stage1_sweep.py::run_dev_sweep

    # Final test-set evaluation with a chosen config:
    uv run modal run scripts/stage1_sweep.py::run_test \\
        --tag final --overrides-json '{"track_buffer": 90, ...}' \\
        --yolo-conf 0.4
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from courtvision.logging import get_logger
from courtvision.modal_app import VOLUMES, app, image, run_yolo_bytetrack
from courtvision.stage1_detection.sweep import (
    RunSummary,
    build_run_config,
    clip_name_to_session_camera,
    pick_best,
    set_by_path,
    summarise_eval_result,
)

_log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "configs" / "stage1_detection"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "stage1_sweep"


# ---------------------------------------------------------------------------
# Modal-side eval (CPU)
# ---------------------------------------------------------------------------


@app.function(image=image, volumes=VOLUMES, timeout=1800, cpu=2)
def eval_sweep_run(tag: str, gt_relpath: str) -> dict[str, Any]:
    """Eval all tracks under ``stage1_sweep/<tag>/`` against GT.

    Runs inside Modal so we don't have to pull tracks + GT locally.
    Returns the same shape as ``evaluate_trackid3x3`` so the caller can
    pass it to ``summarise_eval_result``.
    """
    from courtvision.stage1_detection.eval import evaluate_trackid3x3

    tracks_dir = Path("/artifacts") / f"stage1_sweep/{tag}"
    gt_dir = Path("/data") / gt_relpath
    return evaluate_trackid3x3(tracks_dir, gt_dir)


# ---------------------------------------------------------------------------
# Sweep helpers (local-side; live in this script to keep the modal app
# dependency-light).
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return dict(yaml.safe_load(f))


def _run_one_config(
    *,
    tag: str,
    base_config: dict[str, Any],
    clip_names: list[str],
    video_relpath_prefix: str,
    video_ext: str,
    gt_relpath: str,
    knob: str | None = None,
    value: Any | None = None,
) -> RunSummary:
    """Inference (parallel) + eval for one config across all clips."""
    inputs: list[tuple[str, dict[str, Any]]] = []
    for clip in clip_names:
        session_id, camera_id = clip_name_to_session_camera(clip)
        video_relpath = f"{video_relpath_prefix}/{clip}.{video_ext}"
        output_relpath = f"stage1_sweep/{tag}/tracks_{clip}.json"
        cfg = build_run_config(
            base_config,
            session_id=session_id,
            camera_id=camera_id,
            output_relpath=output_relpath,
        )
        inputs.append((video_relpath, cfg))

    _log.info("[%s] inference: %d clips, knob=%s value=%s", tag, len(inputs), knob, value)
    t0 = time.time()
    # Modal Function.starmap returns an iterable of returns; force eval.
    list(run_yolo_bytetrack.starmap(inputs))
    t_inf = time.time() - t0

    _log.info("[%s] inference done in %.1fs; evaluating", tag, t_inf)
    eval_result = eval_sweep_run.remote(tag=tag, gt_relpath=gt_relpath)
    summary = summarise_eval_result(tag=tag, knob=knob, value=value, eval_result=eval_result)
    _log.info(
        "[%s] IDF1 mean=%.4f std=%.4f n=%d MOTA=%.4f",
        tag,
        summary.idf1_mean,
        summary.idf1_std,
        summary.n_clips,
        summary.mota_mean,
    )
    return summary


def _write_results(
    summaries: list[RunSummary],
    axis_summaries: dict[str, list[RunSummary]],
    baseline: RunSummary,
    output_dir: Path,
    target_idf1: float,
) -> None:
    """Persist JSON + markdown + per-axis plots."""
    from courtvision.stage1_detection.report import plot_axis, write_markdown_table

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "all_runs.json").write_text(
        json.dumps([s.to_dict() for s in summaries], indent=2)
    )
    write_markdown_table(summaries, output_dir / "summary.md")
    for axis_name, axis_results in axis_summaries.items():
        plot_axis(
            axis_results,
            axis_name=axis_name,
            baseline_idf1=baseline.idf1_mean,
            target_idf1=target_idf1,
            output_path=output_dir / f"axis_{axis_name}.png",
        )
    _log.info("wrote sweep artifacts to %s", output_dir)


# ---------------------------------------------------------------------------
# Modal local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def run_dev_sweep(
    split_yaml: str = str(_CONFIG_DIR / "split.yaml"),
    sweep_yaml: str = str(_CONFIG_DIR / "sweep.yaml"),
    output_dir: str = str(_DEFAULT_OUTPUT_DIR),
    target_idf1: float = 0.70,
) -> None:
    """Coordinate-descent sweep across all axes on the dev split."""
    split = _load_yaml(Path(split_yaml))
    sweep = _load_yaml(Path(sweep_yaml))
    dev_clips: list[str] = list(split["dev"])
    video_prefix: str = split["video_relpath_prefix"]
    video_ext: str = split["video_ext"]
    gt_relpath: str = split["gt_relpath"]
    out_dir = Path(output_dir)

    # Start each axis from the current locked-in baseline config. After
    # each axis we update locked_baseline with the winning knob value.
    locked_baseline = dict(sweep["baseline"])
    all_summaries: list[RunSummary] = []
    axis_summaries: dict[str, list[RunSummary]] = {}

    # 1) Baseline run.
    _log.info("=== baseline ===")
    baseline_summary = _run_one_config(
        tag="baseline",
        base_config=locked_baseline,
        clip_names=dev_clips,
        video_relpath_prefix=video_prefix,
        video_ext=video_ext,
        gt_relpath=gt_relpath,
    )
    all_summaries.append(baseline_summary)

    # 2) Coordinate descent across axes.
    for axis in sweep["axes"]:
        axis_name: str = axis["name"]
        knob_path: str = axis["knob_path"]
        values: list[Any] = list(axis["values"])
        _log.info("=== axis: %s (values=%s) ===", axis_name, values)

        axis_runs: list[RunSummary] = []
        for v in values:
            tag = f"{axis_name}_{v}"
            cfg = set_by_path(locked_baseline, knob_path, v)
            summary = _run_one_config(
                tag=tag,
                base_config=cfg,
                clip_names=dev_clips,
                video_relpath_prefix=video_prefix,
                video_ext=video_ext,
                gt_relpath=gt_relpath,
                knob=axis_name,
                value=v,
            )
            axis_runs.append(summary)
            all_summaries.append(summary)
        axis_summaries[axis_name] = axis_runs

        # Lock in the winner before sweeping the next axis.
        best = pick_best(axis_runs)
        _log.info(
            "axis %s best: value=%s IDF1=%.4f (locking)",
            axis_name,
            best.value,
            best.idf1_mean,
        )
        locked_baseline = set_by_path(locked_baseline, knob_path, best.value)

    _write_results(all_summaries, axis_summaries, baseline_summary, out_dir, target_idf1)
    print(
        json.dumps(
            {
                "baseline_idf1": baseline_summary.idf1_mean,
                "best_config": locked_baseline,
                "output_dir": str(out_dir),
            },
            indent=2,
            default=str,
        )
    )


@app.local_entrypoint()
def run_test(
    tag: str,
    overrides_json: str = "{}",
    yolo_conf: float = 0.4,
    yolo_iou: float = 0.5,
    yolo_imgsz: int = 1280,
    split_yaml: str = str(_CONFIG_DIR / "split.yaml"),
    output_dir: str = str(_DEFAULT_OUTPUT_DIR),
    target_idf1: float = 0.70,
) -> None:
    """Single-config run on the held-out test split.

    Use this after run_dev_sweep, with the best config from the dev sweep,
    to get the final reportable IDF1 number.
    """
    overrides = json.loads(overrides_json)
    split = _load_yaml(Path(split_yaml))
    test_clips: list[str] = list(split["test"])
    base_config: dict[str, Any] = {
        "model": {"conf": yolo_conf, "iou": yolo_iou, "imgsz": yolo_imgsz},
        "tracker": {"overrides": overrides},
    }
    summary = _run_one_config(
        tag=f"test_{tag}",
        base_config=base_config,
        clip_names=test_clips,
        video_relpath_prefix=split["video_relpath_prefix"],
        video_ext=split["video_ext"],
        gt_relpath=split["gt_relpath"],
        knob="test",
        value=tag,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"test_{tag}.json").write_text(json.dumps(summary.to_dict(), indent=2))

    gate = "PASS" if summary.idf1_mean >= target_idf1 else "FAIL"
    print(
        json.dumps(
            {
                "tag": tag,
                "n_clips": summary.n_clips,
                "idf1_mean": summary.idf1_mean,
                "idf1_std": summary.idf1_std,
                "idf1_ci95": summary.idf1_ci95,
                "mota_mean": summary.mota_mean,
                "target_idf1": target_idf1,
                "gate": gate,
            },
            indent=2,
        )
    )
