"""Stage 2 eval — wrist PCK@0.05 (TrackID3x3, primary) + COCO-17 (MultiSubjects).

Local (Hydra)::

    uv run python scripts/stage2_eval.py \\
        poses_dir=outputs/stage2/trackid_test \\
        gt_dir=examples/trackid3x3/ground_truth/Indoor/pose

Modal::

    uv run modal run scripts/stage2_eval.py::run \\
        --poses-relpath stage2/trackid_test \\
        --gt-relpath trackid3x3/ground_truth/Indoor/pose
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from courtvision.logging import get_logger
from courtvision.modal_app import VOLUMES, app, artifacts_volume, image
from courtvision.stage2_pose.eval import evaluate_multisubjects, evaluate_trackid3x3

_log = get_logger(__name__)


def _evaluate(poses_dir: Path, gt_dir: Path, subset: str, **kw: float) -> dict:
    if subset == "coco17":
        return evaluate_multisubjects(
            poses_dir, gt_dir, frame_w=kw.get("frame_w", 1.0), frame_h=kw.get("frame_h", 1.0)
        )
    return evaluate_trackid3x3(poses_dir, gt_dir)


def _summarise(result: dict, target_wrist_pck: float) -> dict:
    summary = result["summary"]
    wrist_pck = summary.get("wrist_pck")
    gate = "PASS" if wrist_pck is not None and wrist_pck >= target_wrist_pck else "FAIL"
    _log.info(
        "stage2 eval: n_sequences=%d wrist_pck=%.4f (sided=%.4f, target>=%.2f) %s",
        result["n_sequences"],
        wrist_pck or float("nan"),
        summary.get("wrist_pck_sided", float("nan")),
        target_wrist_pck,
        gate,
    )
    if "all_keypoint_pck" in summary:
        _log.info("  all-keypoint PCK=%.4f", summary["all_keypoint_pck"])
    if "per_landmark" in summary:
        breakdown = "  ".join(f"{g}={v:.3f}" for g, v in summary["per_landmark"].items())
        _log.info("  per-landmark PCK: %s", breakdown)
    return {"wrist_pck": wrist_pck, "gate": gate}


@hydra.main(version_base=None, config_path="../configs/stage2_pose", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Local entrypoint."""
    result = _evaluate(
        Path(cfg.poses_dir),
        Path(cfg.gt_dir),
        cfg.get("subset", "trackid3x3"),
        frame_w=float(cfg.get("frame_w", 1.0)),
        frame_h=float(cfg.get("frame_h", 1.0)),
    )
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    summary = _summarise(result, float(cfg.target_wrist_pck))
    print(json.dumps({"output": str(out), **summary}))


@app.function(image=image, volumes=VOLUMES, timeout=3600, cpu=2)
def eval_stage2(
    poses_relpath: str,
    gt_relpath: str,
    subset: str = "trackid3x3",
    target_wrist_pck: float = 0.85,
    frame_w: float = 1.0,
    frame_h: float = 1.0,
) -> dict:
    """CPU-only Modal eval: read poses + GT off Volumes, write summary JSON.

    ``poses_relpath`` is relative to ``/artifacts``; ``gt_relpath`` to ``/data``.
    """
    poses_dir = Path("/artifacts") / poses_relpath
    gt_dir = Path("/data") / gt_relpath
    result = _evaluate(poses_dir, gt_dir, subset, frame_w=frame_w, frame_h=frame_h)

    out_relpath = f"stage2/eval_stage2_{subset}_{Path(poses_relpath).name}.json"
    out_path = Path("/artifacts") / out_relpath
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=float))
    artifacts_volume.commit()

    summary = _summarise(result, target_wrist_pck)
    return {"artifacts_relpath": out_relpath, **summary}


@app.local_entrypoint()
def run(
    poses_relpath: str,
    gt_relpath: str,
    subset: str = "trackid3x3",
    target_wrist_pck: float = 0.85,
    frame_w: float = 1.0,
    frame_h: float = 1.0,
) -> None:
    """Modal entrypoint."""
    out = eval_stage2.remote(
        poses_relpath, gt_relpath, subset, target_wrist_pck, frame_w, frame_h
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
