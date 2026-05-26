"""Stage 1 eval — MOTA / IDF1 on TrackID3x3.

Local (Hydra)::

    uv run python scripts/stage1_eval.py \\
        tracks_dir=outputs/stage1/trackid_test \\
        gt_dir=data/mirror/trackid3x3/annotations

Modal::

    uv run modal run scripts/stage1_eval.py::run \\
        --tracks-relpath stage1/trackid_test \\
        --gt-relpath trackid3x3/annotations
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from courtvision.logging import get_logger
from courtvision.modal_app import VOLUMES, app, artifacts_volume, image
from courtvision.stage1_detection.eval import evaluate_trackid3x3

_log = get_logger(__name__)


def _summarise(result: dict, target_idf1: float) -> dict:
    idf1 = result["summary"].get("idf1")
    mota = result["summary"].get("mota")
    gate = "PASS" if idf1 is not None and idf1 >= target_idf1 else "FAIL"
    _log.info(
        "stage1 eval: n_sequences=%d MOTA=%.4f IDF1=%.4f (target>=%.2f) %s",
        result["n_sequences"],
        mota or float("nan"),
        idf1 or float("nan"),
        target_idf1,
        gate,
    )
    return {"mota": mota, "idf1": idf1, "gate": gate}


@hydra.main(version_base=None, config_path="../configs/stage1_detection", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Local entrypoint."""
    result = evaluate_trackid3x3(Path(cfg.tracks_dir), Path(cfg.gt_dir))
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    summary = _summarise(result, float(cfg.target_idf1))
    print(json.dumps({"output": str(out), **summary}))


@app.function(image=image, volumes=VOLUMES, timeout=3600, cpu=2)
def eval_stage1_trackid3x3(
    tracks_relpath: str, gt_relpath: str, target_idf1: float = 0.70
) -> dict:
    """CPU-only Modal eval: read tracks + GT off Volumes, write summary JSON.

    ``tracks_relpath`` is relative to ``/artifacts`` (e.g.
    ``stage1/trackid_test``). ``gt_relpath`` is relative to ``/data``
    (e.g. ``trackid3x3/annotations``).
    """
    tracks_dir = Path("/artifacts") / tracks_relpath
    gt_dir = Path("/data") / gt_relpath
    result = evaluate_trackid3x3(tracks_dir, gt_dir)

    out_relpath = f"stage1/eval_trackid3x3_{Path(tracks_relpath).name}.json"
    out_path = Path("/artifacts") / out_relpath
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=float))
    artifacts_volume.commit()

    summary = _summarise(result, target_idf1)
    return {"artifacts_relpath": out_relpath, **summary}


@app.local_entrypoint()
def run(tracks_relpath: str, gt_relpath: str, target_idf1: float = 0.70) -> None:
    """Modal entrypoint."""
    out = eval_stage1_trackid3x3.remote(tracks_relpath, gt_relpath, target_idf1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
