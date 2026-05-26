"""Stage 1 eval — MOTA / IDF1 on TrackID3x3.

Usage::

    uv run python scripts/stage1_eval.py \\
        tracks_dir=outputs/stage1/trackid_test \\
        gt_dir=data/mirror/trackid3x3/annotations
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from courtvision.logging import get_logger
from courtvision.stage1_detection.eval import evaluate_trackid3x3

_log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs/stage1_detection", config_name="eval")
def main(cfg: DictConfig) -> None:
    result = evaluate_trackid3x3(Path(cfg.tracks_dir), Path(cfg.gt_dir))
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))

    idf1 = result["summary"].get("idf1")
    mota = result["summary"].get("mota")
    target = float(cfg.target_idf1)
    gate = "PASS" if idf1 is not None and idf1 >= target else "FAIL"
    _log.info(
        "stage1 eval: n_sequences=%d MOTA=%.4f IDF1=%.4f (target>=%.2f) %s",
        result["n_sequences"],
        mota or float("nan"),
        idf1 or float("nan"),
        target,
        gate,
    )
    print(json.dumps({"output": str(out), "mota": mota, "idf1": idf1, "gate": gate}))


if __name__ == "__main__":
    main()
