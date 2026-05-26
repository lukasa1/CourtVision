"""Stage 5 external-benchmark eval — MultiSubjects / FineSports."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from courtvision.stage5_classifier.eval import (
    evaluate_finesports,
    evaluate_multisubjects,
)


@hydra.main(version_base=None, config_path="../configs/stage5_classifier", config_name="infer")
def main(cfg: DictConfig) -> None:
    ckpt = Path(cfg.ckpt)
    dataset = cfg.get("dataset", "multisubjects")
    if dataset == "finesports":
        result = evaluate_finesports(ckpt)
    else:
        result = evaluate_multisubjects(ckpt)
    print(result)


if __name__ == "__main__":
    main()
