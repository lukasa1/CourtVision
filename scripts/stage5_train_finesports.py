"""Stage 5 transfer-stage-3 training entrypoint — FineSports on A100-40GB."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from courtvision.stage5_classifier.train import train


@hydra.main(
    version_base=None,
    config_path="../configs/stage5_classifier",
    config_name="train_stage3_finesports",
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
