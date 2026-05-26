"""Stage 3 eval — mAP on U Arizona, release-window recall on WASB basketball."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from courtvision.stage3_ball.eval import (
    evaluate_release_window_recall,
    evaluate_u_arizona,
)


@hydra.main(version_base=None, config_path="../configs/stage3_ball", config_name="eval")
def main(cfg: DictConfig) -> None:
    preds_dir = Path(cfg.preds_dir)
    gt_dir = Path(cfg.gt_dir)
    mode = cfg.get("mode", "map")
    if mode == "release_window_recall":
        result = evaluate_release_window_recall(preds_dir, gt_dir)
    else:
        result = evaluate_u_arizona(preds_dir, gt_dir)
    print(result)


if __name__ == "__main__":
    main()
