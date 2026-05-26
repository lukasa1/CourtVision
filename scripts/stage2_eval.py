"""Stage 2 eval — wrist PCK + COCO-17 comparison."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from courtvision.stage2_pose.eval import evaluate_multisubjects, evaluate_trackid3x3


@hydra.main(version_base=None, config_path="../configs/stage2_pose", config_name="eval")
def main(cfg: DictConfig) -> None:
    poses_dir = Path(cfg.poses_dir)
    gt_dir = Path(cfg.gt_dir)
    subset = cfg.get("subset", "trackid3x3")
    if subset == "coco17":
        result = evaluate_multisubjects(poses_dir, gt_dir)
    else:
        result = evaluate_trackid3x3(poses_dir, gt_dir)
    print(result)


if __name__ == "__main__":
    main()
