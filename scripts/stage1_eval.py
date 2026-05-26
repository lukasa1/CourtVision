"""Stage 1 eval — MOTA / IDF1 on TrackID3x3."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from courtvision.stage1_detection.eval import evaluate_trackid3x3


@hydra.main(version_base=None, config_path="../configs/stage1_detection", config_name="eval")
def main(cfg: DictConfig) -> None:
    result = evaluate_trackid3x3(Path(cfg.tracks_dir), Path(cfg.gt_dir))
    print(result)


if __name__ == "__main__":
    main()
