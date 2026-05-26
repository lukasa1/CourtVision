"""Stage 1 runner. PRD §10 Phase B."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from courtvision.io_utils import write_json
from courtvision.stage1_detection.yolo_bytetrack import run_stage1


@hydra.main(version_base=None, config_path="../configs/stage1_detection", config_name="yolov8m_bytetrack")
def main(cfg: DictConfig) -> None:
    video = Path(cfg.input)
    result = run_stage1(video, OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    out = Path(cfg.output)
    write_json(result, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
