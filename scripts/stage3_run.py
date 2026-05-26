"""Stage 3 runner."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from courtvision.io_utils import write_json
from courtvision.stage3_ball.wasb_inference import run_stage3


@hydra.main(version_base=None, config_path="../configs/stage3_ball", config_name="wasb")
def main(cfg: DictConfig) -> None:
    video = Path(cfg.input)
    result = run_stage3(video, OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    out = Path(cfg.output)
    write_json(result, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
