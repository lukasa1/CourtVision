"""Stage 2 runner."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from courtvision.io_utils import read_json_as, write_json
from courtvision.schemas import TrackSequence
from courtvision.stage2_pose.rtmpose_inference import run_stage2


@hydra.main(
    version_base=None, config_path="../configs/stage2_pose", config_name="rtmpose_l_halpe26"
)
def main(cfg: DictConfig) -> None:
    tracks = read_json_as(TrackSequence, Path(cfg.tracks))
    result = run_stage2(tracks, Path(cfg.input), OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    out = Path(cfg.output)
    write_json(result, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
