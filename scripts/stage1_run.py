"""Stage 1 runner. PRD §10 Phase B.

Usage::

    uv run python scripts/stage1_run.py \\
        input=data/mirror/multisubjects/<clip>.mp4 \\
        session_id=ms_demo camera_id=cam0
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from courtvision.io_utils import write_json
from courtvision.logging import get_logger
from courtvision.stage1_detection.yolo_bytetrack import run_stage1

_log = get_logger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/stage1_detection", config_name="yolov8m_bytetrack"
)
def main(cfg: DictConfig) -> None:
    video = Path(cfg.input)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    result = run_stage1(video, cfg_dict)
    out = Path(cfg.output)
    write_json(result, out)
    _log.info(
        "wrote %s — %d tracks across %d frames", out, len(result.tracks), result.n_frames
    )
    # Stdout sentinel so callers can grep for the result location.
    print(json.dumps({"output": str(out), "n_tracks": len(result.tracks)}))


if __name__ == "__main__":
    main()
