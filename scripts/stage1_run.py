"""Stage 1 runner. PRD §10 Phase B.

Two entrypoints — same backing inference function:

Local (Hydra)::

    uv run python scripts/stage1_run.py \\
        input=data/mirror/multisubjects/<clip>.mp4 \\
        session_id=ms_demo camera_id=cam0

Modal::

    uv run modal run scripts/stage1_run.py::run \\
        --video-relpath trackid3x3/test/<clip>.mp4 \\
        --session-id trackid_test --camera-id cam0
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from courtvision.io_utils import write_json
from courtvision.logging import get_logger
from courtvision.modal_app import app, run_yolo_bytetrack
from courtvision.stage1_detection.yolo_bytetrack import run_stage1

_log = get_logger(__name__)

# Default model knobs mirror configs/stage1_detection/yolov8m_bytetrack.yaml.
# We duplicate them here so the Modal entrypoint can construct a config
# without needing Hydra inside the Modal container (Hydra wants a cwd
# resolvable to the config tree, which gets awkward on the Modal side).
_DEFAULT_MODEL = {
    "name": "yolov8m",
    "weights": "/artifacts/checkpoints/yolov8m.pt",
    "classes": [0],
    "conf": 0.4,
    "iou": 0.5,
    "imgsz": 1280,
}
_DEFAULT_TRACKER = {"name": "bytetrack", "cfg": "bytetrack.yaml"}


@hydra.main(
    version_base=None, config_path="../configs/stage1_detection", config_name="yolov8m_bytetrack"
)
def main(cfg: DictConfig) -> None:
    """Local entrypoint: read a Hydra config, run inference, write JSON."""
    video = Path(cfg.input)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    result = run_stage1(video, cfg_dict)
    out = Path(cfg.output)
    write_json(result, out)
    _log.info(
        "wrote %s — %d tracks across %d frames", out, len(result.tracks), result.n_frames
    )
    print(json.dumps({"output": str(out), "n_tracks": len(result.tracks)}))


@app.local_entrypoint()
def run(
    video_relpath: str,
    session_id: str,
    camera_id: str,
    weights_path: str = _DEFAULT_MODEL["weights"],  # type: ignore[assignment]
    conf: float = _DEFAULT_MODEL["conf"],  # type: ignore[assignment]
    iou: float = _DEFAULT_MODEL["iou"],  # type: ignore[assignment]
    imgsz: int = _DEFAULT_MODEL["imgsz"],  # type: ignore[assignment]
) -> None:
    """Modal entrypoint: kick off Stage 1 inference on the GPU.

    ``video_relpath`` is relative to the data Volume root (e.g.
    ``trackid3x3/test/<clip>.mp4``). Returns nothing; the Modal function
    writes its TrackSequence JSON to the artifacts Volume and prints the
    relpath.
    """
    config = {
        "session_id": session_id,
        "camera_id": camera_id,
        "model": {**_DEFAULT_MODEL, "weights": weights_path, "conf": conf, "iou": iou, "imgsz": imgsz},
        "tracker": _DEFAULT_TRACKER,
    }
    out_relpath = run_yolo_bytetrack.remote(video_relpath, config)
    print(json.dumps({"artifacts_relpath": out_relpath}))


if __name__ == "__main__":
    main()
