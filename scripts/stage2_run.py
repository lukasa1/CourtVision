"""Stage 2 runner — RTMPose-l Halpe-26. PRD §10 Phase C.

Two entrypoints — same backing inference function:

Local (Hydra) — needs the OpenMMLab stack, which is Modal-only (F1/F17), so
this path only works in an environment that has mmpose installed::

    uv run python scripts/stage2_run.py \\
        input=data/mirror/<clip>.mp4 \\
        tracks=outputs/stage1/<session>/tracks_<session>_<cam>.json \\
        output=outputs/stage2/<session>/pose2d_<session>_<cam>.json

Modal (T4)::

    uv run modal run scripts/stage2_run.py::run \\
        --video-relpath trackid3x3/videos/Indoor/raw/<clip>.mp4 \\
        --tracks-relpath stage1/<session>/tracks_<session>_<cam>.json
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from courtvision.io_utils import read_json_as, write_json
from courtvision.logging import get_logger
from courtvision.modal_app import app, run_rtmpose
from courtvision.schemas import TrackSequence

_log = get_logger(__name__)

# Mirrors configs/stage2_pose/rtmpose_l_halpe26.yaml so the Modal entrypoint
# can build a config without Hydra inside the container.
_DEFAULT_MODEL = {
    "name": "rtmpose-l",
    "onnx_model": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip"
    ),
    "keypoint_format": "halpe26",
    "num_keypoints": 26,
    "input_size": [192, 256],
    "backend": "onnxruntime",
    "device": "cpu",
}


@hydra.main(
    version_base=None, config_path="../configs/stage2_pose", config_name="rtmpose_l_halpe26"
)
def main(cfg: DictConfig) -> None:
    """Local entrypoint: read tracks + clip, run inference, write Pose2DSequence."""
    from courtvision.stage2_pose.rtmpose_inference import run_stage2

    tracks = read_json_as(TrackSequence, Path(cfg.tracks))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    result = run_stage2(tracks, Path(cfg.input), cfg_dict)
    out = Path(cfg.output)
    write_json(result, out)
    _log.info("wrote %s — %d poses", out, len(result.poses))
    print(json.dumps({"output": str(out), "n_poses": len(result.poses)}))


@app.local_entrypoint()
def run(
    video_relpath: str,
    tracks_relpath: str,
    device: str = _DEFAULT_MODEL["device"],  # type: ignore[assignment]
) -> None:
    """Modal entrypoint: kick off Stage 2 inference (rtmlib/ONNX, CPU).

    ``video_relpath`` is relative to ``/data``; ``tracks_relpath`` to
    ``/artifacts`` (the TrackSequence written by Stage 1). The Modal function
    writes its Pose2DSequence JSON to the artifacts Volume and prints the relpath.
    """
    config = {"model": {**_DEFAULT_MODEL, "device": device}}
    out_relpath = run_rtmpose.remote(video_relpath, tracks_relpath, config)
    print(json.dumps({"artifacts_relpath": out_relpath}))


if __name__ == "__main__":
    main()
