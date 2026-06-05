"""Stage 5 smoke — Phase E.

The model factory + inference need MMAction2 + a GPU (the dedicated
``poseconv3d_image``), so their full paths run on Modal, not here. Locally we
cover what's GPU-free: clean module import (heavy imports deferred), the NMS
peak-picker, and the ShotEvent schema round-trip.
"""

from __future__ import annotations

from pathlib import Path

from courtvision.io_utils import read_json_as, write_json
from courtvision.schemas import ShotEvent, ShotEventSequence


def test_model_and_infer_import_without_mmaction() -> None:
    """Both modules import on a laptop (no mmaction/torch at module scope)."""
    from courtvision.stage5_classifier import infer
    from courtvision.stage5_classifier.model import build_poseconv3d

    assert callable(build_poseconv3d)
    assert callable(infer.run_shot_inference)
    assert not hasattr(infer, "torch")  # heavy imports are deferred into functions


def test_nms_peaks_picks_local_maxima() -> None:
    """The shot-event NMS keeps the top peak and suppresses neighbours within radius."""
    from courtvision.stage5_classifier.infer import _nms_peaks

    # two clear peaks at centers 10 and 50, plus a near-neighbour of the first.
    scores = [0.9, 0.95, 0.2, 0.85]
    centers = [8, 10, 30, 50]
    kept = _nms_peaks(scores, centers, threshold=0.5, radius=16)
    assert kept == [1, 3]  # center 10 (suppresses 8) and center 50; 30 below threshold


def test_shot_event_sequence_round_trip(tmp_path: Path) -> None:
    seq = ShotEventSequence(
        session_id="rt",
        events=[
            ShotEvent(
                event_id="rt_0",
                shooter_track_id=0,
                window_start_frame=0,
                window_end_frame=32,
                peak_frame=16,
                p_shoot=0.91,
                p_layup=0.02,
                p_dribble=0.05,
                p_background=0.02,
                per_camera_probs={"cam": {"shoot": 0.91}},
            )
        ],
    )
    p = tmp_path / "shotevents_rt.json"
    write_json(seq, p)
    assert read_json_as(ShotEventSequence, p) == seq
