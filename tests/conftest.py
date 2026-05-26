"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """A 5-frame 128x128 mp4 with no people. Used as inference smoke input.

    We deliberately render featureless rectangles on a black background: the
    point is to exercise the decode → track → schema pipeline, not detection
    accuracy. Expected ``run_stage1`` output is an empty track list.
    """
    out = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out), fourcc, 30.0, (128, 128))
    try:
        for i in range(5):
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            # Drift a coloured rectangle so the frames differ.
            x = 10 + i * 5
            cv2.rectangle(frame, (x, 30), (x + 30, 60), (0, 200, 0), -1)
            writer.write(frame)
    finally:
        writer.release()
    return out
