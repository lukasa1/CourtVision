"""Stage 1 smoke — Phase A: assert stub raises NotImplementedError."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtvision.stage1_detection.yolo_bytetrack import run_stage1


def test_stage1_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="Phase B"):
        run_stage1(Path("/dev/null"), {})
