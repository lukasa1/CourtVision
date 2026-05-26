"""Stage 3 smoke — Phase A: assert stub raises NotImplementedError."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtvision.stage3_ball.wasb_inference import run_stage3


def test_stage3_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="Phase D"):
        run_stage3(Path("/dev/null"), {})
