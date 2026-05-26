"""Stage 5 smoke — Phase A: assert stubs raise NotImplementedError."""

from __future__ import annotations

import pytest

from courtvision.stage5_classifier.model.poseconv3d import build_poseconv3d


def test_build_poseconv3d_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="Phase E1"):
        build_poseconv3d()
