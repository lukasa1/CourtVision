"""Shared 32-frame window-sampling logic for Stage 5 dataloaders.

32 frames @ 60 fps ≈ 0.53 s, matching MMAction2's K400 PoseConv3D config.
"""

from __future__ import annotations


def sample_windows(n_frames: int, window: int = 32, stride: int = 4) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs covering ``n_frames`` with ``stride``."""
    raise NotImplementedError("Phase E2")


__all__ = ["sample_windows"]
