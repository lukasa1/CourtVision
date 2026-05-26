"""Stage 3 — WASB basketball detection. PRD §9.3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from courtvision.schemas import Ball2DSequence


def run_stage3(video_path: Path, config: dict[str, Any]) -> Ball2DSequence:
    """Per-camera ball detection. Returns ``Ball2DSequence``.

    Model: WASB (BMVC 2023), basketball-pretrained, off-the-shelf.
    Code via ``nttcom/WASB-SBDT``.
    """
    raise NotImplementedError("Phase D")


__all__ = ["run_stage3"]
