"""Stage 6 eval — head-to-head forward vs. backward release rules.

STUB — weeks 3-4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_head_to_head(
    releases_forward: Path, releases_backward: Path, ground_truth: Path
) -> dict[str, Any]:
    raise NotImplementedError("weeks 3-4")


__all__ = ["evaluate_head_to_head"]
