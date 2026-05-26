"""Ball-trajectory MLP stream. PRD §9.5.1.

2-3 layer MLP, hidden dim 64. Input: ball-z time series (len 32) + scalar
summaries (max_z, time-to-peak, peak_v_z, ascent rate, endpoint z values).

Trained in transfer-stage-5 (weeks 3-4). Architecture defined now so the
two-stream model wiring is stable.
"""

from __future__ import annotations

from typing import Any


def build_ball_mlp(input_dim: int = 37, hidden_dim: int = 64, output_dim: int = 64) -> Any:
    """Build the ball-trajectory MLP.

    Default ``input_dim`` = 32 (z time series) + 5 (scalar summaries).
    Returns a torch ``nn.Module`` whose output is concatenated with the pose
    stream features before the classification head.
    """
    raise NotImplementedError("Phase E (architecture); training weeks 3-4")


__all__ = ["build_ball_mlp"]
