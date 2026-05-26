"""FineSports dataloader — transfer stage 3. PRD §9.5.2, §E4.

10K NBA clips, continuous video → synthesized 32-frame windows.
Positives ±5-frame jitter; negatives from non-action portions → ``background``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class FineSportsDataset:
    def __init__(self, root: Path, split: str, **kwargs: Any) -> None:
        raise NotImplementedError("Phase E4")


__all__ = ["FineSportsDataset"]
