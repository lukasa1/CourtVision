"""Own-footage dataloader — transfer stage 4 (LP-FT).

STUB — implementation in weeks 3-4 (requires labeled releases).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class OwnFootageDataset:
    def __init__(self, root: Path, split: str, **kwargs: Any) -> None:
        raise NotImplementedError("weeks 3-4")


__all__ = ["OwnFootageDataset"]
