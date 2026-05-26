"""MultiSubjects dataloader — transfer stage 2. PRD §9.5.2."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MultiSubjectsDataset:
    """6,144 clips, BlazePose-33 → COCO-17, 32-frame windows.

    Head extended from 3 to 4 classes (add ``background`` zero-pad).
    """

    def __init__(self, root: Path, split: str, **kwargs: Any) -> None:
        raise NotImplementedError("Phase E2")


__all__ = ["MultiSubjectsDataset"]
