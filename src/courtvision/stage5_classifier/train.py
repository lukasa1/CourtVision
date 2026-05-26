"""Hydra-driven training entrypoint for Stage 5. Handles MultiSubjects
(transfer stage 2) and FineSports (transfer stage 3) via config swap.

PRD §9.5.3.
"""

from __future__ import annotations

from typing import Any


def train(cfg: Any) -> None:
    raise NotImplementedError("Phase E2")


__all__ = ["train"]
