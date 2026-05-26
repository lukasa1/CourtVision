"""External-benchmark + ablation eval for Stage 5.

PRD §9.5.3: MultiSubjects top-1, FineSports mAP@tIoU={0.3, 0.5, 0.7},
ablation harness over Row 1 / Row 2 (Phase E sprint scope) plus stubbed
Rows 3 / 4 (weeks 3-4).
"""

from __future__ import annotations

from pathlib import Path


def evaluate_multisubjects(ckpt_path: Path) -> dict:
    raise NotImplementedError("Phase E3")


def evaluate_finesports(ckpt_path: Path) -> dict:
    raise NotImplementedError("Phase E4")


def run_ablation(ckpts: list[Path], test_set: Path) -> dict:
    raise NotImplementedError("Phase E5")


__all__ = ["evaluate_multisubjects", "evaluate_finesports", "run_ablation"]
