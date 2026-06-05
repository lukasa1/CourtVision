"""Subject-disjoint train/val/test split for MultiSubjects.

The same person never appears in two splits, so the held-out test measures
generalisation to **unseen subjects** (not just unseen clips of seen subjects).
Random subject shuffle naturally preserves the ~38/13/49 dribble/layup/shoot
class proportions across splits because >80% of subjects span >1 action class
(verified at split time; see ``docs/stage5_training_method.md``).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def _subject_id(frame_dir: str) -> str:
    """MS clip names are ``<subject>_<action_letter>_<idx>``; the first
    underscore-separated token is the subject ID (e.g. ``442_s_3`` → ``"442"``)."""
    return frame_dir.split("_", 1)[0]


def compute_subject_disjoint_split(
    items: list[dict[str, Any]],
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[str]]:
    """Shuffle subjects with ``seed``, allocate ``ratios``, and return
    ``{"train": [frame_dirs], "val": [...], "test": [...]}``. Deterministic."""
    by_subj: dict[str, list[str]] = defaultdict(list)
    for x in items:
        by_subj[_subject_id(x["frame_dir"])].append(x["frame_dir"])
    subjects = sorted(by_subj.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_tr = round(n * ratios[0])
    n_va = round(n * ratios[1])
    train_s = set(subjects[:n_tr])
    val_s = set(subjects[n_tr : n_tr + n_va])
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for s, fds in by_subj.items():
        bucket = "train" if s in train_s else "val" if s in val_s else "test"
        out[bucket].extend(fds)
    return out


__all__ = ["compute_subject_disjoint_split"]
