"""Environment-aware path resolution. See PRD §8.4.

On Modal, datasets live at ``/data/<name>`` and artifacts at ``/artifacts/...``.
Locally, a tiny mirror lives at ``data/mirror/<name>`` for dataloader
iteration; artifacts mirror to ``outputs/``. Stage code calls
``dataset_path("multisubjects")`` and gets the right one for the environment.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path


@cache
def repo_root() -> Path:
    """Repo root, resolved by walking up from this file until a ``pyproject.toml`` is found."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"could not find repo root from {p}")


def on_modal() -> bool:
    """True when running inside a Modal function."""
    return "MODAL_TASK_ID" in os.environ or os.environ.get("CV_ENV") == "modal"


def dataset_path(name: str) -> Path:
    """Path to a dataset by short name (e.g. ``"multisubjects"``).

    Returns ``/data/<name>`` on Modal, ``data/mirror/<name>`` locally.
    Does not create the directory.
    """
    if on_modal():
        return Path("/data") / name
    return repo_root() / "data" / "mirror" / name


def artifacts_path(relpath: str = "") -> Path:
    """Path under the artifacts volume (Modal) or local outputs/ mirror.

    ``relpath`` is relative — e.g. ``"checkpoints/yolov8m.pt"`` or
    ``"stage1/session42/tracks_cam1.json"``. Does not create the parent.
    """
    base = Path("/artifacts") if on_modal() else repo_root() / "outputs"
    return base / relpath if relpath else base


__all__ = ["artifacts_path", "dataset_path", "on_modal", "repo_root"]
