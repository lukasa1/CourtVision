"""Path resolution sanity."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from courtvision.paths import artifacts_path, dataset_path, on_modal, repo_root


def test_repo_root_points_at_pyproject() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").exists()


def test_dataset_path_local() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MODAL_TASK_ID", None)
        os.environ.pop("CV_ENV", None)
        p = dataset_path("multisubjects")
    assert p.is_absolute()
    assert p.parts[-3:] == ("data", "mirror", "multisubjects")
    assert on_modal() is False


def test_dataset_path_modal() -> None:
    with patch.dict(os.environ, {"CV_ENV": "modal"}):
        p = dataset_path("multisubjects")
    assert p == Path("/data/multisubjects")


def test_artifacts_path_local() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MODAL_TASK_ID", None)
        os.environ.pop("CV_ENV", None)
        p = artifacts_path("checkpoints/yolo.pt")
    assert p.parts[-3:] == ("outputs", "checkpoints", "yolo.pt")


def test_artifacts_path_modal() -> None:
    with patch.dict(os.environ, {"CV_ENV": "modal"}):
        p = artifacts_path("checkpoints/yolo.pt")
    assert p == Path("/artifacts/checkpoints/yolo.pt")
