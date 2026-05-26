"""JSON read/write with Pydantic schema validation.

Every stage output is a JSON file whose contents validate against a model in
``courtvision.schemas``. ``write_json`` round-trips through ``model_dump`` so
extra fields can't sneak in; ``read_json_as`` validates on load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def write_json(model_instance: BaseModel, path: str | Path, *, indent: int = 2) -> Path:
    """Serialize ``model_instance`` to ``path``. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model_instance.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=indent))
    return path


def read_json_as(model: type[M], path: str | Path) -> M:
    """Load + validate JSON at ``path`` against ``model``."""
    path = Path(path)
    payload = json.loads(path.read_text())
    return model.model_validate(payload)


__all__ = ["write_json", "read_json_as"]
