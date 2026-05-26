"""Logging + W&B helpers.

``get_logger(name)`` returns a rich-configured ``logging.Logger`` — use this
instead of ``print`` in library code (PRD §12.1). ``init_wandb`` is a thin
wrapper that no-ops if ``WANDB_MODE=disabled`` or the key is missing, so test
suites and local dry runs don't need a W&B account.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rich.logging import RichHandler

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a rich-configured logger. Use in place of ``print``."""
    _configure_root()
    return logging.getLogger(name)


def init_wandb(
    project: str = "courtvision",
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Initialise a W&B run; returns the run object (or ``None`` if disabled).

    Honors ``WANDB_MODE=disabled`` and falls back to disabled mode when no
    ``WANDB_API_KEY`` is present, so local debug runs work without a key.
    """
    import wandb  # heavy import, deferred

    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    if not os.environ.get("WANDB_API_KEY"):
        # No key — silently disable so tests pass without auth.
        os.environ["WANDB_MODE"] = "disabled"
        return None
    return wandb.init(project=project, name=run_name, config=config, tags=tags)


__all__ = ["get_logger", "init_wandb"]
