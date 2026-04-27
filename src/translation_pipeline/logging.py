from __future__ import annotations

import logging
from typing import Final

from rich.logging import RichHandler

_CONFIGURED: Final[str] = "_translation_pipeline_logging_configured"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Install a Rich-based handler on the root logger. Idempotent.

    Default level INFO; DEBUG when verbose=True. The CLI plumbs --verbose into this in Phase 4.
    """
    root = logging.getLogger()
    if not getattr(root, _CONFIGURED, False):
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        setattr(root, _CONFIGURED, True)

    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logging.getLogger("translation_pipeline")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger under translation_pipeline."""
    if name is None:
        return logging.getLogger("translation_pipeline")
    return logging.getLogger(f"translation_pipeline.{name}")
