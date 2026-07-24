"""
Enterprise RAG Logger

Features
--------
✔ Console logging
✔ Rotating log files
✔ Rich formatted console
✔ Thread-safe
✔ Reusable throughout the project
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from config import LOG_FILE, LOG_LEVEL

# -----------------------------------------------------------------------------
# Create log directory
# -----------------------------------------------------------------------------

Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Console
# -----------------------------------------------------------------------------

console = Console()

# -----------------------------------------------------------------------------
# Logger Factory
# -----------------------------------------------------------------------------

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger.

    Example
    -------
    logger = get_logger(__name__)
    logger.info("Loading PDF")
    """

    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    logger.propagate = False

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # -------------------------------------------------------------------------
    # Console Handler
    # -------------------------------------------------------------------------

    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
    )

    console_handler.setLevel(
        getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    )

    console_formatter = logging.Formatter(
        "%(message)s"
    )

    console_handler.setFormatter(console_formatter)

    # -------------------------------------------------------------------------
    # File Handler
    # -------------------------------------------------------------------------

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(
        getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    )

    file_formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(filename)s:%(lineno)d | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(file_formatter)

    # -------------------------------------------------------------------------
    # Attach handlers
    # -------------------------------------------------------------------------

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    _LOGGERS[name] = logger

    return logger


# -----------------------------------------------------------------------------
# Global Logger
# -----------------------------------------------------------------------------

logger = get_logger("EnterpriseRAG")