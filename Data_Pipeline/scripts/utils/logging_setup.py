"""Shared logger configuration: console output plus a rotating log file."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from scripts.utils.config import PROJECT_ROOT

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def get_logger(name, log_file=None):
    """Return a logger that writes to the console and logs/pipeline.log.

    Inside Airflow, console output is captured in each task's log as well.
    """
    global _configured
    if not _configured:
        root = logging.getLogger("cops")
        root.setLevel(logging.INFO)
        formatter = logging.Formatter(_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

        path = Path(log_file) if log_file else PROJECT_ROOT / "logs" / "pipeline.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # Read-only file systems should not stop the pipeline.
            root.warning("Could not create log file %s: %s", path, exc)

        root.propagate = False
        _configured = True
    return logging.getLogger(f"cops.{name}")
