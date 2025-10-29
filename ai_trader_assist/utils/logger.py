"""Logging helpers for CLI jobs."""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator, Tuple


# ---------------------------------------------------------------------------
# Custom log levels used across the pipeline.
# ---------------------------------------------------------------------------

STEP_LEVEL = logging.INFO + 1
RESULT_LEVEL = logging.INFO + 2
OK_LEVEL = logging.INFO + 3

logging.addLevelName(STEP_LEVEL, "STEP")
logging.addLevelName(RESULT_LEVEL, "RESULT")
logging.addLevelName(OK_LEVEL, "OK")


def _clear_handlers(logger: logging.Logger) -> None:
    """Remove existing handlers to avoid duplicate log lines."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = False


def _format_component(component: str, message: str) -> str:
    return f"[{component}] {message}"


def log_step(logger: logging.Logger, component: str, message: str) -> None:
    logger.log(STEP_LEVEL, _format_component(component, message))


def log_result(logger: logging.Logger, component: str, message: str) -> None:
    logger.log(RESULT_LEVEL, _format_component(component, message))


def log_ok(logger: logging.Logger, component: str, message: str) -> None:
    logger.log(OK_LEVEL, _format_component(component, message))


def log_warn(logger: logging.Logger, component: str, message: str) -> None:
    logger.warning(_format_component(component, message))


def log_error(logger: logging.Logger, component: str, message: str) -> None:
    logger.error(_format_component(component, message))


@contextmanager
def log_timed_stage(
    logger: logging.Logger,
    component: str,
    message: str,
) -> Iterator[Callable[[str], None]]:
    """Context manager that logs the start, results, and completion of a stage."""

    log_step(logger, component, message)
    start = perf_counter()
    try:
        yield lambda result_message: log_result(logger, component, result_message)
    except Exception:
        logger.exception(_format_component(component, "阶段执行失败"))
        raise
    else:
        duration = perf_counter() - start
        log_ok(logger, component, f"Completed in {duration:.2f}s")


def setup_logger(
    name: str = "run_daily",
    log_dir: str | Path = "storage/logs",
    level: int = logging.INFO,
    console_level: int | None = None,
) -> Tuple[logging.Logger, Path]:
    """Configure a logger that writes to both stdout and a rotating file.

    Parameters
    ----------
    name:
        Logical name of the logger, also used as the file prefix.
    log_dir:
        Directory where the log file will be stored. It will be created if missing.
    level:
        Logging level applied to the logger and the file handler.
    console_level:
        Optional level for the console handler. When ``None`` the console handler
        uses ``level``.

    Returns
    -------
    tuple
        A ``(logger, log_path)`` pair.
    """

    log_path_dir = Path(log_dir)
    log_path_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_path_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    _clear_handlers(logger)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler_level = console_level if console_level is not None else level
    stream_handler.setLevel(stream_handler_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_path
