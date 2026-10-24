"""File-based debug logging for Skryb.

The tray app and PyInstaller build suppress stdout, so this module
provides file-based logging for production diagnostics. The log file
is truncated on each startup to reflect the current run.
"""

import logging
import sys
from pathlib import Path


_APP_LOGGER_NAME = "skryb"
_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
_CONSOLE_FORMAT = "[%(levelname)-5s] %(message)s"


class SkrybFileHandler(logging.FileHandler):
    """Marker subclass identifying a file handler owned by this module.

    Subclassing lets :func:`setup_debug_logger` recognise its own
    handlers via :func:`isinstance` instead of monkey-patching a
    sentinel attribute onto stock :class:`logging.FileHandler` objects.
    """


class SkrybConsoleHandler(logging.StreamHandler):
    """Marker subclass identifying a stderr handler owned by this module."""


# Silence the ``skryb`` logger by default so tests and any caller
# that imports our modules without calling :func:`setup_debug_logger`
# don't dump diagnostics to stderr. The file handler installed by
# ``setup_debug_logger`` at startup adds to this so real-app logs still
# land in the debug file.
_root = logging.getLogger(_APP_LOGGER_NAME)
_root.addHandler(logging.NullHandler())
_root.propagate = False


def setup_debug_logger(log_path: Path, *, console: bool = False) -> None:
    """Configure the ``skryb`` logger to write DEBUG+ to a file.

    When *console* is ``True`` an additional ``StreamHandler`` at INFO
    level writes to stderr so the developer sees status messages when
    running from a terminal. The tray app and PyInstaller build pass
    ``console=False`` (the default) so production stays file-only.

    Idempotent: calling this twice will not attach duplicate handlers.
    Safe to call from ``bootstrap.build_app`` before any instrumented
    code runs.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_APP_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    _remove_previously_installed_handlers(logger)
    logger.addHandler(_build_file_handler(log_path))
    if console:
        logger.addHandler(_build_console_handler())

    # Don't propagate to the root logger; we own the format and file
    # for the ``skryb`` subtree.
    logger.propagate = False

    logger.info("debug logger initialized (file=%s, console=%s)", log_path, console)


def _remove_previously_installed_handlers(logger: logging.Logger) -> None:
    """Drop handlers this module installed on a prior call.

    Repeated calls happen on tests and first-run to main-run transitions.
    Stacking identical handlers multiplies writes; leaving a
    :class:`FileHandler` open would leak its file descriptor and block
    log rotation on Windows. Each dropped handler is closed immediately.
    """
    for existing in list(logger.handlers):
        if isinstance(existing, (SkrybFileHandler, SkrybConsoleHandler)):
            logger.removeHandler(existing)
            existing.close()


def _build_file_handler(log_path: Path) -> SkrybFileHandler:
    handler = SkrybFileHandler(str(log_path), mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def _build_console_handler() -> SkrybConsoleHandler:
    handler = SkrybConsoleHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    return handler
