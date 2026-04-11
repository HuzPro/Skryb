"""File-based debug logging for VoiceToText.

The tray app and PyInstaller build suppress stdout, so this module
provides file-based logging for production diagnostics. The log file
is truncated on each startup to reflect the current run.
"""

import logging
import sys
from pathlib import Path


_APP_LOGGER_NAME = "voicetotext"
_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
_CONSOLE_FORMAT = "[%(levelname)-5s] %(message)s"


# Silence the ``voicetotext`` logger by default so tests and any caller
# that imports our modules without calling :func:`setup_debug_logger`
# don't dump diagnostics to stderr. The file handler installed by
# ``setup_debug_logger`` at startup overrides this — or rather adds to
# it — so real-app logs still land in the debug file.
_root = logging.getLogger(_APP_LOGGER_NAME)
_root.addHandler(logging.NullHandler())
_root.propagate = False


def setup_debug_logger(log_path: Path, *, console: bool = False) -> None:
    """Configure the ``voicetotext`` logger to write DEBUG+ to a file.

    When *console* is ``True`` an additional ``StreamHandler`` at INFO
    level writes to stderr so the developer sees status messages when
    running from a terminal.  The tray app / PyInstaller build passes
    ``console=False`` (the default) so production stays file-only.

    Idempotent: calling this twice will not attach duplicate handlers.
    Safe to call from ``bootstrap.build_app`` before any instrumented
    code runs.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_APP_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # Idempotency: drop any handler we previously attached so repeated
    # calls (e.g. tests, first-run → main-run transitions) do not stack
    # identical handlers and multiply writes.
    for existing in list(logger.handlers):
        if getattr(existing, "_voicetotext_debug", False):
            logger.removeHandler(existing)

    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_handler._voicetotext_debug = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        console_handler._voicetotext_debug = True  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    # Don't propagate to the root logger — we own the format and file
    # for the ``voicetotext`` subtree.
    logger.propagate = False

    logger.info("debug logger initialized (file=%s, console=%s)", log_path, console)
