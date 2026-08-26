import logging
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOGGER_NAME = "voicetotext"
_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
_started_at = time.perf_counter()
_session_id = ""
_session_handler: RotatingFileHandler | None = None


def begin_startup_timing() -> None:
    global _started_at
    _started_at = time.perf_counter()


def initialize_session_log(path: Path, *, console: bool = False) -> None:
    global _session_id, _session_handler
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    if _session_handler is not None:
        logger.removeHandler(_session_handler)
        _session_handler.close()
    _session_handler = RotatingFileHandler(
        path, maxBytes=1_000_000, backupCount=3, encoding="utf-8",
    )
    _session_handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(_session_handler)
    if console and not any(
        isinstance(item, logging.StreamHandler)
        and not isinstance(item, logging.FileHandler)
        for item in logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    logger.propagate = False
    _session_id = uuid.uuid4().hex
    logger.info("session-start id=%s", _session_id)
    milestone("process-start")


def milestone(name: str) -> None:
    logging.getLogger(_LOGGER_NAME).info(
        "milestone=%s elapsed_ms=%d session=%s",
        name,
        round((time.perf_counter() - _started_at) * 1000),
        _session_id,
    )
