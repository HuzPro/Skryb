import logging
import time
import uuid
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from voicetotext.core.debug_logger import VoiceToTextConsoleHandler

_LOGGER_NAME = "voicetotext"
_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
_started_at = time.perf_counter()
_started_at_utc = datetime.now(UTC)
_session_id = ""
_session_handler: RotatingFileHandler | None = None


def begin_startup_timing() -> None:
    global _started_at, _started_at_utc
    _started_at = time.perf_counter()
    _started_at_utc = datetime.now(UTC)


def initialize_session_log(
    diagnostics_dir: Path, *, console: bool = False,
) -> Path:
    global _session_id, _session_handler
    _session_id = uuid.uuid4().hex
    session_name = (
        f"{_started_at_utc.strftime('%Y%m%dT%H%M%SZ')}-{_session_id[:8]}"
    )
    session_dir = diagnostics_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    path = session_dir / "session.log"
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
        isinstance(item, VoiceToTextConsoleHandler) for item in logger.handlers
    ):
        console_handler = VoiceToTextConsoleHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    logger.propagate = False
    logger.info("session-start id=%s", _session_id)
    milestone("process-start")
    return session_dir


def milestone(name: str) -> None:
    logging.getLogger(_LOGGER_NAME).info(
        "milestone=%s elapsed_ms=%d session=%s",
        name,
        round((time.perf_counter() - _started_at) * 1000),
        _session_id,
    )


def process_started_at() -> datetime:
    return _started_at_utc


def process_uptime_seconds() -> float:
    return time.perf_counter() - _started_at


def session_id() -> str:
    return _session_id
