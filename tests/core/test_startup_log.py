import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from voicetotext.core.debug_logger import VoiceToTextConsoleHandler
from voicetotext.core.startup_log import (
    begin_startup_timing,
    initialize_session_log,
    is_owned_session,
    milestone,
)


def _file_handlers() -> list[logging.Handler]:
    return [
        handler for handler in logging.getLogger("voicetotext").handlers
        if isinstance(handler, RotatingFileHandler)
    ]


def _console_handlers() -> list[logging.Handler]:
    return [
        handler for handler in logging.getLogger("voicetotext").handlers
        if isinstance(handler, VoiceToTextConsoleHandler)
    ]


class TestStartupLog:
    def test_should_write_session_id_and_named_milestones(self, tmp_path) -> None:
        begin_startup_timing()
        session_dir = initialize_session_log(tmp_path)
        assert is_owned_session(session_dir)
        path = session_dir / "session.log"
        milestone("Tk-created")
        milestone("config-loaded")
        milestone("tray-visible")
        milestone("engine-load-started")
        milestone("model-loaded")
        milestone("warm-ready")
        logger = logging.getLogger("voicetotext")
        handlers = _file_handlers()
        for handler in handlers:
            handler.flush()
        text = path.read_text(encoding="utf-8")
        try:
            assert "session-start id=" in text
            for name in (
                "process-start",
                "Tk-created",
                "config-loaded",
                "tray-visible",
                "engine-load-started",
                "model-loaded",
                "warm-ready",
            ):
                assert f"milestone={name}" in text
            assert re.fullmatch(
                r"\d{8}T\d{6}Z-[0-9a-f]{8}", session_dir.name,
            )
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()

    def test_should_not_duplicate_handlers_on_reinitialization(
        self, tmp_path,
    ) -> None:
        logger = logging.getLogger("voicetotext")
        try:
            initialize_session_log(tmp_path / "first", console=True)
            initialize_session_log(tmp_path / "second", console=True)
            assert len(_file_handlers()) == 1
            assert len(_console_handlers()) == 1
        finally:
            for handler in _file_handlers() + _console_handlers():
                logger.removeHandler(handler)
                handler.close()

    def test_should_add_a_console_handler_despite_a_foreign_stream_handler(
        self, tmp_path: Path,
    ) -> None:
        logger = logging.getLogger("voicetotext")
        foreign_handler = logging.StreamHandler()
        logger.addHandler(foreign_handler)
        try:
            initialize_session_log(tmp_path / "log", console=True)

            assert len(_console_handlers()) == 1
        finally:
            logger.removeHandler(foreign_handler)
            for handler in _file_handlers() + _console_handlers():
                logger.removeHandler(handler)
                handler.close()
