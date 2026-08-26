import logging
from logging.handlers import RotatingFileHandler

from voicetotext.core.startup_log import (
    begin_startup_timing,
    initialize_session_log,
    milestone,
)


def _file_handlers() -> list[logging.Handler]:
    return [
        handler for handler in logging.getLogger("voicetotext").handlers
        if isinstance(handler, RotatingFileHandler)
    ]


class TestStartupLog:
    def test_should_write_session_id_and_named_milestones(self, tmp_path) -> None:
        path = tmp_path / "session.log"
        begin_startup_timing()
        initialize_session_log(path)
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
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()

    def test_should_not_duplicate_handlers_on_reinitialization(
        self, tmp_path,
    ) -> None:
        logger = logging.getLogger("voicetotext")
        try:
            initialize_session_log(tmp_path / "first.log", console=True)
            initialize_session_log(tmp_path / "second.log", console=True)
            assert len(_file_handlers()) == 1
            console_handlers = [
                handler for handler in logger.handlers
                if isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
            ]
            assert len(console_handlers) == 1
        finally:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
