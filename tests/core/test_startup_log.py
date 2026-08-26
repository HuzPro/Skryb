import logging

from voicetotext.core.startup_log import (
    SessionFileHandler,
    begin_startup_timing,
    initialize_session_log,
    milestone,
)


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
        handlers = [
            handler for handler in logger.handlers
            if isinstance(handler, SessionFileHandler)
        ]
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
