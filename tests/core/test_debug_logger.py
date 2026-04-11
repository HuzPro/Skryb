import logging

from voicetotext.core.debug_logger import setup_debug_logger


_APP_LOGGER_NAME = "voicetotext"


def _debug_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [
        h for h in logger.handlers if getattr(h, "_voicetotext_debug", False)
    ]


def _file_handlers(logger: logging.Logger) -> list[logging.FileHandler]:
    return [h for h in _debug_handlers(logger) if isinstance(h, logging.FileHandler)]


def _console_handlers(logger: logging.Logger) -> list[logging.StreamHandler]:
    return [
        h
        for h in _debug_handlers(logger)
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]


class TestSetupDebugLoggerFileHandler:
    def test_should_attach_file_handler(self, tmp_path):
        setup_debug_logger(tmp_path / "debug.log")

        logger = logging.getLogger(_APP_LOGGER_NAME)
        assert len(_file_handlers(logger)) == 1

    def test_should_set_logger_level_to_debug(self, tmp_path):
        setup_debug_logger(tmp_path / "debug.log")

        logger = logging.getLogger(_APP_LOGGER_NAME)
        assert logger.level == logging.DEBUG

    def test_should_create_parent_directories(self, tmp_path):
        log_path = tmp_path / "nested" / "deep" / "debug.log"

        setup_debug_logger(log_path)

        assert log_path.exists()


class TestSetupDebugLoggerConsoleHandler:
    def test_should_not_attach_console_handler_by_default(self, tmp_path):
        setup_debug_logger(tmp_path / "debug.log")

        logger = logging.getLogger(_APP_LOGGER_NAME)
        assert len(_console_handlers(logger)) == 0

    def test_should_attach_console_handler_when_console_true(self, tmp_path):
        setup_debug_logger(tmp_path / "debug.log", console=True)

        logger = logging.getLogger(_APP_LOGGER_NAME)
        consoles = _console_handlers(logger)
        assert len(consoles) == 1
        assert consoles[0].level == logging.INFO

    def test_should_not_attach_console_handler_when_console_false(self, tmp_path):
        setup_debug_logger(tmp_path / "debug.log", console=False)

        logger = logging.getLogger(_APP_LOGGER_NAME)
        assert len(_console_handlers(logger)) == 0


class TestSetupDebugLoggerIdempotency:
    def test_should_not_duplicate_handlers_on_repeated_calls(self, tmp_path):
        setup_debug_logger(tmp_path / "debug.log", console=True)
        setup_debug_logger(tmp_path / "debug.log", console=True)

        logger = logging.getLogger(_APP_LOGGER_NAME)
        assert len(_file_handlers(logger)) == 1
        assert len(_console_handlers(logger)) == 1

    def test_should_remove_console_handler_when_switching_to_file_only(self, tmp_path):
        setup_debug_logger(tmp_path / "debug.log", console=True)
        setup_debug_logger(tmp_path / "debug.log", console=False)

        logger = logging.getLogger(_APP_LOGGER_NAME)
        assert len(_console_handlers(logger)) == 0
