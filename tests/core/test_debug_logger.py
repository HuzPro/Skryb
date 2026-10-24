import logging

import pytest

from skryb.core.debug_logger import (
    SkrybConsoleHandler,
    SkrybFileHandler,
    setup_debug_logger,
)


_APP_LOGGER_NAME = "skryb"
_OwnedHandler = (SkrybFileHandler, SkrybConsoleHandler)


@pytest.fixture(autouse=True)
def _release_debug_handlers():
    """Close and detach any debug handlers attached by a test.

    Without this, FileHandler instances pointing at ``tmp_path`` leak
    into later tests (which pytest flags as
    ``PytestUnraisableExceptionWarning``) and pollute test isolation.
    """
    yield
    logger = logging.getLogger(_APP_LOGGER_NAME)
    for handler in list(logger.handlers):
        if isinstance(handler, _OwnedHandler):
            logger.removeHandler(handler)
            handler.close()


def _file_handlers(logger: logging.Logger) -> list[SkrybFileHandler]:
    return [h for h in logger.handlers if isinstance(h, SkrybFileHandler)]


def _console_handlers(logger: logging.Logger) -> list[SkrybConsoleHandler]:
    return [h for h in logger.handlers if isinstance(h, SkrybConsoleHandler)]


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
