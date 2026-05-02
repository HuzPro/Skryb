import io
import sys
from unittest.mock import MagicMock, patch

from voicetotext.__main__ import _configured_model_installed, _show_error


class TestConfiguredModelInstalled:
    def test_should_return_true_when_model_present_on_disk(self):
        config = MagicMock()
        config.get.return_value = "base"
        model_manager = MagicMock()
        model_manager.is_installed.return_value = True

        assert _configured_model_installed(config, model_manager) is True
        model_manager.is_installed.assert_called_once_with("base")

    def test_should_return_false_when_model_missing_from_disk(self):
        config = MagicMock()
        config.get.return_value = "small"
        model_manager = MagicMock()
        model_manager.is_installed.return_value = False

        assert _configured_model_installed(config, model_manager) is False
        model_manager.is_installed.assert_called_once_with("small")

    def test_should_default_to_base_when_config_missing(self):
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: default
        model_manager = MagicMock()
        model_manager.is_installed.return_value = True

        _configured_model_installed(config, model_manager)

        model_manager.is_installed.assert_called_once_with("base")


class TestShowError:
    def test_should_write_message_to_stderr(self):
        captured = io.StringIO()
        mock_widgets = MagicMock()

        with (
            patch("sys.stderr", captured),
            patch.dict(sys.modules, {"PySide6.QtWidgets": mock_widgets}),
        ):
            _show_error("something broke")

        assert "something broke" in captured.getvalue()

    def test_should_write_to_stderr_even_when_qt_unavailable(self):
        captured = io.StringIO()

        with (
            patch("sys.stderr", captured),
            patch.dict(sys.modules, {"PySide6.QtWidgets": None}),
        ):
            _show_error("import failed")

        assert "import failed" in captured.getvalue()
