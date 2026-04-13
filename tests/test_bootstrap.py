from unittest.mock import MagicMock, call, patch

import pytest

from voicetotext.bootstrap import (
    _build_output,
    _load_engine_with_fallback,
    migrate_output_mode,
)
from voicetotext.output.clipboard_adapter import ClipboardAdapter
from voicetotext.output.direct_paste_adapter import DirectPasteAdapter
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from voicetotext.transcription.model_info import migrate_removed_models


class TestLoadEngineWithFallback:
    @patch("voicetotext.bootstrap.create_engine")
    def test_should_return_engine_when_first_attempt_succeeds(self, mock_create):
        engine = MagicMock()
        mock_create.return_value = engine
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback(config, paths)

        assert result is engine
        engine.load_model.assert_called_once()

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_retry_on_cpu_when_default_load_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("CUDA fail")
        cpu_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback(config, paths)

        assert result is cpu_engine
        assert mock_create.call_args_list == [
            call(config, paths),
            call(config, paths, device_override="cpu"),
        ]

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_catch_os_error_from_missing_model_files(self, mock_create):
        broken_engine = MagicMock()
        broken_engine.load_model.side_effect = OSError("Can't load feature extractor")
        cpu_engine = MagicMock()
        mock_create.side_effect = [broken_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback(config, paths)

        assert result is cpu_engine

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_propagate_error_when_cpu_fallback_also_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("fail")
        cpu_engine = MagicMock()
        cpu_engine.load_model.side_effect = RuntimeError("fail")
        mock_create.side_effect = [gpu_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        with pytest.raises(RuntimeError):
            _load_engine_with_fallback(config, paths)


class TestMigrateRemovedModels:
    def test_should_rewrite_medium_to_large_v3_turbo(self):
        config = MagicMock()
        config.get.return_value = "medium"

        migrate_removed_models(config)

        config.set.assert_called_once_with(
            "transcription.model_size", "large-v3-turbo"
        )
        config.save.assert_called_once()

    def test_should_leave_other_models_alone(self):
        config = MagicMock()
        config.get.return_value = "small"

        migrate_removed_models(config)

        config.set.assert_not_called()
        config.save.assert_not_called()

    def test_should_leave_large_v3_turbo_alone(self):
        config = MagicMock()
        config.get.return_value = "large-v3-turbo"

        migrate_removed_models(config)

        config.set.assert_not_called()

    def test_should_leave_large_v3_alone(self):
        config = MagicMock()
        config.get.return_value = "large-v3"

        migrate_removed_models(config)

        config.set.assert_not_called()


class TestMigrateOutputMode:
    def test_should_rewrite_auto_paste_to_paste_and_clipboard(self):
        config = MagicMock()
        config.get.return_value = "auto_paste"

        migrate_output_mode(config)

        config.set.assert_called_once_with("output.mode", "paste_and_clipboard")
        config.save.assert_called_once()

    def test_should_leave_direct_paste_unchanged(self):
        config = MagicMock()
        config.get.return_value = "direct_paste"

        migrate_output_mode(config)

        config.set.assert_not_called()
        config.save.assert_not_called()

    def test_should_leave_clipboard_only_unchanged(self):
        config = MagicMock()
        config.get.return_value = "clipboard_only"

        migrate_output_mode(config)

        config.set.assert_not_called()
        config.save.assert_not_called()

    def test_should_leave_paste_and_clipboard_unchanged(self):
        config = MagicMock()
        config.get.return_value = "paste_and_clipboard"

        migrate_output_mode(config)

        config.set.assert_not_called()
        config.save.assert_not_called()


class TestBuildOutput:
    @patch("voicetotext.bootstrap.Win32ClipboardPreserver")
    def test_should_return_direct_paste_adapter_with_preserver_injected(
        self, mock_preserver_cls
    ):
        config = MagicMock()
        config.get.return_value = "direct_paste"

        result = _build_output(config)

        assert isinstance(result, DirectPasteAdapter)
        mock_preserver_cls.assert_called_once()
        assert result._preserver is mock_preserver_cls.return_value

    def test_should_return_paste_and_clipboard_adapter_for_paste_and_clipboard_mode(
        self,
    ):
        config = MagicMock()
        config.get.return_value = "paste_and_clipboard"

        result = _build_output(config)

        assert isinstance(result, PasteAndClipboardAdapter)

    def test_should_return_clipboard_adapter_for_clipboard_only_mode(self):
        config = MagicMock()
        config.get.return_value = "clipboard_only"

        result = _build_output(config)

        assert isinstance(result, ClipboardAdapter)

    @patch("voicetotext.bootstrap.Win32ClipboardPreserver")
    def test_should_default_to_direct_paste_for_unknown_mode(
        self, mock_preserver_cls
    ):
        config = MagicMock()
        config.get.return_value = "something_unexpected"

        result = _build_output(config)

        assert isinstance(result, DirectPasteAdapter)
