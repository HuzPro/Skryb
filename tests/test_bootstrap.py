from unittest.mock import MagicMock, call, patch

import pytest

from voicetotext.bootstrap import (
    _build_output,
    _load_engine_with_fallback,
    apply_settings,
    migrate_language,
    migrate_output_mode,
)
from voicetotext.output.clipboard_adapter import ClipboardAdapter
from voicetotext.output.clipboard_preserving_decorator import (
    ClipboardPreservingDecorator,
)
from voicetotext.output.output_mode import OutputMode
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from voicetotext.settings.settings_draft import ChangeSet
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


class TestMigrateLanguage:
    def test_should_rewrite_none_language_to_en(self):
        config = MagicMock()
        config.get.return_value = None

        migrate_language(config)

        config.set.assert_called_once_with("transcription.language", "en")
        config.save.assert_called_once()

    def test_should_rewrite_legacy_auto_language_to_en(self):
        config = MagicMock()
        config.get.return_value = "auto"

        migrate_language(config)

        config.set.assert_called_once_with("transcription.language", "en")
        config.save.assert_called_once()

    def test_should_leave_english_unchanged(self):
        config = MagicMock()
        config.get.return_value = "en"

        migrate_language(config)

        config.set.assert_not_called()
        config.save.assert_not_called()

    def test_should_leave_other_language_unchanged(self):
        config = MagicMock()
        config.get.return_value = "fr"

        migrate_language(config)

        config.set.assert_not_called()
        config.save.assert_not_called()


class TestBuildOutput:
    @patch("voicetotext.bootstrap.Win32ClipboardPreserver")
    def test_should_return_decorator_wrapping_paste_and_clipboard_with_preserver(
        self, mock_preserver_cls
    ):
        result = _build_output(OutputMode.DIRECT_PASTE)

        assert isinstance(result, ClipboardPreservingDecorator)
        assert isinstance(result._inner, PasteAndClipboardAdapter)
        mock_preserver_cls.assert_called_once()
        assert result._preserver is mock_preserver_cls.return_value

    def test_should_return_paste_and_clipboard_adapter_for_paste_and_clipboard_mode(
        self,
    ):
        result = _build_output(OutputMode.PASTE_AND_CLIPBOARD)

        assert isinstance(result, PasteAndClipboardAdapter)

    def test_should_return_clipboard_adapter_for_clipboard_only_mode(self):
        result = _build_output(OutputMode.CLIPBOARD_ONLY)

        assert isinstance(result, ClipboardAdapter)


class TestApplySettings:
    """Characterize the hot-reload dispatch: apply_settings routes each
    ChangeSet flag to the correct adapter swap so runtime changes take
    effect without an app restart.
    """

    def _make_changes(
        self,
        *,
        engine_reload_required: bool = False,
        hotkey_changed: bool = False,
        output_changed: bool = False,
    ) -> ChangeSet:
        return ChangeSet(
            engine_reload_required=engine_reload_required,
            hotkey_changed=hotkey_changed,
            output_changed=output_changed,
        )

    @patch("voicetotext.bootstrap._build_output")
    def test_should_rebuild_output_when_output_changed(self, mock_build_output):
        config = MagicMock()
        config.get.return_value = "clipboard_only"
        use_case = MagicMock()
        app = MagicMock()
        changes = self._make_changes(output_changed=True)

        apply_settings(changes, config, use_case, app)

        mock_build_output.assert_called_once_with(OutputMode.CLIPBOARD_ONLY)
        use_case.set_output.assert_called_once_with(mock_build_output.return_value)

    @patch("voicetotext.bootstrap._build_output")
    def test_should_not_rebuild_output_when_output_unchanged(self, mock_build_output):
        config = MagicMock()
        use_case = MagicMock()
        app = MagicMock()
        changes = self._make_changes(output_changed=False)

        apply_settings(changes, config, use_case, app)

        mock_build_output.assert_not_called()
        use_case.set_output.assert_not_called()

    @patch("voicetotext.bootstrap._build_hotkey_listener")
    def test_should_rebuild_hotkey_listener_when_hotkey_changed(
        self, mock_build_listener
    ):
        config = MagicMock()
        use_case = MagicMock()
        app = MagicMock()
        changes = self._make_changes(hotkey_changed=True)

        apply_settings(changes, config, use_case, app)

        mock_build_listener.assert_called_once_with(config, use_case)
        app.set_hotkey_listener.assert_called_once_with(
            mock_build_listener.return_value
        )

    @patch("voicetotext.bootstrap._build_hotkey_listener")
    def test_should_not_rebuild_hotkey_listener_when_hotkey_unchanged(
        self, mock_build_listener
    ):
        config = MagicMock()
        use_case = MagicMock()
        app = MagicMock()
        changes = self._make_changes(hotkey_changed=False)

        apply_settings(changes, config, use_case, app)

        mock_build_listener.assert_not_called()
        app.set_hotkey_listener.assert_not_called()

    @patch("voicetotext.bootstrap._build_hotkey_listener")
    @patch("voicetotext.bootstrap._build_output")
    def test_should_handle_multiple_changes_in_a_single_call(
        self, mock_build_output, mock_build_listener
    ):
        config = MagicMock()
        use_case = MagicMock()
        app = MagicMock()
        changes = self._make_changes(hotkey_changed=True, output_changed=True)

        apply_settings(changes, config, use_case, app)

        use_case.set_output.assert_called_once_with(mock_build_output.return_value)
        app.set_hotkey_listener.assert_called_once_with(
            mock_build_listener.return_value
        )
