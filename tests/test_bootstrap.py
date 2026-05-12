from unittest.mock import MagicMock, call, patch

import pytest

from voicetotext.bootstrap import (
    _build_output,
    _load_engine_and_attach,
    _load_engine_with_fallback,
    _make_destroyed_handler,
    apply_settings,
    migrate_language,
    migrate_output_mode,
)
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.output.clipboard_adapter import ClipboardAdapter
from voicetotext.output.clipboard_preserving_decorator import (
    ClipboardPreservingDecorator,
)
from voicetotext.output.gui_thread_output_decorator import (
    GuiThreadOutputDecorator,
)
from voicetotext.output.output_mode import OutputMode
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from voicetotext.settings.settings_draft import ChangeSet
from voicetotext.transcription.model_info import migrate_removed_models


_HARDWARE = MagicMock()
_REGISTRY = MagicMock()


class TestLoadEngineWithFallback:
    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_return_engine_when_first_attempt_succeeds(self, mock_create):
        engine = MagicMock()
        mock_create.return_value = engine

        result = _load_engine_with_fallback(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
        )

        assert result is engine
        engine.load_model.assert_called_once()

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_retry_on_cpu_when_default_load_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("CUDA fail")
        cpu_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback(config, paths, _REGISTRY, _HARDWARE)

        assert result is cpu_engine
        first_call, second_call = mock_create.call_args_list
        assert first_call.kwargs.get("device_override") is None
        assert second_call.kwargs == {"device_override": "cpu"}

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_catch_os_error_from_missing_model_files(self, mock_create):
        broken_engine = MagicMock()
        broken_engine.load_model.side_effect = OSError("Can't load feature extractor")
        cpu_engine = MagicMock()
        mock_create.side_effect = [broken_engine, cpu_engine]

        result = _load_engine_with_fallback(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
        )

        assert result is cpu_engine

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_propagate_error_when_cpu_fallback_also_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("fail")
        cpu_engine = MagicMock()
        cpu_engine.load_model.side_effect = RuntimeError("fail")
        mock_create.side_effect = [gpu_engine, cpu_engine]

        with pytest.raises(RuntimeError):
            _load_engine_with_fallback(
                MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            )

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_warm_engine_up_after_successful_load(self, mock_create):
        # Warmup runs one inference pass right after ``load_model`` so
        # ROCm/CUDA kernels are compiled before the user's first hotkey
        # press — without it, the first transcription pays a one-off
        # multi-second penalty even after the model is fully resident
        # in VRAM.
        engine = MagicMock()
        mock_create.return_value = engine

        _load_engine_with_fallback(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
        )

        engine.load_model.assert_called_once()
        engine.warmup.assert_called_once()
        assert (
            engine.method_calls.index(call.warmup())
            > engine.method_calls.index(call.load_model())
        ), "warmup must follow load_model, never precede it"

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_swallow_warmup_failures_without_dropping_engine(
        self, mock_create,
    ):
        # Warmup is a latency optimisation — if it fails (e.g. a backend
        # quirk on silent audio), the engine is still usable and the
        # first real transcription just pays the cold-start penalty.
        # The app must continue, not flip to CPU fallback or crash.
        engine = MagicMock()
        engine.warmup.side_effect = RuntimeError("warmup crash")
        mock_create.return_value = engine

        result = _load_engine_with_fallback(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
        )

        assert result is engine
        engine.load_model.assert_called_once()
        engine.warmup.assert_called_once()

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_warm_cpu_fallback_engine_too(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("GPU OOM")
        cpu_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine]

        result = _load_engine_with_fallback(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
        )

        assert result is cpu_engine
        cpu_engine.load_model.assert_called_once()
        cpu_engine.warmup.assert_called_once()

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_never_silently_switch_engines_on_fallback(self, mock_create):
        """CPU fallback must reuse the same engine_id, only changing device.

        Cross-engine fallback would download different model files and
        re-introduce the surprise-re-download bug `feedback_single_engine`
        was created to prevent.
        """
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("fail")
        cpu_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        _load_engine_with_fallback(config, paths, _REGISTRY, _HARDWARE)

        # Both calls go through the SAME factory function, which routes
        # to the SAME engine_id from config — only device_override differs.
        first_call, second_call = mock_create.call_args_list
        assert first_call.args == (config, paths, _REGISTRY, _HARDWARE)
        assert second_call.args == (config, paths, _REGISTRY, _HARDWARE)
        assert second_call.kwargs == {"device_override": "cpu"}


class TestLoadEngineAndAttach:
    """The background-load worker swaps the placeholder for the real
    engine and transitions ``LOADING → IDLE`` on success, or
    ``LOADING → ERROR`` on failure. Tests run the worker on the
    calling thread — bootstrap spawns the thread separately."""

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_attach_loaded_engine_to_use_case(self, mock_create):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        loaded_engine = MagicMock()
        mock_create.return_value = loaded_engine
        use_case = MagicMock()

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            use_case=use_case, state_machine=state_machine,
        )

        use_case.set_engine.assert_called_once_with(loaded_engine)

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_transition_to_idle_on_successful_load(self, mock_create):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        mock_create.return_value = MagicMock()

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            use_case=MagicMock(), state_machine=state_machine,
        )

        assert state_machine.state == AppState.IDLE

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_transition_to_error_on_load_failure(self, mock_create):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        broken = MagicMock()
        broken.load_model.side_effect = RuntimeError("VRAM exhausted")
        mock_create.side_effect = [broken, broken]
        use_case = MagicMock()

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            use_case=use_case, state_machine=state_machine,
        )

        assert state_machine.state == AppState.ERROR
        use_case.set_engine.assert_not_called()

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_not_propagate_load_exception(self, mock_create):
        # Exceptions raised on the background thread would otherwise
        # vanish into thread-default handling; the worker swallows the
        # exception after transitioning the state machine to ERROR so
        # the app's diagnostics (state subscriber + logger) own the
        # failure surface.
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        broken = MagicMock()
        broken.load_model.side_effect = RuntimeError("boom")
        mock_create.side_effect = [broken, broken]

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            use_case=MagicMock(), state_machine=state_machine,
        )


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
        dispatcher = MagicMock()

        result = _build_output(OutputMode.DIRECT_PASTE, dispatcher)

        assert isinstance(result, GuiThreadOutputDecorator)
        clipboard_preserving = result._inner
        assert isinstance(clipboard_preserving, ClipboardPreservingDecorator)
        assert isinstance(clipboard_preserving._inner, PasteAndClipboardAdapter)
        mock_preserver_cls.assert_called_once()
        assert clipboard_preserving._preserver is mock_preserver_cls.return_value

    def test_should_return_paste_and_clipboard_adapter_for_paste_and_clipboard_mode(
        self,
    ):
        dispatcher = MagicMock()

        result = _build_output(OutputMode.PASTE_AND_CLIPBOARD, dispatcher)

        assert isinstance(result, GuiThreadOutputDecorator)
        assert isinstance(result._inner, PasteAndClipboardAdapter)

    def test_should_return_clipboard_adapter_for_clipboard_only_mode(self):
        dispatcher = MagicMock()

        result = _build_output(OutputMode.CLIPBOARD_ONLY, dispatcher)

        assert isinstance(result, GuiThreadOutputDecorator)
        assert isinstance(result._inner, ClipboardAdapter)

    def test_should_wrap_inner_with_gui_thread_dispatcher(self):
        dispatcher = MagicMock()

        result = _build_output(OutputMode.CLIPBOARD_ONLY, dispatcher)

        assert result._dispatcher is dispatcher


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
        dispatcher = MagicMock()
        changes = self._make_changes(output_changed=True)

        apply_settings(changes, config, use_case, app, gui_dispatcher=dispatcher)

        mock_build_output.assert_called_once_with(
            OutputMode.CLIPBOARD_ONLY, dispatcher
        )
        use_case.set_output.assert_called_once_with(mock_build_output.return_value)

    @patch("voicetotext.bootstrap._build_output")
    def test_should_not_rebuild_output_when_output_unchanged(self, mock_build_output):
        config = MagicMock()
        use_case = MagicMock()
        app = MagicMock()
        dispatcher = MagicMock()
        changes = self._make_changes(output_changed=False)

        apply_settings(changes, config, use_case, app, gui_dispatcher=dispatcher)

        mock_build_output.assert_not_called()
        use_case.set_output.assert_not_called()

    @patch("voicetotext.bootstrap._build_hotkey_listener")
    def test_should_rebuild_hotkey_listener_when_hotkey_changed(
        self, mock_build_listener
    ):
        config = MagicMock()
        use_case = MagicMock()
        app = MagicMock()
        dispatcher = MagicMock()
        changes = self._make_changes(hotkey_changed=True)

        apply_settings(changes, config, use_case, app, gui_dispatcher=dispatcher)

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
        dispatcher = MagicMock()
        changes = self._make_changes(hotkey_changed=False)

        apply_settings(changes, config, use_case, app, gui_dispatcher=dispatcher)

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
        dispatcher = MagicMock()
        changes = self._make_changes(hotkey_changed=True, output_changed=True)

        apply_settings(changes, config, use_case, app, gui_dispatcher=dispatcher)

        use_case.set_output.assert_called_once_with(mock_build_output.return_value)
        app.set_hotkey_listener.assert_called_once_with(
            mock_build_listener.return_value
        )


class TestMakeDestroyedHandler:
    """Characterize the guard on the ``destroyed`` signal handler used by
    ``open_settings`` — nulling ``current`` must only happen when the
    destroyed window is the one *still* registered. Otherwise a just-
    destroyed previous window would orphan the newly-opened one.
    """

    def test_should_clear_current_when_registered_window_is_destroyed(self):
        window = object()
        ref: dict[str, object | None] = {"current": window}

        handler = _make_destroyed_handler(ref, window)
        handler()

        assert ref["current"] is None

    def test_should_not_clear_current_when_another_window_is_registered(self):
        old_window = object()
        new_window = object()
        ref: dict[str, object | None] = {"current": new_window}

        handler = _make_destroyed_handler(ref, old_window)
        handler()

        assert ref["current"] is new_window

    def test_should_not_raise_when_current_is_already_none(self):
        window = object()
        ref: dict[str, object | None] = {"current": None}

        handler = _make_destroyed_handler(ref, window)
        handler()

        assert ref["current"] is None

    def test_should_accept_qt_signal_arguments(self):
        window = object()
        ref: dict[str, object | None] = {"current": window}

        handler = _make_destroyed_handler(ref, window)
        handler(window)  # Qt's destroyed emits the QObject

        assert ref["current"] is None
