from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from voicetotext.bootstrap import (
    _build_output,
    _build_streaming_engine_for,
    _load_engine_and_attach,
    _load_engine_with_fallback,
    _make_active_engine_provider,
    _warmup_safely,
    apply_settings,
    migrate_language,
    migrate_output_mode,
)
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import EngineLoadFailedEvent, StateChangedEvent
from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    EngineWarmupError,
)
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
from voicetotext.output.text_filter_decorator import TextFilterDecorator
from voicetotext.output.text_pipeline import PipelineMode
from voicetotext.settings.settings_draft import ChangeSet
from voicetotext.transcription.engine_factory import build_engine_registry
from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.model_info import migrate_removed_models


_HARDWARE = MagicMock()
_REGISTRY = MagicMock()

_ROCM_PROFILE = HardwareProfile(
    kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200",
)


def _provider_config(model_size: str = "base", engine: str = "auto"):
    config = MagicMock()
    values = {
        "transcription.model_size": model_size,
        "transcription.engine": engine,
        "transcription.device": "auto",
    }
    config.get.side_effect = lambda key, default=None: values.get(key, default)
    return config


class TestActiveEngineProvider:
    """The provider the composite ModelManager polls must resolve the active
    engine for the configured model the same way the loader does, so the UI
    install state agrees with the loader. Every model now has a single
    compatible engine, so resolution is driven by the configured model.
    """

    def test_should_resolve_whisper_model_to_transformers(self, tmp_path):
        provider = _make_active_engine_provider(
            _provider_config(model_size="base"), _ROCM_PROFILE,
            build_engine_registry(), tmp_path,
        )

        assert provider() == "transformers"

    def test_should_resolve_parakeet_model_to_nano_parakeet(self, tmp_path):
        provider = _make_active_engine_provider(
            _provider_config(model_size="parakeet-tdt-0.6b-v3"), _ROCM_PROFILE,
            build_engine_registry(), tmp_path,
        )

        assert provider() == "nano_parakeet"


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
        # press - without it, the first transcription pays a one-off
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
        # Warmup is a latency optimisation - if it fails (e.g. a backend
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
        # to the SAME engine_id from config - only device_override differs.
        first_call, second_call = mock_create.call_args_list
        assert first_call.args == (config, paths, _REGISTRY, _HARDWARE)
        assert second_call.args == (config, paths, _REGISTRY, _HARDWARE)
        assert second_call.kwargs == {"device_override": "cpu"}


class TestLoadEngineAndAttach:
    """The background-load worker swaps the placeholder for the real
    engine and transitions ``LOADING → IDLE`` on success. On failure it
    publishes the failure and recovers to ``IDLE`` (via ``ERROR``) so the
    app stays swappable rather than locking up. Tests run the worker on
    the calling thread - bootstrap spawns the thread separately."""

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_attach_loaded_engine_to_use_case(self, mock_create):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        loaded_engine = MagicMock()
        mock_create.return_value = loaded_engine
        use_case = MagicMock()

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            use_case=use_case,
            state_machine=state_machine,
            event_bus=bus,
        )

        use_case.set_engine.assert_called_once_with(loaded_engine)

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_transition_to_idle_on_successful_load(self, mock_create):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        mock_create.return_value = MagicMock()

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            use_case=MagicMock(),
            state_machine=state_machine,
            event_bus=bus,
        )

        assert state_machine.state == AppState.IDLE

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_recover_to_idle_after_load_failure(self, mock_create):
        # A failed initial load must not strand the app in ERROR: the swap
        # guard only permits IDLE/LOADING, so ERROR would reject every later
        # swap. The worker passes through ERROR (for the tray) then settles
        # on IDLE so the user can pick another model without restarting.
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        visited: list[str] = []
        bus.subscribe(StateChangedEvent, lambda e: visited.append(e.new_state))
        broken = MagicMock()
        broken.load_model.side_effect = RuntimeError("VRAM exhausted")
        mock_create.side_effect = [broken, broken]
        use_case = MagicMock()

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            use_case=use_case,
            state_machine=state_machine,
            event_bus=bus,
        )

        assert state_machine.state == AppState.IDLE
        assert visited == [AppState.ERROR.value, AppState.IDLE.value]
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
            use_case=MagicMock(),
            state_machine=state_machine,
            event_bus=bus,
        )


class TestLoadEngineAndAttachFailureSurface:
    """When the background engine load fails, the worker must publish
    :class:`EngineLoadFailedEvent` carrying the active model name and
    the underlying error message so the tray can show an actionable
    notification balloon instead of just flipping the icon to ERROR.
    """

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_publish_engine_load_failed_event_when_dependency_missing(
        self, mock_create,
    ):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        config = MagicMock()
        config.get.return_value = "large-v3-turbo"
        broken = MagicMock()
        broken.load_model.side_effect = EngineDependencyMissingError(
            "nano-parakeet not installed: pip install nano-parakeet"
        )
        # ``_load_engine_with_fallback`` only catches RuntimeError/OSError
        # for CPU fallback; EngineDependencyMissingError propagates
        # straight through and is observed once by mock_create.
        mock_create.side_effect = [broken, broken]
        received: list[EngineLoadFailedEvent] = []
        bus.subscribe(EngineLoadFailedEvent, received.append)

        _load_engine_and_attach(
            config, MagicMock(), _REGISTRY, _HARDWARE,
            use_case=MagicMock(),
            state_machine=state_machine,
            event_bus=bus,
        )

        assert len(received) == 1
        assert "nano-parakeet not installed" in received[0].error
        assert received[0].model_name == "large-v3-turbo"

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_publish_event_before_entering_error_then_recover(
        self, mock_create,
    ):
        # The failure event must reach the tray before the icon flips, and
        # the machine must end IDLE having passed through ERROR.
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        config = MagicMock()
        config.get.return_value = "base"
        broken = MagicMock()
        broken.load_model.side_effect = RuntimeError("VRAM exhausted")
        mock_create.side_effect = [broken, broken]
        timeline: list[str] = []
        bus.subscribe(
            EngineLoadFailedEvent, lambda e: timeline.append("failed"),
        )
        bus.subscribe(
            StateChangedEvent, lambda e: timeline.append(e.new_state),
        )

        _load_engine_and_attach(
            config, MagicMock(), _REGISTRY, _HARDWARE,
            use_case=MagicMock(),
            state_machine=state_machine,
            event_bus=bus,
        )

        assert timeline == [
            "failed", AppState.ERROR.value, AppState.IDLE.value,
        ]
        assert state_machine.state == AppState.IDLE

    @patch("voicetotext.bootstrap.create_engine_for_active_model")
    def test_should_not_publish_event_on_successful_load(self, mock_create):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        mock_create.return_value = MagicMock()
        received: list[EngineLoadFailedEvent] = []
        bus.subscribe(EngineLoadFailedEvent, received.append)

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            use_case=MagicMock(),
            state_machine=state_machine,
            event_bus=bus,
        )

        assert received == []


class TestWarmupSafelyErrorClassification:
    """``_warmup_safely`` distinguishes between transient warmup quirks
    (e.g. ``RuntimeError`` from a backend's first-pass kernel compile on
    silent audio) and unrecoverable errors that imply the engine itself
    will never transcribe (missing backend, missing model file, dead
    device). The transient case is swallowed so the first real
    transcription pays the cold-start cost; the unrecoverable case is
    re-raised as :class:`EngineWarmupError` so the bootstrap surface
    can publish ``EngineLoadFailedEvent`` instead of leaving a half-
    broken engine attached.
    """

    def test_should_swallow_runtime_error_as_transient(self):
        engine = MagicMock()
        engine.warmup.side_effect = RuntimeError("backend kernel glitch")

        _warmup_safely(engine)

    def test_should_reraise_engine_dependency_missing_during_warmup(self):
        engine = MagicMock()
        engine.warmup.side_effect = EngineDependencyMissingError(
            "nano-parakeet not installed: pip install nano-parakeet"
        )

        with pytest.raises(EngineWarmupError, match="nano-parakeet"):
            _warmup_safely(engine)

    def test_should_reraise_import_error_during_warmup(self):
        engine = MagicMock()
        engine.warmup.side_effect = ImportError("CUDA runtime missing")

        with pytest.raises(EngineWarmupError, match="CUDA runtime missing"):
            _warmup_safely(engine)

    def test_should_reraise_file_not_found_during_warmup(self):
        engine = MagicMock()
        engine.warmup.side_effect = FileNotFoundError("model.bin")

        with pytest.raises(EngineWarmupError, match="model.bin"):
            _warmup_safely(engine)

    def test_should_reraise_os_error_during_warmup(self):
        engine = MagicMock()
        engine.warmup.side_effect = OSError("device went away")

        with pytest.raises(EngineWarmupError, match="device went away"):
            _warmup_safely(engine)

    def test_should_chain_original_exception_via_from(self):
        original = ImportError("missing cudnn")
        engine = MagicMock()
        engine.warmup.side_effect = original

        with pytest.raises(EngineWarmupError) as exc_info:
            _warmup_safely(engine)

        assert exc_info.value.__cause__ is original


class TestMigrateRemovedModels:
    def test_should_rewrite_medium_to_large_v3_turbo(self):
        config = MagicMock()
        config.get.return_value = "medium"

        migrate_removed_models(config)

        config.set.assert_called_once_with(
            "transcription.model_size", "large-v3-turbo"
        )
        config.save.assert_called_once()

    def test_should_rewrite_removed_compare_model_to_plain_streaming(self):
        config = MagicMock()
        config.get.return_value = "nemotron-streaming-compare"

        migrate_removed_models(config)

        config.set.assert_called_once_with(
            "transcription.model_size", "nemotron-speech-streaming-en-0.6b"
        )
        config.save.assert_called_once()

    def test_should_rewrite_granite_to_large_v3_turbo(self):
        config = MagicMock()
        config.get.return_value = "granite-speech-4.1-2b"

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

        assert isinstance(result, TextFilterDecorator)
        gui_output = result._inner
        assert isinstance(gui_output, GuiThreadOutputDecorator)
        clipboard_preserving = gui_output._inner
        assert isinstance(clipboard_preserving, ClipboardPreservingDecorator)
        assert isinstance(clipboard_preserving._inner, PasteAndClipboardAdapter)
        mock_preserver_cls.assert_called_once()
        assert clipboard_preserving._preserver is mock_preserver_cls.return_value

    def test_should_return_paste_and_clipboard_adapter_for_paste_and_clipboard_mode(
        self,
    ):
        dispatcher = MagicMock()

        result = _build_output(OutputMode.PASTE_AND_CLIPBOARD, dispatcher)

        assert isinstance(result, TextFilterDecorator)
        assert isinstance(result._inner, GuiThreadOutputDecorator)
        assert isinstance(result._inner._inner, PasteAndClipboardAdapter)

    def test_should_return_clipboard_adapter_for_clipboard_only_mode(self):
        dispatcher = MagicMock()

        result = _build_output(OutputMode.CLIPBOARD_ONLY, dispatcher)

        assert isinstance(result, TextFilterDecorator)
        assert isinstance(result._inner, GuiThreadOutputDecorator)
        assert isinstance(result._inner._inner, ClipboardAdapter)

    def test_should_wrap_inner_with_gui_thread_dispatcher(self):
        dispatcher = MagicMock()

        result = _build_output(OutputMode.CLIPBOARD_ONLY, dispatcher)

        assert result._inner._dispatcher is dispatcher

    def test_should_build_reduced_pipeline_for_streaming_mode(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text(
            '{"version": 2, "options": {"spoken_commands": true}, '
            '"rules": [{"type": "literal", "spoken": ["git hub"], '
            '"replacement": "GitHub"}]}',
            encoding="utf-8",
        )

        result = _build_output(
            OutputMode.CLIPBOARD_ONLY,
            MagicMock(),
            path,
            PipelineMode.STREAMING,
        )

        assert result._pipeline.process("git hub comma next") == (
            "git hub comma next"
        )


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
            OutputMode.CLIPBOARD_ONLY, dispatcher, None, PipelineMode.BATCH
        )
        use_case.set_output.assert_called_once_with(mock_build_output.return_value)

    @patch("voicetotext.bootstrap._build_output")
    def test_should_rebuild_streaming_safe_output_for_streaming_mode(
        self, mock_build_output: MagicMock,
    ) -> None:
        config = MagicMock()
        values = {
            "output.mode": "clipboard_only",
            "transcription.mode": "streaming",
        }
        config.get.side_effect = lambda key, default=None: values.get(key, default)
        use_case = MagicMock()
        changes = self._make_changes(output_changed=True)

        apply_settings(
            changes, config, use_case, MagicMock(), gui_dispatcher=MagicMock(),
        )

        assert mock_build_output.call_args.args[3] is PipelineMode.STREAMING

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


class TestBuildStreamingEngineRouting:
    def test_should_build_cache_aware_engine_for_nemotron_models(self, tmp_path):
        from voicetotext.transcription.streaming.cache_aware.engine import (
            CacheAwareStreamingEngine,
        )
        model_info = SimpleNamespace(
            compatible_engines=("nemotron_streaming",),
            repo_id="nvidia/nemotron-speech-streaming-en-0.6b",
        )
        paths = SimpleNamespace(models_dir=tmp_path)

        engine = _build_streaming_engine_for(model_info, paths)

        assert isinstance(engine, CacheAwareStreamingEngine)
