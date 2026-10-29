import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

import skryb.bootstrap as bootstrap
from tests.fakes.config import FakeConfig
from tests.fakes.hardware import FakeHardwareDetector
from tests.fakes.licensing import AllowAllLicensePolicy
from skryb.bootstrap import (
    _build_output,
    _build_streaming_engine_for,
    _load_engine_and_attach,
    _load_engine_with_fallback,
    _load_streaming_engine,
    _make_active_engine_provider,
    _build_edit_capture,
    _delivery_reporter,
    _ignore_delivery,
    _key_observer,
    _resolve_pipeline_mode,
    _warmup_safely,
    apply_settings,
    migrate_hotkey_bindings,
    migrate_language,
    migrate_output_mode,
)
from skryb.core.event_bus import SimpleEventBus
from skryb.core.events import EngineLoadFailedEvent, StateChangedEvent
from skryb.core.exceptions import (
    EngineDependencyMissingError,
    EngineWarmupError,
    LicenseNotAcceptedError,
)
from skryb.core.json_config_adapter import JsonConfigAdapter
from skryb.core.paths import AppPaths
from skryb.core.state_machine import AppState, AppStateMachine
from skryb.diagnostics.session import SessionDiagnostics
from skryb.hotkeys.actions import HotkeyActions
from skryb.edit_capture.delivery_recording_decorator import (
    DeliveryRecordingDecorator,
)
from skryb.output.caps_lock_uppercase_decorator import (
    CapsLockUppercaseDecorator,
)
from skryb.output.caret_context import NO_CONTEXT
from skryb.output.clipboard_adapter import ClipboardAdapter
from skryb.output.clipboard_preserving_decorator import (
    ClipboardPreservingDecorator,
)
from skryb.output.gui_thread_output_decorator import (
    GuiThreadOutputDecorator,
)
from skryb.output.output_mode import OutputMode
from skryb.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from skryb.output.paste_memory import PasteMemory
from skryb.output.text_filter_decorator import TextFilterDecorator
from skryb.output.text_pipeline import PipelineMode
from skryb.output.tk_gui_thread_dispatcher import TkGuiThreadDispatcher
from skryb.settings.settings_draft import ChangeSet
from skryb.transcription.engine_factory import build_engine_registry
from skryb.transcription.engine_types import HardwareProfile
from skryb.transcription.license_policy import LicensePolicy

_HARDWARE = MagicMock()
_REGISTRY = MagicMock()
_ALLOW_ALL_LICENCES = AllowAllLicensePolicy()
_DIAGNOSTICS = MagicMock(spec=SessionDiagnostics)

_ROCM_DETECTOR = FakeHardwareDetector(
    HardwareProfile(
        kind="amd_rocm", device_name="AMD Radeon RX 9060 XT",
        gfx_arch="gfx1200",
    )
)


@pytest.fixture(autouse=True)
def fresh_paste_memory(monkeypatch):
    """Keep one test's paste out of the next, and off the real desktop."""
    monkeypatch.setattr(
        bootstrap, "_paste_memory", PasteMemory(lambda: 0, lambda: 0, 0),
    )



def _provider_config(
    model_size: str = "parakeet-ultra", engine: str = "auto",
) -> MagicMock:
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

    def test_should_resolve_parakeet_model_to_nano_parakeet(self, tmp_path):
        provider = _make_active_engine_provider(
            _provider_config(model_size="parakeet-ultra"), _ROCM_DETECTOR,
            build_engine_registry(), tmp_path,
        )

        assert provider() == "nano_parakeet"


class TestLoadEngineWithFallback:
    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_return_engine_when_first_attempt_succeeds(self, mock_create):
        engine = MagicMock()
        mock_create.return_value = engine
        diagnostics = MagicMock(spec=SessionDiagnostics)

        result = _load_engine_with_fallback(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            _ALLOW_ALL_LICENCES, diagnostics,
        )

        assert result is engine
        engine.load_model.assert_called_once()
        diagnostics.engine_loaded.assert_called_once()

    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_retry_on_cpu_when_default_load_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("CUDA fail")
        cpu_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback(
            config, paths, _REGISTRY, _HARDWARE, _ALLOW_ALL_LICENCES,
            _DIAGNOSTICS,
        )

        assert result is cpu_engine
        first_call, second_call = mock_create.call_args_list
        assert first_call.kwargs.get("device_override") is None
        assert second_call.kwargs["device_override"] == "cpu"

    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_catch_os_error_from_missing_model_files(self, mock_create):
        broken_engine = MagicMock()
        broken_engine.load_model.side_effect = OSError("Can't load feature extractor")
        cpu_engine = MagicMock()
        mock_create.side_effect = [broken_engine, cpu_engine]

        result = _load_engine_with_fallback(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
        )

        assert result is cpu_engine

    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_propagate_error_when_cpu_fallback_also_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("fail")
        cpu_engine = MagicMock()
        cpu_engine.load_model.side_effect = RuntimeError("fail")
        mock_create.side_effect = [gpu_engine, cpu_engine]
        diagnostics = MagicMock(spec=SessionDiagnostics)

        with pytest.raises(RuntimeError):
            _load_engine_with_fallback(
                MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
                _ALLOW_ALL_LICENCES, diagnostics,
            )
        diagnostics.engine_load_failed.assert_called_once()

    @patch("skryb.bootstrap.create_engine_for_active_model")
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
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
        )

        engine.load_model.assert_called_once()
        engine.warmup.assert_called_once()
        assert (
            engine.method_calls.index(call.warmup())
            > engine.method_calls.index(call.load_model())
        ), "warmup must follow load_model, never precede it"

    @patch("skryb.bootstrap.create_engine_for_active_model")
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
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
        )

        assert result is engine
        engine.load_model.assert_called_once()
        engine.warmup.assert_called_once()

    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_warm_cpu_fallback_engine_too(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("GPU OOM")
        cpu_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine]

        result = _load_engine_with_fallback(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
        )

        assert result is cpu_engine
        cpu_engine.load_model.assert_called_once()
        cpu_engine.warmup.assert_called_once()

    @patch("skryb.bootstrap.create_engine_for_active_model")
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

        _load_engine_with_fallback(
            config, paths, _REGISTRY, _HARDWARE, _ALLOW_ALL_LICENCES,
            _DIAGNOSTICS,
        )

        # Both calls go through the SAME factory function, which routes
        # to the SAME engine_id from config - only device_override differs.
        first_call, second_call = mock_create.call_args_list
        assert first_call.args == (config, paths, _REGISTRY, _HARDWARE)
        assert second_call.args == (config, paths, _REGISTRY, _HARDWARE)
        assert second_call.kwargs["device_override"] == "cpu"
        assert first_call.kwargs["license_policy"] is (
            second_call.kwargs["license_policy"]
        )


class TestLoadEngineAndAttach:
    """The background-load worker swaps the placeholder for the real
    engine and transitions ``LOADING → IDLE`` on success. On failure it
    publishes the failure and recovers to ``IDLE`` (via ``ERROR``) so the
    app stays swappable rather than locking up. Tests run the worker on
    the calling thread - bootstrap spawns the thread separately."""

    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_attach_loaded_engine_to_use_case(self, mock_create):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        loaded_engine = MagicMock()
        mock_create.return_value = loaded_engine
        use_case = MagicMock()

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
            use_case=use_case,
            state_machine=state_machine,
            event_bus=bus,
        )

        use_case.set_engine.assert_called_once_with(loaded_engine)

    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_transition_to_idle_on_successful_load(self, mock_create):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        mock_create.return_value = MagicMock()

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
            use_case=MagicMock(),
            state_machine=state_machine,
            event_bus=bus,
        )

        assert state_machine.state == AppState.IDLE

    @patch("skryb.bootstrap.create_engine_for_active_model")
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
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
            use_case=use_case,
            state_machine=state_machine,
            event_bus=bus,
        )

        assert state_machine.state == AppState.IDLE
        assert visited == [AppState.ERROR.value, AppState.IDLE.value]
        use_case.set_engine.assert_not_called()

    @patch("skryb.bootstrap.create_engine_for_active_model")
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
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
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

    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_publish_engine_load_failed_event_when_dependency_missing(
        self, mock_create,
    ):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        config = MagicMock()
        config.get.return_value = "parakeet-ultra"
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
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
            use_case=MagicMock(),
            state_machine=state_machine,
            event_bus=bus,
        )

        assert len(received) == 1
        assert "nano-parakeet not installed" in received[0].error
        assert received[0].model_name == "parakeet-ultra"

    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_publish_event_before_entering_error_then_recover(
        self, mock_create,
    ):
        # The failure event must reach the tray before the icon flips, and
        # the machine must end IDLE having passed through ERROR.
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        config = MagicMock()
        config.get.return_value = "parakeet-ultra"
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
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
            use_case=MagicMock(),
            state_machine=state_machine,
            event_bus=bus,
        )

        assert timeline == [
            "failed", AppState.ERROR.value, AppState.IDLE.value,
        ]
        assert state_machine.state == AppState.IDLE

    @patch("skryb.bootstrap.create_engine_for_active_model")
    def test_should_not_publish_event_on_successful_load(self, mock_create):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        mock_create.return_value = MagicMock()
        received: list[EngineLoadFailedEvent] = []
        bus.subscribe(EngineLoadFailedEvent, received.append)

        _load_engine_and_attach(
            MagicMock(), MagicMock(), _REGISTRY, _HARDWARE,
            _ALLOW_ALL_LICENCES, _DIAGNOSTICS,
            use_case=MagicMock(),
            state_machine=state_machine,
            event_bus=bus,
        )

        assert received == []


class TestLicenceGateAtStartupLoad:
    """Startup loads weights, so it is a gated path. The batch loader
    inherits the check from ``create_engine_for_active_model``; the
    streaming loader bypasses that factory entirely and asserts for
    itself, so both are exercised here.
    """

    _GATED = "crisperwhisper-2.0-turbo"

    def _gated_config(self) -> FakeConfig:
        return FakeConfig({"transcription.model_size": self._GATED})

    def test_streaming_load_should_refuse_before_building_an_engine(
        self, tmp_path: Path,
    ) -> None:
        paths = SimpleNamespace(models_dir=tmp_path)

        with pytest.raises(LicenseNotAcceptedError, match=self._GATED):
            _load_streaming_engine(
                self._gated_config(), paths, _HARDWARE,
                LicensePolicy(FakeConfig()), _DIAGNOSTICS,
            )

    def test_streaming_load_should_leave_an_ungated_model_alone(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        engine = MagicMock()
        monkeypatch.setattr(
            bootstrap, "_build_streaming_engine_for",
            lambda model_info, paths: engine,
        )
        config = FakeConfig(
            {"transcription.model_size": "nemotron-speech-streaming-en-0.6b"},
        )
        diagnostics = MagicMock(spec=SessionDiagnostics)

        loaded = _load_streaming_engine(
            config, SimpleNamespace(models_dir=tmp_path), _HARDWARE,
            LicensePolicy(FakeConfig()), diagnostics,
        )

        assert loaded is engine
        diagnostics.engine_loaded.assert_called_once()

    def test_batch_load_should_publish_the_licence_failure_and_recover(
        self, tmp_path: Path,
    ) -> None:
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus, initial_state=AppState.LOADING)
        received: list[EngineLoadFailedEvent] = []
        bus.subscribe(EngineLoadFailedEvent, received.append)
        use_case = MagicMock()

        _load_engine_and_attach(
            self._gated_config(),
            SimpleNamespace(models_dir=tmp_path),
            build_engine_registry(),
            _ROCM_DETECTOR,
            LicensePolicy(FakeConfig()),
            _DIAGNOSTICS,
            use_case=use_case,
            state_machine=state_machine,
            event_bus=bus,
        )

        assert len(received) == 1
        assert "crisperwhisper-nc" in received[0].error
        assert state_machine.state == AppState.IDLE
        use_case.set_engine.assert_not_called()


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


class TestMigrateHotkeyBindings:
    @pytest.mark.parametrize(("mode", "migrated_mode"), [
        ("push_to_talk", "hold"), ("toggle", "toggle"), ("nonsense", "toggle"),
    ])
    def test_an_old_single_hotkey_becomes_one_dictation_shortcut_and_the_old_keys_go(
        self, tmp_path: Path, mode: str, migrated_mode: str,
    ):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({
            "hotkey": {"combination": "ctrl+shift+alt+m", "mode": mode},
            "output": {"mode": "direct_paste"},
        }), encoding="utf-8")
        config = JsonConfigAdapter(path)
        config.load()

        migrate_hotkey_bindings(config)

        saved = json.loads(path.read_text(encoding="utf-8"))["hotkey"]
        assert saved["bindings"] == [
            {"keys": "ctrl+shift+alt+m", "mode": migrated_mode, "action": "dictate"},
        ]
        assert "combination" not in saved and "mode" not in saved
        assert saved["double_tap_locks_hold"] is True

    def test_an_old_mode_without_a_combination_keeps_the_old_default_keys(
        self, tmp_path: Path,
    ):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"hotkey": {"mode": "push_to_talk"}}), encoding="utf-8")
        config = JsonConfigAdapter(path)
        config.load()

        migrate_hotkey_bindings(config)

        assert config.get("hotkey.bindings") == [
            {"keys": "ctrl+alt+m", "mode": "hold", "action": "dictate"},
        ]

    def test_saved_bindings_win_over_leftover_old_keys(self, tmp_path: Path):
        bindings = [
            {"keys": "mouse4", "mode": "hold", "action": "dictate"},
            {"keys": "f13", "mode": "press", "action": "paste_last"},
        ]
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"hotkey": {
            "bindings": bindings, "combination": "f12", "mode": "toggle",
        }}), encoding="utf-8")
        config = JsonConfigAdapter(path)
        config.load()

        migrate_hotkey_bindings(config)

        assert config.get("hotkey.bindings") == bindings
        assert config.get("hotkey.combination") is None

    def test_a_config_already_on_bindings_is_not_rewritten(self, tmp_path: Path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()

        migrate_hotkey_bindings(config)

        assert not path.exists()


class TestBuildOutput:
    @patch("skryb.bootstrap.Win32ClipboardPreserver")
    def test_should_return_decorator_wrapping_paste_and_clipboard_with_preserver(
        self, mock_preserver_cls
    ):
        dispatcher = MagicMock()

        result = _build_output(OutputMode.DIRECT_PASTE, dispatcher)

        assert isinstance(result, TextFilterDecorator)
        gui_output = result._inner
        assert isinstance(gui_output, GuiThreadOutputDecorator)
        caps_lock = gui_output._inner
        assert isinstance(caps_lock, CapsLockUppercaseDecorator)
        clipboard_preserving = caps_lock._inner
        assert isinstance(clipboard_preserving, ClipboardPreservingDecorator)
        recorded = clipboard_preserving._inner
        assert isinstance(recorded, DeliveryRecordingDecorator)
        assert isinstance(recorded._inner, PasteAndClipboardAdapter)
        mock_preserver_cls.assert_called_once()
        assert clipboard_preserving._preserver is mock_preserver_cls.return_value

    def test_should_return_paste_and_clipboard_adapter_for_paste_and_clipboard_mode(
        self,
    ):
        dispatcher = MagicMock()

        result = _build_output(OutputMode.PASTE_AND_CLIPBOARD, dispatcher)

        assert isinstance(result, TextFilterDecorator)
        assert isinstance(result._inner, GuiThreadOutputDecorator)
        assert isinstance(result._inner._inner, CapsLockUppercaseDecorator)
        recorded = result._inner._inner._inner
        assert isinstance(recorded, DeliveryRecordingDecorator)
        assert isinstance(recorded._inner, PasteAndClipboardAdapter)

    def test_should_return_clipboard_adapter_for_clipboard_only_mode(self):
        dispatcher = MagicMock()

        result = _build_output(OutputMode.CLIPBOARD_ONLY, dispatcher)

        assert isinstance(result, TextFilterDecorator)
        assert isinstance(result._inner, GuiThreadOutputDecorator)
        assert isinstance(result._inner._inner, CapsLockUppercaseDecorator)
        assert isinstance(result._inner._inner._inner, ClipboardAdapter)

    @patch("skryb.bootstrap.caps_lock_enabled", return_value=False)
    @patch("skryb.bootstrap.read_caret_context", return_value=NO_CONTEXT)
    @patch("skryb.bootstrap.Win32KeySender")
    @patch("skryb.bootstrap.Win32ClipboardWriter")
    def test_should_report_a_delivery_when_the_text_is_pasted(
        self, _writer_cls, _sender_cls, _read_context, _caps_lock,
    ):
        reported: list[str] = []
        dispatcher = MagicMock()
        dispatcher.dispatch.side_effect = lambda action: action()

        result = _build_output(
            OutputMode.PASTE_AND_CLIPBOARD, dispatcher,
            on_delivered=reported.append,
        )
        result.output("hello there")

        assert reported == ["hello there"]

    @patch("skryb.bootstrap.Win32ClipboardWriter")
    def test_should_report_no_delivery_when_nothing_is_pasted(self, _writer_cls):
        # Edit capture pairs a report with the Ctrl+V that follows it. A mode
        # that never injects one must stay out of that queue entirely.
        reported: list[str] = []
        dispatcher = MagicMock()
        dispatcher.dispatch.side_effect = lambda action: action()

        result = _build_output(
            OutputMode.CLIPBOARD_ONLY, dispatcher, on_delivered=reported.append,
        )
        result.output("hello there")

        assert reported == []

    @patch("skryb.bootstrap.read_caret_context", return_value=NO_CONTEXT)
    @patch("skryb.bootstrap.caps_lock_enabled", return_value=False)
    @patch("skryb.bootstrap.Win32ModifierKeyStateReader")
    @patch("skryb.bootstrap.Win32ClipboardPreserver")
    @patch("skryb.bootstrap.Win32KeySender")
    @patch("skryb.bootstrap.Win32ClipboardWriter")
    def test_should_paste_into_our_own_window_before_restoring_the_clipboard(
        self, writer_cls, sender_cls, preserver_cls, modifier_reader_cls,
        _caps_lock, _caret_context, tk_root,
    ):
        # The settings window reads the injected Ctrl+V on the same Tk thread
        # that runs the paste, so it only sees the clipboard once that thread
        # gets back to its event loop.
        clipboard = {"text": "previous clipboard"}
        pasted: list[str] = []
        writer_cls.return_value.set_text.side_effect = (
            lambda text: clipboard.update(text=text)
        )
        preserver_cls.return_value.restore.side_effect = (
            lambda: clipboard.update(text="previous clipboard")
        )
        sender_cls.return_value.send_paste.side_effect = (
            lambda _method, _text: tk_root.after(0, lambda: pasted.append(clipboard["text"]))
        )
        modifier_reader_cls.return_value.any_modifier_down.return_value = False
        dispatcher = TkGuiThreadDispatcher(tk_root)
        output = _build_output(OutputMode.DIRECT_PASTE, dispatcher)

        output.output("hello there")
        deadline = time.monotonic() + 3.0
        while not (pasted and clipboard["text"] == "previous clipboard"):
            if time.monotonic() > deadline:
                break
            tk_root.update()
            time.sleep(0.005)
        dispatcher.cancel()

        assert pasted == ["hello there"]

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

    @patch("skryb.bootstrap._build_output")
    def test_should_rebuild_output_when_output_changed(self, mock_build_output):
        config = MagicMock()
        config.get.return_value = "clipboard_only"
        use_case = MagicMock()
        app = MagicMock()
        dispatcher = MagicMock()
        changes = self._make_changes(output_changed=True)

        apply_settings(changes, config, use_case, app, gui_dispatcher=dispatcher)

        mock_build_output.assert_called_once_with(
            OutputMode.CLIPBOARD_ONLY, dispatcher, None, PipelineMode.BATCH, None,
            _ignore_delivery, hooks=None,
        )
        use_case.set_output.assert_called_once_with(mock_build_output.return_value)

    @patch("skryb.bootstrap._build_output")
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

    @patch("skryb.bootstrap._build_output")
    def test_should_not_rebuild_output_when_output_unchanged(self, mock_build_output):
        config = MagicMock()
        use_case = MagicMock()
        app = MagicMock()
        dispatcher = MagicMock()
        changes = self._make_changes(output_changed=False)

        apply_settings(changes, config, use_case, app, gui_dispatcher=dispatcher)

        mock_build_output.assert_not_called()
        use_case.set_output.assert_not_called()

    @patch("skryb.bootstrap._build_hotkey_listener")
    def test_should_rebuild_hotkey_listener_when_hotkey_changed(
        self, mock_build_listener
    ):
        config = MagicMock()
        use_case = MagicMock()
        app = MagicMock()
        dispatcher = MagicMock()
        changes = self._make_changes(hotkey_changed=True)

        apply_settings(changes, config, use_case, app, gui_dispatcher=dispatcher)

        mock_build_listener.assert_called_once_with(
            config, use_case, bootstrap._hotkey_actions(use_case),
            bootstrap._paste_memory.observe_key,
        )
        app.set_hotkey_listener.assert_called_once_with(
            mock_build_listener.return_value
        )

    @patch("skryb.bootstrap._build_hotkey_listener")
    def test_a_rebuilt_listener_keeps_every_shortcut_action_it_was_given(
        self, mock_build_listener
    ):
        use_case = MagicMock()
        actions = HotkeyActions(
            dictate_activate=use_case.on_hotkey_activated,
            dictate_deactivate=use_case.on_hotkey_deactivated,
            paste_last=MagicMock(),
            copy_last=MagicMock(),
        )

        apply_settings(
            self._make_changes(hotkey_changed=True), MagicMock(), use_case,
            MagicMock(), gui_dispatcher=MagicMock(), hotkey_actions=actions,
        )

        assert mock_build_listener.call_args.args[2] is actions

    @patch("skryb.bootstrap._build_hotkey_listener")
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

    @patch("skryb.bootstrap._build_hotkey_listener")
    @patch("skryb.bootstrap._build_output")
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
        from skryb.transcription.streaming.cache_aware.engine import (
            CacheAwareStreamingEngine,
        )
        model_info = SimpleNamespace(
            compatible_engines=("nemotron_streaming",),
            repo_id="nvidia/nemotron-speech-streaming-en-0.6b",
        )
        paths = SimpleNamespace(models_dir=tmp_path)

        engine = _build_streaming_engine_for(model_info, paths)

        assert isinstance(engine, CacheAwareStreamingEngine)


class TestPipelineModeForVerbatimTranscription:
    @staticmethod
    def _config(model: str, mode: str, transcription_mode: str = "batch"):
        config = JsonConfigAdapter(Path("unused.json"))
        config.load()
        config.set("transcription.model_size", model)
        config.set("transcription.crisperwhisper_mode", mode)
        config.set("transcription.mode", transcription_mode)
        return config

    def test_should_bypass_the_pipeline_for_verbatim_crisperwhisper(self) -> None:
        config = self._config("crisperwhisper-2.0-turbo", "verbatim")

        assert _resolve_pipeline_mode(config) is PipelineMode.VERBATIM

    def test_should_keep_the_pipeline_for_intended_crisperwhisper(self) -> None:
        config = self._config("crisperwhisper-2.0-turbo", "intended")

        assert _resolve_pipeline_mode(config) is PipelineMode.BATCH

    def test_should_ignore_a_stale_verbatim_value_on_another_model(self) -> None:
        config = self._config("parakeet-ultra", "verbatim")

        assert _resolve_pipeline_mode(config) is PipelineMode.BATCH

    def test_should_ignore_a_stale_verbatim_value_when_streaming(self) -> None:
        config = self._config(
            "nemotron-speech-streaming-en-0.6b", "verbatim", "streaming",
        )

        assert _resolve_pipeline_mode(config) is PipelineMode.STREAMING


class TestEditCaptureSwitch:
    """Edit capture records keystrokes, so it must stay off unless asked for."""

    def _paths(self, tmp_path: Path) -> AppPaths:
        paths = AppPaths(base=tmp_path)
        paths.ensure_dirs()
        session = paths.diagnostics_dir / "session"
        session.mkdir()
        paths.activate_session(session)
        return paths

    def test_should_stay_off_when_the_setting_is_absent(self, tmp_path: Path):
        capture = _build_edit_capture(FakeConfig(), self._paths(tmp_path))

        assert capture is None

    def test_should_stay_off_when_the_setting_is_merely_truthy(
        self, tmp_path: Path,
    ):
        config = FakeConfig({"diagnostics.capture_edits": "yes"})

        capture = _build_edit_capture(config, self._paths(tmp_path))

        assert capture is None

    def test_should_stay_off_when_diagnostics_are_disabled(self, tmp_path: Path):
        config = FakeConfig({
            "diagnostics.capture_edits": True,
            "diagnostics.enabled": False,
        })

        capture = _build_edit_capture(config, self._paths(tmp_path))

        assert capture is None

    def test_should_write_nothing_while_it_is_off(self, tmp_path: Path):
        paths = self._paths(tmp_path)

        _build_edit_capture(FakeConfig(), paths)

        assert not (paths.session_dir / "edits.jsonl").exists()

    def test_should_start_when_explicitly_enabled(self, tmp_path: Path):
        config = FakeConfig({
            "diagnostics.capture_edits": True,
            "diagnostics.capture_content": True,
        })

        capture = _build_edit_capture(config, self._paths(tmp_path))

        assert capture is not None

    def test_should_hand_the_hotkey_hook_only_the_paste_memory_while_off(self):
        assert _key_observer(None) == bootstrap._paste_memory.observe_key

    def test_should_hand_the_output_chain_a_no_op_reporter_while_off(self):
        reporter = _delivery_reporter(None, PipelineMode.BATCH, OutputMode.DIRECT_PASTE)

        assert reporter is _ignore_delivery


class TestMicrophoneWiring:
    def _audio(self, peak: float):
        import numpy as np

        from skryb.recording.audio_segment import AudioSegment

        data = (np.sin(np.arange(16000) / 5) * peak).astype(np.float32)
        return AudioSegment(data=data, sample_rate=16000)

    def test_boosts_quiet_speech_unless_switched_off(self):
        quiet = self._audio(0.05)

        boosted = bootstrap._prepare_audio(FakeConfig(), quiet)
        untouched = bootstrap._prepare_audio(
            FakeConfig({"audio.boost_quiet_speech": False}), quiet,
        )

        assert float(abs(boosted.data).max()) > 0.4
        assert untouched is quiet

    @pytest.mark.parametrize("saved", [None, "", 0])
    def test_an_unset_microphone_means_the_windows_default(self, saved):
        assert bootstrap._chosen_microphone(FakeConfig({"audio.input_device": saved})) == ""

    def test_choosing_from_the_tray_saves_it(self):
        config = FakeConfig()

        bootstrap._choose_microphone(config, "Microphone (Maono PD300X Microp")

        assert config.get("audio.input_device") == "Microphone (Maono PD300X Microp"


class TestLearningWiring:
    def _paths(self, tmp_path: Path) -> AppPaths:
        paths = AppPaths(base=tmp_path)
        paths.ensure_dirs()
        session = paths.diagnostics_dir / "session"
        session.mkdir()
        paths.activate_session(session)
        return paths

    def test_the_stale_batch_block_is_dropped_and_history_is_left_alone(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({
            "batch": {"cleanup": "none", "deadline_seconds": 2, "model_path": "C:/old.gguf"},
            "history": {"keep_days": 30},
            "hotkey": {"combination": "ctrl+shift+alt+m"},
        }), encoding="utf-8")
        config = JsonConfigAdapter(path)
        config.load()

        bootstrap.migrate_stale_batch_block(config)

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert "batch" not in saved
        assert saved["history"]["keep_days"] == 30
        assert config.get("hotkey.combination") == "ctrl+shift+alt+m"

    def test_a_config_without_a_batch_block_is_not_rewritten(self):
        config = FakeConfig({"hotkey.combination": "f9"})

        bootstrap.migrate_stale_batch_block(config)

        assert config.save_count == 0

    def test_learning_alone_runs_the_edit_watcher_but_writes_no_edits_log(self, tmp_path):
        paths = self._paths(tmp_path)
        learner = MagicMock()

        capture = _build_edit_capture(FakeConfig(), paths, learner)
        assert capture is not None
        capture._recorder.delivered("Ship it to Jon.", "batch", "direct_paste")

        learner.observe.assert_called_once()
        assert not (paths.session_dir / "edits.jsonl").exists()

    @pytest.mark.parametrize(
        "saved",
        [
            {"dictionary.learn_words": False},
            {"writing.learn_words": False, "writing.learn_style": True, "writing.cleanup": "light"},
        ],
        ids=["switched-off", "switched-off-in-an-old-config"],
    )
    def test_learning_switched_off_in_settings_leaves_the_watcher_off(self, tmp_path, saved):
        assert _build_edit_capture(FakeConfig(saved), self._paths(tmp_path), MagicMock()) is None

    def test_the_edits_log_and_the_learner_both_receive_events_when_both_are_on(self, tmp_path):
        paths = self._paths(tmp_path)
        learner = MagicMock()
        config = FakeConfig({
            "diagnostics.capture_edits": True,
            "diagnostics.capture_content": True,
        })

        capture = _build_edit_capture(config, paths, learner)
        assert capture is not None
        capture._recorder.delivered("Ship it to Jon.", "batch", "direct_paste")

        learner.observe.assert_called_once()
        assert (paths.session_dir / "edits.jsonl").read_text(encoding="utf-8").count("\n") == 1

    def test_background_work_starts_on_the_first_idle_and_only_once(self):
        bus = SimpleEventBus()
        started: list[str] = []
        bootstrap._start_once_idle(bus, lambda: started.append("edits"), lambda: started.append("other"))

        bus.publish(StateChangedEvent(old_state="loading", new_state="error"))
        assert started == []
        bus.publish(StateChangedEvent(old_state="error", new_state="idle"))
        bus.publish(StateChangedEvent(old_state="recording", new_state="idle"))

        assert started == ["edits", "other"]

    def test_every_shortcut_action_reaches_the_use_case(self):
        use_case = MagicMock()

        actions = bootstrap._hotkey_actions(use_case)

        assert actions.paste_last is use_case.paste_last
        assert actions.copy_last is use_case.copy_last
        assert actions.dictate_activate is use_case.on_hotkey_activated

    @patch("skryb.bootstrap._build_hotkey_listener")
    def test_switching_learning_on_hands_the_new_edit_watcher_to_a_new_listener(
        self, build_listener,
    ):
        app = MagicMock()
        watcher = MagicMock()
        changes = ChangeSet.between(_draft(learn_words=False), _draft(learn_words=True))

        with patch("skryb.bootstrap._build_output"):
            apply_settings(
                changes, FakeConfig(), MagicMock(), app, gui_dispatcher=MagicMock(),
                edit_capture=watcher,
            )

        assert changes.learning_turned_on
        build_listener.assert_called_once()
        app.set_hotkey_listener.assert_called_once_with(build_listener.return_value)


class TestAppAwareOutputChain:
    def _hooks(self, failures: list, method=None):
        from skryb.output.app_aware_delivery import DeliveryHooks
        from skryb.output.last_dictation import LastDictation
        from skryb.output.paste_method import PasteMethod

        return DeliveryHooks(
            last_dictation=LastDictation(),
            paste_method=method or (lambda: PasteMethod.CTRL_V),
            on_failed=lambda failure, text: failures.append((failure.value, text)),
        )

    @patch("skryb.bootstrap.text_field_focused", return_value=False)
    @patch("skryb.bootstrap.read_caret_context", return_value=NO_CONTEXT)
    @patch("skryb.bootstrap.Win32ClipboardPreserver")
    @patch("skryb.bootstrap.Win32KeySender")
    @patch("skryb.bootstrap.Win32ClipboardWriter")
    def test_with_no_text_field_the_text_goes_to_the_clipboard_and_nothing_is_typed(
        self, writer_cls, sender_cls, preserver_cls, _context, _field,
    ):
        failures: list = []
        dispatcher = MagicMock()
        dispatcher.dispatch.side_effect = lambda action: action()
        hooks = self._hooks(failures)

        output = _build_output(OutputMode.DIRECT_PASTE, dispatcher, hooks=hooks)
        output.output("Meeting moved to Thursday.")

        writer_cls.return_value.set_text.assert_called_once_with("Meeting moved to Thursday. ")
        sender_cls.return_value.send_paste.assert_not_called()
        preserver_cls.return_value.restore.assert_not_called()
        assert failures == [("no_text_field", "Meeting moved to Thursday. ")]
        assert hooks.last_dictation.text() == "Meeting moved to Thursday. "

    @patch("skryb.bootstrap.caps_lock_enabled", return_value=False)
    @patch("skryb.bootstrap.Win32ModifierKeyStateReader")
    @patch("skryb.bootstrap.text_field_focused", return_value=True)
    @patch("skryb.bootstrap.read_caret_context", return_value=NO_CONTEXT)
    @patch("skryb.bootstrap.Win32ClipboardPreserver")
    @patch("skryb.bootstrap.Win32KeySender")
    @patch("skryb.bootstrap.Win32ClipboardWriter")
    def test_a_text_field_gets_the_paste_with_the_method_for_its_app(
        self, writer_cls, sender_cls, preserver_cls, _context, _field,
        modifier_reader_cls, _caps,
    ):
        from skryb.output.paste_method import PasteMethod

        failures: list = []
        dispatcher = MagicMock()
        dispatcher.dispatch.side_effect = lambda action: action()
        modifier_reader_cls.return_value.any_modifier_down.return_value = False
        hooks = self._hooks(failures, method=lambda: PasteMethod.SHIFT_INSERT)

        output = _build_output(OutputMode.DIRECT_PASTE, dispatcher, hooks=hooks)
        output.output("ls -la")
        hooks.last_dictation.paste()

        assert sender_cls.return_value.send_paste.call_args_list == [
            call(PasteMethod.SHIFT_INSERT, "ls -la"),
            call(PasteMethod.SHIFT_INSERT, "ls -la"),
        ]
        assert preserver_cls.return_value.restore.call_count == 2
        assert failures == []

    def test_clipboard_only_and_streaming_never_ask_about_text_fields(self):
        from skryb.output.app_aware_delivery import AppAwareDelivery
        from skryb.output.last_dictation import LastDictationDecorator

        hooks = self._hooks([])

        clipboard = _build_output(OutputMode.CLIPBOARD_ONLY, MagicMock(), hooks=hooks)
        streaming = _build_output(
            OutputMode.DIRECT_PASTE, MagicMock(), pipeline_mode=PipelineMode.STREAMING,
            hooks=hooks,
        )
        batch = _build_output(OutputMode.DIRECT_PASTE, MagicMock(), hooks=hooks)

        assert isinstance(clipboard._inner, LastDictationDecorator)
        assert isinstance(clipboard._inner._inner, GuiThreadOutputDecorator)
        assert isinstance(streaming._inner, GuiThreadOutputDecorator)
        assert isinstance(batch._inner._inner, AppAwareDelivery)

    def test_a_changed_retention_restarts_history_pruning(self):
        pruner = MagicMock()
        changes = ChangeSet(
            engine_reload_required=False, hotkey_changed=False, output_changed=False,
            history_retention_changed=True,
        )

        apply_settings(
            changes, FakeConfig(), MagicMock(), MagicMock(), gui_dispatcher=MagicMock(),
            history_pruner=pruner,
        )

        pruner.prune_now.assert_called_once_with()

    def test_an_unchanged_retention_prunes_nothing(self):
        pruner = MagicMock()
        changes = ChangeSet(
            engine_reload_required=False, hotkey_changed=False, output_changed=False,
        )

        apply_settings(
            changes, FakeConfig(), MagicMock(), MagicMock(), gui_dispatcher=MagicMock(),
            history_pruner=pruner,
        )

        pruner.prune_now.assert_not_called()


def _draft(**overrides):
    from skryb.output.output_mode import OutputMode
    from skryb.settings.settings_draft import SettingsDraft

    fields = {
        "model_size": "crisperwhisper-2.0-turbo",
        "language": "en",
        "device": "auto",
        "engine": "auto",
        "output_mode": OutputMode.DIRECT_PASTE,
    }
    fields.update(overrides)
    return SettingsDraft(**fields)
