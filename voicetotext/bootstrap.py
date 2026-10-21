import atexit
import functools
import logging
import threading
import time
import tkinter
from collections.abc import Callable
from pathlib import Path

from voicetotext.core.action_dispatcher import (
    ActionDispatcher,
    ThreadedActionDispatcher,
)
from voicetotext.core.build_info import (
    RunningBuild,
    source_changed_since,
    source_checkout,
)
from voicetotext.core.checkout_update import pull_main
from voicetotext.core.dev_install import start_another_copy
from voicetotext.core.event_bus import EventBus, SimpleEventBus
from voicetotext.core.events import (
    DeliveryFailedEvent,
    EngineLoadFailedEvent,
    MicrophoneMissingEvent,
    StateChangedEvent,
    WritingNoticeEvent,
    WritingServerFailedEvent,
)
from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    EngineWarmupError,
)
from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.core.paths import AppPaths
from voicetotext.core.self_update import SelfUpdate
from voicetotext.core.single_instance import SecondLaunches
from voicetotext.core.startup_log import milestone, process_started_at, session_id
from voicetotext.core.timestamps import utc_now
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.diagnostics.session import SessionDiagnostics, content_recording_enabled
from voicetotext.diagnostics.jsonl import JsonlWriter
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.edit_capture.delivery_recording_decorator import (
    DeliveryRecordingDecorator,
)
from voicetotext.edit_capture.observer import Win32EditObserver
from voicetotext.edit_capture.recorder import EditRecorder
from voicetotext.edit_capture.win32_foreground import (
    foreground_hwnd,
    foreground_window,
)
from voicetotext.edit_capture.win32_input import caps_lock_enabled, click_count
from voicetotext.hotkeys.actions import HotkeyActions, strategies_for
from voicetotext.hotkeys.bindings import (
    BindingAction,
    migrate_legacy_hotkey,
    parse_bindings,
)
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.win32_copilot_key import copilot_key_acts_as_right_ctrl
from voicetotext.hotkeys.win32_hook_adapter import Win32HookAdapter
from voicetotext.hotkeys.win32_keyboard_hook import KeyObserver, ignore_key_event
from voicetotext.output.app_aware_delivery import (
    AppAwareDelivery,
    DeliveryFailure,
    DeliveryHooks,
)
from voicetotext.output.caps_lock_uppercase_decorator import (
    CapsLockUppercaseDecorator,
)
from voicetotext.output.caret_context import log_caret_text_when
from voicetotext.output.clipboard_adapter import ClipboardAdapter
from voicetotext.output.clipboard_preserving_decorator import (
    ClipboardPreservingDecorator,
)
from voicetotext.output.dictionary_config import DictionaryConfig
from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.output.gui_thread_output_decorator import (
    GuiThreadOutputDecorator,
)
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.last_dictation import LastDictation, LastDictationDecorator
from voicetotext.output.modifier_release_waiting_key_sender import (
    ModifierReleaseWaitingKeySender,
)
from voicetotext.output.output_mode import OutputMode
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from voicetotext.output.paste_memory import PasteMemory, read_or_remember
from voicetotext.output.paste_method import PasteMethod, paste_method_for
from voicetotext.output.sleeper import TimeSleeper
from voicetotext.output.text_filter_decorator import (
    TextFilterDecorator,
    load_dictionary,
)
from voicetotext.output.text_pipeline import PipelineMode, build_pipeline
from voicetotext.output.win32_caret_context import (
    read_caret_context,
    text_field_focused,
)
from voicetotext.output.tk_gui_thread_dispatcher import TkGuiThreadDispatcher
from voicetotext.output.win32_clipboard_preserver import Win32ClipboardPreserver
from voicetotext.output.win32_clipboard_writer import Win32ClipboardWriter
from voicetotext.output.win32_key_sender import (
    VOICETOTEXT_INPUT_TAG,
    Win32KeySender,
    send_menu_mask,
)
from voicetotext.output.win32_modifier_key_state_reader import (
    Win32ModifierKeyStateReader,
)
from voicetotext.history.retention import HistoryPruner, history_retention_days
from voicetotext.history.service import HistoryService
from voicetotext.recording.audio_ducking import AudioDucking, core_audio_sessions
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.hot_plug import MicrophoneHotPlug
from voicetotext.recording.input_devices import Microphones
from voicetotext.recording.notice import Notice
from voicetotext.recording.overlay import RecordingOverlay
from voicetotext.recording.preferences import (
    RecordingPreferences,
    load_pill_position,
    save_pill_position,
)
from voicetotext.recording.silence_watchdog import SilenceWatchdog
from voicetotext.recording.recording_cue_player import CueBeeper, EngineReadyBeeper
from voicetotext.recording.recovery_notices import RecoveryNotices
from voicetotext.recording.sounddevice_adapter import SoundDeviceAdapter
from voicetotext.recording.sounddevice_cue_player import SoundDeviceCuePlayer
from voicetotext.recording.speech_level import boost_quiet_speech
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import (
    EngineHotSwapCoordinator,
    NullEngineHotSwap,
)
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.service import SettingsService
from voicetotext.settings.settings_draft import ChangeSet
from voicetotext.settings.win32_capture_bridge import Win32CaptureBridge
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_preload import EnginePreload
from voicetotext.transcription.engine_factory import (
    build_composite_model_manager,
    create_engine_for_active_model,
    engines_with_model_on_disk,
    resolve_device,
    resolve_engine_id,
    selectable_engine_ids,
)
from voicetotext.transcription.engine_install import (
    EngineInstaller,
    make_engine_installer,
)
from voicetotext.transcription.engine_registry import EngineRegistry
from voicetotext.transcription.engine_types import EngineId
from voicetotext.transcription.hardware_detector import HardwareDetector
from voicetotext.transcription.license_policy import LicensePolicy
from voicetotext.transcription.model_info import (
    DEFAULT_MODEL,
    ModelInfo,
    get_model_info,
    has_crisperwhisper_modes,
)
from voicetotext.transcription.model_manager import ModelManager
from voicetotext.transcription.placeholder_engine import PlaceholderEngine
from voicetotext.transcription.streaming.live_streaming_controller import (
    LiveStreamingController,
)
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionEngine,
)
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase
from voicetotext.tray.app import SystemTrayApp
from voicetotext.tray.win32_tray_adapter import build_win32_tray_adapter
from voicetotext.ui.api_handlers import CoreApi, ShortcutCapture
from voicetotext.ui.api_server import ApiServer
from voicetotext.ui.launcher import ShellLauncher
from voicetotext.writing.install import WritingPaths, managed_root, server_files
from voicetotext.writing.learning import Learner
from voicetotext.writing.llama_server import LlamaServer
from voicetotext.writing.service import WritingService, warm_up_writing_model
from voicetotext.writing.settings import read_writing_settings
from voicetotext.writing.transforms import TransformStore
from voicetotext.writing.voice_edit import VoiceEditor

_logger = logging.getLogger(__name__)


def migrate_output_mode(config: JsonConfigAdapter) -> None:
    """Rewrite a legacy or unknown ``output.mode`` config value in place.

    Delegates to :meth:`OutputMode.migrate` for the string-to-enum
    mapping and only touches the config when the canonical form
    differs from what was stored.
    """
    current = config.get("output.mode")
    canonical = OutputMode.migrate(current).value
    if canonical == current:
        return
    _logger.info("Migrating output mode '%s' -> '%s'.", current, canonical)
    config.set("output.mode", canonical)
    config.save()


def migrate_language(config: JsonConfigAdapter) -> None:
    """Rewrite a legacy language of ``None`` or ``"auto"`` to ``"en"``."""
    current = config.get("transcription.language")
    if current not in (None, "auto"):
        return
    _logger.info("Migrating transcription language %r -> 'en'.", current)
    config.set("transcription.language", "en")
    config.save()


def migrate_stale_batch_block(config: JsonConfigAdapter) -> None:
    """Drop the ``batch`` block an abandoned writing feature left behind.

    Its values mean nothing to the ``writing`` settings that replaced it, so
    they are removed rather than carried over.
    """
    document = config.get_all()
    if "batch" not in document:
        return
    _logger.info("Removing the stale 'batch' settings block.")
    del document["batch"]
    config.replace_all(document)
    config.save()


def migrate_hotkey_bindings(config: JsonConfigAdapter) -> None:
    """Turn a single ``hotkey.combination``/``hotkey.mode`` into ``hotkey.bindings``."""
    if not migrate_legacy_hotkey(config):
        return
    _logger.info("Migrated the hotkey to shortcut bindings.")
    config.save()


def _ignore_delivery(text: str) -> None:
    return None


# One per process, like the keyboard hook and the caret it stands in for.
_paste_memory = PasteMemory(foreground_hwnd, click_count, VOICETOTEXT_INPUT_TAG)
_REREAD_CARET_AFTER_SECONDS = 0.1
# A recording that starts after a minute without a decode wakes the engine.
_WAKE_AFTER_IDLE_SECONDS = 60.0


# Folders that a sync client uploads. Edit capture records what the user
# typed, so a session log inside one leaves the machine.
_SYNCED_FOLDER_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive")


def _warn_if_synced_folder(directory: Path) -> None:
    lowered = str(directory).casefold()
    if not any(marker in lowered for marker in _SYNCED_FOLDER_MARKERS):
        return
    _logger.warning(
        "Edit capture is writing keystrokes to '%s', which looks like a "
        "synced folder. Move diagnostics.directory somewhere local.",
        directory,
    )


def _edit_capture_enabled(config: JsonConfigAdapter) -> bool:
    return (
        content_recording_enabled(config)
        and config.get("diagnostics.capture_edits", False) is True
    )


def _learning_enabled(config: JsonConfigAdapter, learner: Learner | None) -> bool:
    return learner is not None and read_writing_settings(config).learns


def _build_edit_capture(
    config: JsonConfigAdapter, paths: AppPaths, learner: Learner | None = None,
) -> Win32EditObserver | None:
    """The keystroke watcher, or nothing unless something needs it.

    It feeds two sinks. ``edits.jsonl`` records what the user types into
    the window a transcription was pasted into, so it stays off by default
    and config-only. The learner keeps only word corrections and style
    tallies, and follows the Writing page's learning checkboxes.
    """
    capture = _edit_capture_enabled(config)
    if not capture and not _learning_enabled(config, learner):
        return None
    write_jsonl: Callable[[dict[str, object]], None] | None = None
    if capture:
        _warn_if_synced_folder(paths.diagnostics_dir)
        write_jsonl = JsonlWriter(paths.session_dir / "edits.jsonl").append
        _logger.info("Edit capture is on; writing to %s", paths.session_dir)

    def write(event: dict[str, object]) -> None:
        if write_jsonl is not None and _edit_capture_enabled(config):
            write_jsonl(event)
        if learner is not None and _learning_enabled(config, learner):
            learner.observe(event)

    return Win32EditObserver(EditRecorder(
        write, utc_now,
        lambda: (write_jsonl is not None and _edit_capture_enabled(config))
        or _learning_enabled(config, learner),
    ))


def _delivery_reporter(
    capture: Win32EditObserver | None,
    pipeline_mode: PipelineMode,
    output_mode: OutputMode,
) -> Callable[[str], None]:
    if capture is None:
        return _ignore_delivery
    return lambda text: capture.observe_delivery(
        text, pipeline_mode.value, output_mode.value,
    )


def _key_observer(capture: Win32EditObserver | None) -> KeyObserver:
    if capture is None:
        return _paste_memory.observe_key

    def observe(is_key_down: bool, vk: int, flags: int, extra: int) -> None:
        _paste_memory.observe_key(is_key_down, vk, flags, extra)
        capture.observe_key(is_key_down, vk, flags, extra)

    return observe


def _remember_then_report(
    on_delivered: Callable[[str], None], text: str,
) -> None:
    _paste_memory.remember(text)
    on_delivered(text)


def _build_output(
    mode: OutputMode,
    gui_dispatcher: GuiThreadDispatcher,
    dictionary_path: Path | None = None,
    pipeline_mode: PipelineMode = PipelineMode.BATCH,
    models_dir: Path | None = None,
    on_delivered: Callable[[str], None] = _ignore_delivery,
    writing: WritingService | None = None,
    hooks: DeliveryHooks | None = None,
) -> TextOutputHandler:
    sleeper = TimeSleeper()
    writer = Win32ClipboardWriter(sleeper=sleeper)
    # Waiting for the hotkey's physical modifiers to come up keeps the
    # injected Ctrl+V from reaching the target as Ctrl+Shift+Alt+V,
    # which most apps silently drop.
    key_sender = ModifierReleaseWaitingKeySender(
        inner=Win32KeySender(),
        modifier_state=Win32ModifierKeyStateReader(),
        sleeper=sleeper,
    )
    # Reporting the delivery from directly above the adapter that injects
    # Ctrl+V is what keeps edit capture's pairing sound. The report and the
    # keystroke then happen on the same thread in the same order, and a mode
    # or a failure that never pastes reports nothing at all, so a later paste
    # can never claim an earlier dictation's text.
    def pasting_adapter() -> TextOutputHandler:
        return DeliveryRecordingDecorator(
            inner=PasteAndClipboardAdapter(
                sleeper=sleeper, writer=writer, key_sender=key_sender,
                paste_method=hooks.paste_method if hooks is not None else None,
            ),
            record=functools.partial(_remember_then_report, on_delivered),
        )

    builders: dict[OutputMode, Callable[[], TextOutputHandler]] = {
        OutputMode.CLIPBOARD_ONLY: lambda: ClipboardAdapter(writer=writer),
        OutputMode.PASTE_AND_CLIPBOARD: pasting_adapter,
        # The wait for the target to read the paste runs on the GUI thread,
        # which is also the target when the settings window has the focus.
        OutputMode.DIRECT_PASTE: lambda: ClipboardPreservingDecorator(
            inner=pasting_adapter(),
            preserver=Win32ClipboardPreserver(sleeper=sleeper),
            sleeper=gui_dispatcher,
        ),
    }
    # ``GetKeyState`` answers from the calling thread's message queue, so the
    # Caps Lock check sits below the GUI hop where the queue is actually
    # pumped; on the daemon worker it could report a stale toggle.
    inner = CapsLockUppercaseDecorator(builders[mode](), caps_lock_enabled)
    # Always marshal output onto the GUI thread. Win32 clipboard and
    # ``SendInput`` calls require a UI thread, and the use case runs
    # on a daemon worker. Wrapping every concrete handler with this
    # decorator means the worker never touches Win32 directly.
    gui_output: TextOutputHandler = GuiThreadOutputDecorator(
        inner=inner, dispatcher=gui_dispatcher,
    )
    # Streaming pastes fragments: none is a dictation to repeat, and each
    # would read as a paste that the next one pushed out of place.
    if hooks is not None and pipeline_mode is not PipelineMode.STREAMING:
        gui_output = _deliver_with_hooks(
            hooks, gui_output, inner, writer, gui_dispatcher,
            app_aware=mode is not OutputMode.CLIPBOARD_ONLY,
        )
    dictionary = (
        load_dictionary(dictionary_path)
        if dictionary_path is not None
        else DictionaryConfig()
    )
    return TextFilterDecorator(
        inner=gui_output,
        pipeline=build_pipeline(
            dictionary,
            pipeline_mode,
            models_dir=models_dir,
            caret_context=read_or_remember(
                read_caret_context,
                _paste_memory,
                functools.partial(time.sleep, _REREAD_CARET_AFTER_SECONDS),
            ),
            cleanup=writing.cleanup_stage(dictionary) if writing is not None else None,
        ),
    )


def _build_writing(
    config: JsonConfigAdapter,
    paths: AppPaths,
    event_bus: EventBus,
    on_dictionary_changed: Callable[[], None],
    show_notice: Callable[[str], None],
) -> WritingService:
    """The local writing model: nothing runs until ``start``, and no file is read.

    This is the one place writing notices are routed. Anything about a
    dictation or a voice edit shows in the notice beside it; the server
    giving up for good is a tray balloon, once, since no dictation is on
    screen when it happens.
    """
    event_bus.subscribe(WritingNoticeEvent, lambda event: show_notice(event.message))
    settings = functools.partial(read_writing_settings, config)
    writing_paths = WritingPaths(managed_root())
    server = LlamaServer(
        locate=lambda: server_files(writing_paths, settings()),
        log_path=paths.session_dir / "writing-server.log",
        warm_up=functools.partial(warm_up_writing_model, settings),
    )
    return WritingService(
        settings,
        writing_paths,
        server,
        Learner(
            paths.base / "learning.json",
            paths.dictionary_path,
            settings,
            on_dictionary_changed,
            session_id(),
        ),
        TransformStore(paths.base / "transforms.json"),
        notify=lambda message: event_bus.publish(WritingServerFailedEvent(message=message)),
        run_in_background=ThreadedActionDispatcher().dispatch,
        read_window=foreground_window,
    )


def _start_once_idle(event_bus: EventBus, *starters: Callable[[], None]) -> None:
    """Start background work the first time dictation is ready, never before.

    Only once: after a writing server that gave up, later dictations must
    not keep relaunching it; Retry on the Writing page does that.
    """
    def start(event: StateChangedEvent) -> None:
        if event.new_state != AppState.IDLE.value:
            return
        event_bus.unsubscribe(StateChangedEvent, start)
        for starter in starters:
            starter()

    event_bus.subscribe(StateChangedEvent, start)


def _writing_seconds(
    config: JsonConfigAdapter, writing: WritingService, audio_seconds: float,
) -> float:
    if _resolve_pipeline_mode(config) is not PipelineMode.BATCH:
        return 0.0
    return writing.estimated_seconds(audio_seconds)


def _deliver_with_hooks(
    hooks: DeliveryHooks,
    gui_output: TextOutputHandler,
    inner: TextOutputHandler,
    writer: Win32ClipboardWriter,
    gui_dispatcher: GuiThreadDispatcher,
    *,
    app_aware: bool,
) -> TextOutputHandler:
    """Remember each dictation for paste-last, and paste only into text fields."""
    delivery = gui_output
    if app_aware:
        delivery = AppAwareDelivery(
            paste=inner,
            copy=ClipboardAdapter(writer=writer),
            gui=gui_dispatcher,
            on_failed=hooks.on_failed,
            text_field_focused=text_field_focused,
        )
    hooks.last_dictation.route(
        paste=delivery,
        copy=GuiThreadOutputDecorator(
            inner=ClipboardAdapter(writer=writer), dispatcher=gui_dispatcher,
        ),
    )
    return LastDictationDecorator(delivery, hooks.last_dictation)


def _foreground_paste_method(config: JsonConfigAdapter) -> PasteMethod:
    return paste_method_for(
        foreground_window().process,
        config.get("output.paste_method_overrides", {}),
    )


def _publish_delivery_failure(
    event_bus: EventBus, failure: DeliveryFailure, text: str,
) -> None:
    event_bus.publish(DeliveryFailedEvent(reason=failure.value, text=text))


def _resolve_output_mode(config: JsonConfigAdapter) -> OutputMode:
    return OutputMode.migrate(config.get("output.mode"))


def _resolve_pipeline_mode(config: JsonConfigAdapter) -> PipelineMode:
    if _transcribes_verbatim(config):
        return PipelineMode.VERBATIM
    if config.get("transcription.mode", "batch") == "streaming":
        return PipelineMode.STREAMING
    return PipelineMode.BATCH


def _transcribes_verbatim(config: JsonConfigAdapter) -> bool:
    """Whether the active model is decoding in verbatim mode.

    The mode key is engine-specific, so it only suppresses cleanup while
    a CrisperWhisper model is selected. A value left behind from an
    earlier model must not silently disable the pipeline for engines
    that never had modes.
    """
    model = config.get("transcription.model_size", DEFAULT_MODEL)
    if not isinstance(model, str) or not has_crisperwhisper_modes(model):
        return False
    return config.get("transcription.crisperwhisper_mode", "intended") == "verbatim"


def _hotkey_actions(use_case: RecordAndTranscribeUseCase) -> HotkeyActions:
    return HotkeyActions(
        dictate_activate=use_case.on_hotkey_activated,
        dictate_deactivate=use_case.on_hotkey_deactivated,
        edit_activate=use_case.on_edit_hotkey_activated,
        edit_deactivate=use_case.on_edit_hotkey_deactivated,
        paste_last=use_case.paste_last,
        copy_last=use_case.copy_last,
    )


def _build_hotkey_listener(
    config: JsonConfigAdapter,
    use_case: RecordAndTranscribeUseCase,
    actions: HotkeyActions,
    observer: KeyObserver = ignore_key_event,
) -> HotkeyListener:
    bindings = parse_bindings(config)
    _paste_memory.set_hotkeys(
        [key.vk for key in binding.keys] for binding in bindings
    )
    return Win32HookAdapter(
        strategies_for(
            bindings, actions,
            double_tap_locks_hold=(
                config.get("hotkey.double_tap_locks_hold", True) is not False
            ),
        ),
        observer,
        cancel=functools.partial(_cancel_with_escape, use_case),
        listen_to_mouse=any(binding.uses_mouse() for binding in bindings),
        mask_menus=send_menu_mask,
    )


def _describe_dictation_shortcuts(config: JsonConfigAdapter) -> str:
    return ", ".join(
        f"{binding.combo} ({binding.mode.value})"
        for binding in parse_bindings(config)
        if binding.action is BindingAction.DICTATE
    ) or "none"



def _chosen_microphone(config: JsonConfigAdapter) -> str:
    return str(config.get("audio.input_device", "") or "")


def _wake_after_idle_seconds(
    config: JsonConfigAdapter, diagnostics: SessionDiagnostics,
) -> float:
    # A CPU engine has no clocked-down GPU to wake, and its warmup decode
    # is slow enough to hold up the dictation behind it.
    if diagnostics.active_device == "cpu":
        return 0.0
    value = config.get("transcription.wake_after_idle_seconds", _WAKE_AFTER_IDLE_SECONDS)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _WAKE_AFTER_IDLE_SECONDS
    return max(float(value), 0.0)


def _prepare_audio(config: JsonConfigAdapter, audio: AudioSegment) -> AudioSegment:
    if config.get("audio.boost_quiet_speech", True) is False:
        return audio
    return boost_quiet_speech(audio)


def _microphone_menu(
    config: JsonConfigAdapter, microphones: Microphones,
) -> tuple[list[tuple[str, str]], str]:
    names = microphones.names(rescan=True)
    labels = microphones.labels(names)
    return [(name, labels.get(name, name)) for name in names], _chosen_microphone(config)


def _choose_microphone(config: JsonConfigAdapter, name: str) -> None:
    config.set("audio.input_device", name)
    config.save()
    _logger.info("Microphone set to %r.", name or "the Windows default")


def _cancel_with_escape(use_case: RecordAndTranscribeUseCase) -> bool:
    """Esc cancels a recording in progress, and only then is it swallowed."""
    if not use_case.is_recording():
        return False
    use_case.on_cancel()
    return True


def _load_engine_with_fallback(
    config: JsonConfigAdapter,
    paths: AppPaths,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
    diagnostics: SessionDiagnostics,
    preload: EnginePreload | None = None,
) -> TranscriptionEngine:
    """Try the active engine on its preferred device, falling back to CPU.

    Never silently switches engines on failure - the user picked the
    engine explicitly (or via ``transcription.engine="auto"`` at config
    time), and a runtime swap to a different engine would download
    different model files. CPU-fallback within the same engine is the
    only automatic step. A *preload* that startup already began stands
    in for the first attempt.
    """
    started_at = time.monotonic()
    device = resolve_device(
        config, registry, hardware_detector,
        selectable_engine_ids(config, registry, paths.models_dir),
    )
    try:
        try:
            engine = (
                _take_preloaded(preload) if preload is not None
                else _load_engine(
                    config, paths, registry, hardware_detector, license_policy,
                )
            )
        except (RuntimeError, OSError) as exc:
            _logger.warning("load_engine: primary device failed: %s", exc)
            _logger.info("load_engine: retrying on CPU")
            device = "cpu"
            engine = _load_engine_on_cpu(
                config, paths, registry, hardware_detector, license_policy,
            )
    except Exception as exc:
        diagnostics.engine_load_failed(
            time.monotonic() - started_at, exc, device,
        )
        raise
    diagnostics.engine_loaded(time.monotonic() - started_at, device)
    return engine


def _load_engine(
    config: JsonConfigAdapter,
    paths: AppPaths,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
) -> TranscriptionEngine:
    engine = create_engine_for_active_model(
        config, paths, registry, hardware_detector, license_policy=license_policy,
    )
    engine.load_model()
    milestone("model-loaded")
    _warmup_safely(engine)
    return engine


def _take_preloaded(preload: EnginePreload) -> TranscriptionEngine:
    """The engine startup began loading before the tray existed."""
    engine = preload.wait()
    if not preload.warmed:
        _warmup_safely(engine)
    return engine


def _load_engine_on_cpu(
    config: JsonConfigAdapter,
    paths: AppPaths,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
) -> TranscriptionEngine:
    engine = create_engine_for_active_model(
        config, paths, registry, hardware_detector,
        license_policy=license_policy, device_override="cpu",
    )
    engine.load_model()
    milestone("model-loaded")
    _warmup_safely(engine)
    return engine


_STREAMING_DEVICE = "cuda"


def _load_streaming_engine(
    config: JsonConfigAdapter,
    paths: AppPaths,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
    diagnostics: SessionDiagnostics,
) -> StreamingTranscriptionEngine:
    """Build, load, and warm the GPU-only streaming engine for the active model.

    Streaming bypasses ``engine_factory`` entirely, so it asserts the
    licence itself rather than inheriting the factory's check.
    """
    started_at = time.monotonic()
    try:
        model_info = get_model_info(
            config.get("transcription.model_size", DEFAULT_MODEL)
        )
        license_policy.assert_accepted(model_info)
        engine = _build_streaming_engine_for(model_info, paths)
        engine.load_model()
        milestone("model-loaded")
        _warmup_safely(engine)
    except Exception as exc:
        diagnostics.engine_load_failed(
            time.monotonic() - started_at, exc, _STREAMING_DEVICE,
        )
        raise
    diagnostics.engine_loaded(
        time.monotonic() - started_at, _STREAMING_DEVICE,
    )
    return engine


def _build_streaming_engine_for(
    model_info: ModelInfo, paths: AppPaths,
) -> StreamingTranscriptionEngine:
    """Route to the streaming engine class declared by the model.

    ``nemotron_streaming`` models use the clean-room cache-aware engine; all
    other streaming models use the windowed nano-parakeet engine.
    """
    if "nemotron_streaming" in model_info.compatible_engines:
        return _build_cache_aware_streaming_engine(model_info, paths)
    from voicetotext.transcription.streaming.parakeet_streaming_engine import (
        StreamingNanoParakeetEngine,
    )

    return StreamingNanoParakeetEngine(
        repo_id=model_info.repo_id,
        model_dir=paths.models_dir,
        device=_STREAMING_DEVICE,
    )


def _build_cache_aware_streaming_engine(
    model_info: ModelInfo, paths: AppPaths,
) -> StreamingTranscriptionEngine:
    from voicetotext.transcription.streaming.cache_aware.engine import (
        CacheAwareStreamingEngine,
    )

    return CacheAwareStreamingEngine(
        repo_id=model_info.repo_id,
        model_dir=paths.models_dir,
        device=_STREAMING_DEVICE,
    )


def _load_active_engine_installer(
    config: JsonConfigAdapter,
    paths: AppPaths,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
    diagnostics: SessionDiagnostics,
    preload: EnginePreload | None = None,
) -> EngineInstaller:
    """Load the engine for the active mode and return its installer.

    Streaming is GPU-only and never falls back to CPU; batch keeps the
    CPU fallback inside ``_load_engine_with_fallback``. Startup preloads
    batch engines only, so a *preload* never reaches the streaming path.
    """
    if config.get("transcription.mode", "batch") == "streaming":
        engine = _load_streaming_engine(
            config, paths, hardware_detector, license_policy, diagnostics,
        )
        return make_engine_installer(
            engine, lambda use_case, loaded: use_case.set_streaming_engine(loaded),
        )
    engine = _load_engine_with_fallback(
        config, paths, registry, hardware_detector, license_policy, diagnostics,
        preload,
    )
    return make_engine_installer(
        engine, lambda use_case, loaded: use_case.set_engine(loaded),
    )


def _load_engine_and_attach(
    config: JsonConfigAdapter,
    paths: AppPaths,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
    diagnostics: SessionDiagnostics,
    *,
    use_case: RecordAndTranscribeUseCase,
    state_machine: AppStateMachine,
    event_bus: EventBus,
    preload: EnginePreload | None = None,
) -> None:
    """Background-thread entry point for engine load + state attach.

    Loads the mode-appropriate engine and installs it into the use case,
    then flips ``LOADING → IDLE``. Failures publish
    :class:`EngineLoadFailedEvent` and recover to ``IDLE`` (via ``ERROR``)
    so the app stays swappable - see :func:`_recover_from_failed_load`.
    Exceptions never escape - daemon threads silently drop them.
    """
    milestone("engine-load-started")
    try:
        installer = _load_active_engine_installer(
            config, paths, registry, hardware_detector, license_policy, diagnostics,
            preload,
        )
        installer(use_case)
    except Exception as exc:
        _recover_from_failed_load(exc, config, state_machine, event_bus)
        return
    milestone("warm-ready")
    state_machine.transition_to(AppState.IDLE)
    _logger.info("Engine ready.")


def _recover_from_failed_load(
    exc: Exception,
    config: JsonConfigAdapter,
    state_machine: AppStateMachine,
    event_bus: EventBus,
) -> None:
    """Surface a failed initial load, then return to a swappable IDLE state.

    ``RecordAndTranscribeUseCase._require_swappable_state`` only permits
    ``IDLE``/``LOADING``, so leaving the machine in ``ERROR`` would make
    every later model swap raise ``EngineSwapError`` and lock the app until
    restart (the reported "Can only swap engine when idle or loading"). The
    machine passes *through* ``ERROR`` - the tray paints its sticky error
    icon on :class:`EngineLoadFailedEvent` and holds it across the
    subsequent ``ERROR → IDLE`` change - then settles on ``IDLE`` so the
    user can pick or re-download a working model. Going via ``ERROR``
    (rather than straight ``LOADING → IDLE``) also avoids replaying a hotkey
    that was queued during the failed load into the still-placeholder
    engine, since that replay only fires on a direct ``LOADING → IDLE``
    transition.
    """
    _logger.exception("Engine load failed: %s", exc)
    event_bus.publish(
        EngineLoadFailedEvent(
            model_name=str(
                config.get("transcription.model_size", DEFAULT_MODEL)
            ),
            error=str(exc),
        )
    )
    state_machine.transition_to(AppState.ERROR)
    state_machine.transition_to(AppState.IDLE)


def _spawn_engine_load_thread(
    config: JsonConfigAdapter,
    paths: AppPaths,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
    diagnostics: SessionDiagnostics,
    *,
    use_case: RecordAndTranscribeUseCase,
    state_machine: AppStateMachine,
    event_bus: EventBus,
    preload: EnginePreload | None = None,
) -> None:
    thread = threading.Thread(
        target=_load_engine_and_attach,
        args=(
            config, paths, registry, hardware_detector, license_policy, diagnostics,
        ),
        kwargs={
            "use_case": use_case,
            "state_machine": state_machine,
            "event_bus": event_bus,
            "preload": preload,
        },
        daemon=True,
        name="engine-load",
    )
    thread.start()


def _warmup_safely(engine: TranscriptionEngine) -> None:
    """Run warmup, classifying failures into transient vs. unrecoverable.

    Transient failures (e.g. ``RuntimeError`` from a backend's first
    kernel compile on silent audio) are swallowed so the first real
    transcription just pays the cold-start cost. Unrecoverable failures
    - missing backend, missing model file, dead device - are re-raised
    as :class:`EngineWarmupError` so the bootstrap layer can publish
    :class:`EngineLoadFailedEvent` and surface an actionable message
    to the tray instead of leaving a half-broken engine attached.
    """
    started_at = time.monotonic()
    try:
        _run_warmup(engine)
    except (
        EngineDependencyMissingError,
        ImportError,
        FileNotFoundError,
        OSError,
    ) as exc:
        raise EngineWarmupError(str(exc)) from exc
    except Exception as exc:
        _logger.warning("Engine warmup failed (non-fatal): %s", exc)
        return
    _logger.info(
        "Engine warmup complete in %.2fs", time.monotonic() - started_at,
    )


def _run_warmup(engine: TranscriptionEngine) -> None:
    engine.warmup()


def _make_active_engine_provider(
    config: JsonConfigAdapter,
    hardware_detector: HardwareDetector,
    registry: EngineRegistry,
    models_dir: Path,
) -> Callable[[], EngineId]:
    """Return a callable the composite ModelManager polls on each op.

    Resolves the active engine the SAME way the loader does - biased
    toward whichever engine actually has loadable files on disk (see
    ``create_engine_for_active_model``). The composite routes
    ``is_installed`` to this active child, so without the on-disk bias a
    model present only under another engine's repo looked uninstalled in
    the UI even though the loader could open it. Tying both to the same
    resolution keeps the
    install state and the loader in agreement.
    """
    def provider() -> EngineId:
        model_name = config.get(
            "transcription.model_size", DEFAULT_MODEL,
        )
        model_info = get_model_info(model_name)
        return resolve_engine_id(
            config, model_info, hardware_detector, registry,
            installed_engines=engines_with_model_on_disk(
                model_info, models_dir,
            ),
        )
    return provider


_ENGINE_LOAD_TIMEOUT_SECONDS = 120.0
# A restart onto newer code waits while either is under way.
_DICTATING = (AppState.RECORDING, AppState.TRANSCRIBING)


def _build_settings_service(
    config: JsonConfigAdapter,
    paths: AppPaths,
    event_bus: SimpleEventBus,
    model_manager: ModelManager,
    use_case: RecordAndTranscribeUseCase,
    action_dispatcher: ActionDispatcher,
    state_machine: AppStateMachine,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
    diagnostics: SessionDiagnostics,
) -> SettingsService:
    install_cache = ModelInstallationCache(model_manager, event_bus)
    downloads = DownloadCoordinator(model_manager)
    hot_swap = EngineHotSwapCoordinator(
        use_case=use_case,
        event_bus=event_bus,
        loader=lambda: _load_active_engine_installer(
            config, paths, registry, hardware_detector, license_policy, diagnostics,
        ),
        action_dispatcher=action_dispatcher,
        timeout_seconds=_ENGINE_LOAD_TIMEOUT_SECONDS,
    )
    service = SettingsService(
        config=config,
        model_manager=model_manager,
        event_bus=event_bus,
        install_cache=install_cache,
        downloads=downloads,
        hot_swap=hot_swap,
        state_machine=state_machine,
        dictionary_path=paths.dictionary_path,
        license_policy=license_policy,
        registry=registry,
        hardware_detector=hardware_detector,
    )
    # Populate the cache on a background thread so the first settings
    # open doesn't pay the filesystem-scan cost.
    threading.Thread(
        target=install_cache.populate_all,
        daemon=True,
        name="install-cache-populate",
    ).start()
    return service


def build_first_run_settings_service(
    config: JsonConfigAdapter,
    event_bus: SimpleEventBus,
    model_manager: ModelManager,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    dictionary_path: Path,
    license_policy: LicensePolicy,
) -> SettingsService:
    """Build a SettingsService for the first-run setup window.

    No engine is loaded yet (the user hasn't picked a model), so the
    service uses a :class:`NullEngineHotSwap` - the first-run window
    never triggers a hot-swap because ``build_app`` will load the
    initial engine after setup completes.
    """
    install_cache = ModelInstallationCache(model_manager, event_bus)
    downloads = DownloadCoordinator(model_manager)
    hot_swap = NullEngineHotSwap()
    return SettingsService(
        config=config,
        model_manager=model_manager,
        event_bus=event_bus,
        install_cache=install_cache,
        downloads=downloads,
        hot_swap=hot_swap,
        state_machine=AppStateMachine(event_bus),
        dictionary_path=dictionary_path,
        license_policy=license_policy,
        registry=registry,
        hardware_detector=hardware_detector,
    )


def build_app(
    config: JsonConfigAdapter,
    paths: AppPaths,
    root: tkinter.Misc,
    event_bus: SimpleEventBus | None = None,
    download_manager: DownloadManager | None = None,
    *,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
    api_server: ApiServer,
    preload: EnginePreload | None = None,
    second_launches: SecondLaunches | None = None,
    check_for_updates: bool = False,
) -> SystemTrayApp:
    from voicetotext.transcription.model_info import (
        migrate_engine_field,
        migrate_removed_models,
    )
    migrate_removed_models(config)
    migrate_engine_field(config)
    migrate_output_mode(config)
    migrate_language(config)
    migrate_stale_batch_block(config)
    migrate_hotkey_bindings(config)

    if event_bus is None:
        event_bus = SimpleEventBus()
    if download_manager is None:
        download_manager = DownloadManager(event_bus=event_bus)

    # Short audio cues: record start/stop (both batch and streaming) and a
    # distinct chime when the engine becomes ready to transcribe. The event
    # bus retains the subscriptions, so the beepers outlive this function
    # without an explicit reference.
    preferences = functools.partial(RecordingPreferences.from_config, config)
    cue_player = SoundDeviceCuePlayer(
        enabled=lambda: preferences().cues_enabled,
        volume=lambda: preferences().cue_volume,
    )
    CueBeeper(event_bus, cue_player)
    EngineReadyBeeper(event_bus, cue_player)

    state_machine = AppStateMachine(event_bus, initial_state=AppState.LOADING)
    diagnostics = SessionDiagnostics(paths, config, registry, hardware_detector)
    atexit.register(diagnostics.dispose)
    log_caret_text_when(functools.partial(content_recording_enabled, config))
    # The learner adds words to dictionary.json from the edit-capture thread;
    # the rebuilt output chain is what puts them to use. Both callbacks are
    # bound late: they only run once the app is up.
    writing = _build_writing(
        config, paths, event_bus, lambda: _use_learned_words(),
        lambda message: gui_dispatcher.dispatch(lambda: notice.show(message)),
    )
    edit_capture = _build_edit_capture(config, paths, writing.learner)
    if edit_capture is not None:
        atexit.register(edit_capture.stop)

    recorder = SoundDeviceAdapter(
        sample_rate=config.get("audio.sample_rate", 16000),
        device_name=functools.partial(_chosen_microphone, config),
        on_device_missing=lambda name: event_bus.publish(
            MicrophoneMissingEvent(name=name),
        ),
        preferred_names=lambda: preferences().preferred_devices,
    )
    # Re-scanning the hardware restarts PortAudio, which would cut off a
    # recording, so it waits until the app is idle.
    microphones = Microphones(
        is_busy=lambda: state_machine.state != AppState.IDLE,
    )
    SilenceWatchdog(
        event_bus,
        level=lambda: recorder.current_rms,
        device_name=lambda: microphones.label(recorder.device_name),
        enabled=lambda: preferences().warn_when_silent,
    )
    ducking = AudioDucking(
        event_bus,
        mode=lambda: preferences().duck_other_audio,
        level=lambda: preferences().duck_level,
        list_sessions=core_audio_sessions,
    )

    # The GUI thread is the only thread that owns a window and a
    # message pump, so Win32 clipboard / SendInput calls have to run
    # there. The dispatcher's queue is drained by a poller on the Tk
    # ``root`` (the main thread), and it is injected everywhere output
    # handlers are constructed.
    gui_dispatcher = TkGuiThreadDispatcher(root)
    output_mode = _resolve_output_mode(config)
    pipeline_mode = _resolve_pipeline_mode(config)
    last_dictation = LastDictation()
    delivery_hooks = DeliveryHooks(
        last_dictation=last_dictation,
        paste_method=functools.partial(_foreground_paste_method, config),
        on_failed=functools.partial(_publish_delivery_failure, event_bus),
    )
    output = _build_output(
        output_mode, gui_dispatcher, paths.dictionary_path,
        pipeline_mode, paths.models_dir,
        _delivery_reporter(edit_capture, pipeline_mode, output_mode),
        writing=writing,
        hooks=delivery_hooks,
    )
    action_dispatcher: ActionDispatcher = ThreadedActionDispatcher()

    # The real engine load happens on a background thread so the tray
    # appears immediately. The use case needs an engine reference for
    # construction - give it a placeholder that raises if anything
    # actually tries to transcribe through it. Background thread will
    # swap in the real engine and flip the state machine to IDLE.
    streaming_controller = LiveStreamingController(recorder, output)
    record_and_transcribe_use_case = RecordAndTranscribeUseCase(
        recorder=recorder,
        engine=PlaceholderEngine(),
        output=output,
        state_machine=state_machine,
        event_bus=event_bus,
        action_dispatcher=action_dispatcher,
        diagnostics=diagnostics,
        max_recording_seconds=lambda: preferences().max_duration_seconds,
        warn_before_cap_seconds=lambda: preferences().warn_before_cap_seconds,
        streaming_controller=streaming_controller,
        is_streaming_mode=lambda: (
            config.get("transcription.mode", "batch") == "streaming"
        ),
        prepare_audio=functools.partial(_prepare_audio, config),
        voice_edit=VoiceEditor(
            writing.client,
            deadline=lambda: read_writing_settings(config).deadline_seconds,
            read_context=read_caret_context,
            read_window=foreground_window,
        ),
        writing_seconds=functools.partial(_writing_seconds, config, writing),
        last_dictation=last_dictation,
        wake_after_idle_seconds=functools.partial(
            _wake_after_idle_seconds, config, diagnostics,
        ),
    )

    def _use_learned_words() -> None:
        output_mode = _resolve_output_mode(config)
        pipeline_mode = _resolve_pipeline_mode(config)
        record_and_transcribe_use_case.set_output(
            _build_output(
                output_mode, gui_dispatcher, paths.dictionary_path,
                pipeline_mode, paths.models_dir,
                _delivery_reporter(edit_capture, pipeline_mode, output_mode),
                writing=writing,
                hooks=delivery_hooks,
            )
        )

    def _edit_watcher(changes: ChangeSet) -> Win32EditObserver | None:
        """The edit watcher, started now if learning was just switched on."""
        nonlocal edit_capture
        if edit_capture is None and changes.learning_turned_on:
            edit_capture = _build_edit_capture(config, paths, writing.learner)
            if edit_capture is not None:
                edit_capture.start()
                atexit.register(edit_capture.stop)
        return edit_capture

    hotkey_actions = _hotkey_actions(record_and_transcribe_use_case)
    listener = _build_hotkey_listener(
        config, record_and_transcribe_use_case, hotkey_actions,
        _key_observer(edit_capture),
    )
    overlay = (
        RecordingOverlay(
            root=root,
            event_bus=event_bus,
            dispatcher=gui_dispatcher,
            level_provider=lambda: recorder.current_rms,
            on_cancel=record_and_transcribe_use_case.on_cancel,
            on_pause_toggle=record_and_transcribe_use_case.on_pause_toggled,
            on_finish=record_and_transcribe_use_case.on_hotkey_deactivated,
            on_press=_paste_memory.excuse_click,
            on_tick=_paste_memory.sync_clicks,
            load_position=functools.partial(load_pill_position, config),
            save_position=functools.partial(save_pill_position, config),
        )
        if config.get("overlay.enabled", True)
        else None
    )

    active_engine_provider = _make_active_engine_provider(
        config, hardware_detector, registry, paths.models_dir,
    )
    model_manager = build_composite_model_manager(
        paths.models_dir, download_manager, active_engine_provider,
        license_policy,
    )

    settings_service = _build_settings_service(
        config=config,
        paths=paths,
        event_bus=event_bus,
        model_manager=model_manager,
        use_case=record_and_transcribe_use_case,
        action_dispatcher=action_dispatcher,
        state_machine=state_machine,
        registry=registry,
        hardware_detector=hardware_detector,
        license_policy=license_policy,
        diagnostics=diagnostics,
    )

    def apply_saved_settings(changes: ChangeSet) -> None:
        apply_settings(
            changes, config, record_and_transcribe_use_case, app,
            gui_dispatcher=gui_dispatcher,
            dictionary_path=paths.dictionary_path,
            models_dir=paths.models_dir,
            edit_capture=_edit_watcher(changes),
            writing=writing,
            hotkey_actions=hotkey_actions,
            delivery_hooks=delivery_hooks,
            history_pruner=history_pruner,
        )

    # The shell's requests arrive on the API server's threads; applying
    # settings rebuilds the output chain and the hotkey hook, which the
    # settings window always did on the GUI thread.
    api = CoreApi(
        settings_service,
        api_server.hub,
        microphones=microphones,
        capture=ShortcutCapture(
            Win32CaptureBridge(),
            functools.partial(api_server.hub.publish, "capture"),
            pause=lambda: app.pause_listener(),
            resume=lambda: app.resume_listener(),
            copilot_acts_as_right_ctrl=copilot_key_acts_as_right_ctrl,
        ),
        diagnostics_dir=paths.diagnostics_dir,
        state=lambda: state_machine.state,
        on_save=lambda changes: gui_dispatcher.dispatch(
            lambda: apply_saved_settings(changes),
        ),
        writing=writing,
        history=HistoryService(
            paths.diagnostics_dir, record_and_transcribe_use_case.retranscribe,
        ),
    )
    launcher = ShellLauncher(api_server, notify=lambda message: app.notify(message))

    def show_page(page: str) -> None:
        # Looking for the exe and starting it stay off the GUI thread, which
        # the tray menu and the notice buttons call from.
        action_dispatcher.dispatch(functools.partial(launcher.open, page))

    hot_plug = MicrophoneHotPlug(
        root,
        microphones,
        preferred=lambda: preferences().preferred_devices,
        chosen=functools.partial(_chosen_microphone, config),
        event_bus=event_bus,
        run_in_background=action_dispatcher.dispatch,
    )

    notice = Notice(root, load_position=functools.partial(load_pill_position, config))
    RecoveryNotices(
        event_bus,
        notice,
        gui_dispatcher.dispatch,
        retry=record_and_transcribe_use_case.retry_last_recording,
        open_history=functools.partial(show_page, "history"),
    )
    history_pruner = HistoryPruner(
        paths.diagnostics_dir, functools.partial(history_retention_days, config),
    )
    history_pruner.start_when_ready(event_bus)

    app = SystemTrayApp(
        use_case=record_and_transcribe_use_case,
        state_machine=state_machine,
        event_bus=event_bus,
        hotkey_listener=listener,
        tray_factory=functools.partial(
            build_win32_tray_adapter, root=root, dispatcher=gui_dispatcher,
            on_history_clicked=functools.partial(show_page, "history"),
            on_paste_last_clicked=record_and_transcribe_use_case.paste_last,
            on_copy_last_clicked=record_and_transcribe_use_case.copy_last,
            list_microphones=functools.partial(_microphone_menu, config, microphones),
            on_microphone_chosen=functools.partial(_choose_microphone, config),
            on_devices_changed=hot_plug.devices_changed,
        ),
        on_open_settings=functools.partial(show_page, "settings"),
        build_status=RunningBuild(source_checkout()).status,
        start_replacement=functools.partial(start_another_copy, "--restart"),
        download_manager=download_manager,
        owned_disposables=(
            (overlay, ducking, history_pruner, writing) if overlay is not None
            else (ducking, history_pruner, writing)
        ),
    )

    # Only once the tray app exists: saving settings and recording a
    # shortcut both reach it.
    api.serve_on(api_server)
    config.save()
    model_name = config.get("transcription.model_size")
    _logger.info(
        "Tray online. Loading '%s' in the background. "
        "Dictation shortcuts %s will queue until ready.",
        model_name, _describe_dictation_shortcuts(config),
    )

    # Keys observed before this only queue up; the observer catches up.
    _start_once_idle(
        event_bus,
        writing.start,
        *((edit_capture.start,) if edit_capture is not None else ()),
    )
    _spawn_engine_load_thread(
        config, paths, registry, hardware_detector, license_policy, diagnostics,
        use_case=record_and_transcribe_use_case,
        state_machine=state_machine,
        event_bus=event_bus,
        preload=preload,
    )
    self_update = SelfUpdate(
        pull=functools.partial(pull_main, source_checkout()),
        source_changed=functools.partial(
            source_changed_since, process_started_at().timestamp(),
        ),
        is_busy=lambda: state_machine.state in _DICTATING,
        restart=app.restart,
        tell=lambda message: gui_dispatcher.dispatch(lambda: notice.show(message)),
        on_gui_thread=gui_dispatcher.dispatch,
        in_background=action_dispatcher.dispatch,
    )
    if second_launches is not None:
        second_launches.on_launch = functools.partial(self_update.check, asked=True)
    if check_for_updates:
        self_update.check(asked=False)

    return app


def apply_settings(
    changes: ChangeSet,
    config: JsonConfigAdapter,
    use_case: RecordAndTranscribeUseCase,
    app: SystemTrayApp,
    gui_dispatcher: GuiThreadDispatcher,
    dictionary_path: Path | None = None,
    models_dir: Path | None = None,
    edit_capture: Win32EditObserver | None = None,
    writing: WritingService | None = None,
    hotkey_actions: HotkeyActions | None = None,
    delivery_hooks: DeliveryHooks | None = None,
    history_pruner: HistoryPruner | None = None,
) -> None:
    """Apply settings changes that don't require a model hot-swap.

    Model swaps are started asynchronously by the settings window via
    :meth:`SettingsService.trigger_engine_swap` so they don't block the
    UI; this callback only handles the lightweight dispatches.
    """
    if changes.output_changed:
        output_mode = _resolve_output_mode(config)
        pipeline_mode = _resolve_pipeline_mode(config)
        use_case.set_output(
            _build_output(
                output_mode, gui_dispatcher, dictionary_path,
                pipeline_mode, models_dir,
                _delivery_reporter(edit_capture, pipeline_mode, output_mode),
                writing=writing,
                hooks=delivery_hooks,
            )
        )

    if changes.writing_server_changed and writing is not None:
        writing.restart_server()

    if changes.history_retention_changed and history_pruner is not None:
        history_pruner.prune_now()

    # A just-started edit watcher only sees keys once the listener has it.
    if changes.hotkey_changed or changes.learning_turned_on:
        new_listener = _build_hotkey_listener(
            config, use_case, hotkey_actions or _hotkey_actions(use_case),
            _key_observer(edit_capture),
        )
        app.set_hotkey_listener(new_listener)
