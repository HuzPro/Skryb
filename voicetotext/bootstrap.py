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
from voicetotext.core.event_bus import EventBus, SimpleEventBus
from voicetotext.core.events import EngineLoadFailedEvent
from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    EngineWarmupError,
)
from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.core.paths import AppPaths
from voicetotext.core.startup_log import milestone
from voicetotext.core.timestamps import utc_now
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.diagnostics.session import SessionDiagnostics
from voicetotext.diagnostics.jsonl import JsonlWriter
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.edit_capture.delivery_recording_decorator import (
    DeliveryRecordingDecorator,
)
from voicetotext.edit_capture.observer import Win32EditObserver
from voicetotext.edit_capture.recorder import EditRecorder
from voicetotext.hotkeys.key_parser import parse_hotkey_combo
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.push_to_talk_strategy import PushToTalkStrategy
from voicetotext.hotkeys.toggle_strategy import ToggleStrategy
from voicetotext.hotkeys.win32_hook_adapter import Win32HookAdapter
from voicetotext.hotkeys.win32_keyboard_hook import KeyObserver, ignore_key_event
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
from voicetotext.output.modifier_release_waiting_key_sender import (
    ModifierReleaseWaitingKeySender,
)
from voicetotext.output.output_mode import OutputMode
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from voicetotext.output.sleeper import TimeSleeper
from voicetotext.output.text_filter_decorator import (
    TextFilterDecorator,
    load_dictionary,
)
from voicetotext.output.text_pipeline import PipelineMode, build_pipeline
from voicetotext.output.tk_gui_thread_dispatcher import TkGuiThreadDispatcher
from voicetotext.output.win32_clipboard_preserver import Win32ClipboardPreserver
from voicetotext.output.win32_clipboard_writer import Win32ClipboardWriter
from voicetotext.output.win32_key_sender import Win32KeySender
from voicetotext.output.win32_modifier_key_state_reader import (
    Win32ModifierKeyStateReader,
)
from voicetotext.recording.overlay import RecordingOverlay
from voicetotext.recording.recording_cue_player import CueBeeper, EngineReadyBeeper
from voicetotext.recording.sounddevice_adapter import SoundDeviceAdapter
from voicetotext.recording.sounddevice_cue_player import SoundDeviceCuePlayer
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import (
    EngineHotSwapCoordinator,
    NullEngineHotSwap,
)
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.service import SettingsService
from voicetotext.settings.settings_draft import ChangeSet
from voicetotext.settings.tk_window import TkSettingsWindow
from voicetotext.settings.win32_capture_bridge import Win32CaptureBridge
from voicetotext.transcription.engine import TranscriptionEngine
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


def _ignore_delivery(text: str) -> None:
    return None


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


def _build_edit_capture(
    config: JsonConfigAdapter, paths: AppPaths,
) -> Win32EditObserver | None:
    """The keystroke watcher, or nothing unless it was explicitly asked for.

    Off by default and deliberately config-only: it records what the user
    types into the window a transcription was pasted into.
    """
    if config.get("diagnostics.capture_edits", False) is not True:
        return None
    if config.get("diagnostics.enabled", True) is not True:
        return None
    _warn_if_synced_folder(paths.diagnostics_dir)
    writer = JsonlWriter(paths.session_dir / "edits.jsonl")
    _logger.info("Edit capture is on; writing to %s", paths.session_dir)
    return Win32EditObserver(EditRecorder(writer.append, utc_now))


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
    return ignore_key_event if capture is None else capture.observe_key


def _build_output(
    mode: OutputMode,
    gui_dispatcher: GuiThreadDispatcher,
    dictionary_path: Path | None = None,
    pipeline_mode: PipelineMode = PipelineMode.BATCH,
    models_dir: Path | None = None,
    on_delivered: Callable[[str], None] = _ignore_delivery,
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
            ),
            record=on_delivered,
        )

    builders: dict[OutputMode, Callable[[], TextOutputHandler]] = {
        OutputMode.CLIPBOARD_ONLY: lambda: ClipboardAdapter(writer=writer),
        OutputMode.PASTE_AND_CLIPBOARD: pasting_adapter,
        OutputMode.DIRECT_PASTE: lambda: ClipboardPreservingDecorator(
            inner=pasting_adapter(),
            preserver=Win32ClipboardPreserver(sleeper=sleeper),
            sleeper=sleeper,
        ),
    }
    inner = builders[mode]()
    # Always marshal output onto the GUI thread. Win32 clipboard and
    # ``SendInput`` calls require a UI thread, and the use case runs
    # on a daemon worker. Wrapping every concrete handler with this
    # decorator means the worker never touches Win32 directly.
    gui_output = GuiThreadOutputDecorator(inner=inner, dispatcher=gui_dispatcher)
    dictionary = (
        load_dictionary(dictionary_path)
        if dictionary_path is not None
        else DictionaryConfig()
    )
    return TextFilterDecorator(
        inner=gui_output,
        pipeline=build_pipeline(dictionary, pipeline_mode, models_dir=models_dir),
    )


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


def _build_hotkey_listener(
    config: JsonConfigAdapter,
    use_case: RecordAndTranscribeUseCase,
    observer: KeyObserver = ignore_key_event,
) -> HotkeyListener:
    combo_string = config.get("hotkey.combination", "ctrl+alt+m")
    keys = parse_hotkey_combo(combo_string)

    if config.get("hotkey.mode") == "push_to_talk":
        strategy = PushToTalkStrategy(keys=keys)
    else:
        strategy = ToggleStrategy(keys=keys)

    listener = Win32HookAdapter(strategy, observer)
    listener.set_on_activate(use_case.on_hotkey_activated)
    listener.set_on_deactivate(use_case.on_hotkey_deactivated)
    return listener



def _load_engine_with_fallback(
    config: JsonConfigAdapter,
    paths: AppPaths,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
    diagnostics: SessionDiagnostics,
) -> TranscriptionEngine:
    """Try the active engine on its preferred device, falling back to CPU.

    Never silently switches engines on failure - the user picked the
    engine explicitly (or via ``transcription.engine="auto"`` at config
    time), and a runtime swap to a different engine would download
    different model files. CPU-fallback within the same engine is the
    only automatic step.
    """
    started_at = time.monotonic()
    device = resolve_device(
        config, registry, hardware_detector,
        selectable_engine_ids(config, registry, paths.models_dir),
    )
    try:
        try:
            engine = _load_engine(
                config, paths, registry, hardware_detector, license_policy,
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
) -> EngineInstaller:
    """Load the engine for the active mode and return its installer.

    Streaming is GPU-only and never falls back to CPU; batch keeps the
    CPU fallback inside ``_load_engine_with_fallback``.
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
) -> SystemTrayApp:
    from voicetotext.transcription.model_info import (
        migrate_engine_field,
        migrate_removed_models,
    )
    migrate_removed_models(config)
    migrate_engine_field(config)
    migrate_output_mode(config)
    migrate_language(config)

    if event_bus is None:
        event_bus = SimpleEventBus()
    if download_manager is None:
        download_manager = DownloadManager(event_bus=event_bus)

    # Short audio cues: record start/stop (both batch and streaming) and a
    # distinct chime when the engine becomes ready to transcribe. The event
    # bus retains the subscriptions, so the beepers outlive this function
    # without an explicit reference.
    cue_player = SoundDeviceCuePlayer()
    CueBeeper(event_bus, cue_player)
    EngineReadyBeeper(event_bus, cue_player)

    state_machine = AppStateMachine(event_bus, initial_state=AppState.LOADING)
    diagnostics = SessionDiagnostics(paths, config, registry, hardware_detector)
    atexit.register(diagnostics.dispose)
    edit_capture = _build_edit_capture(config, paths)
    if edit_capture is not None:
        edit_capture.start()
        atexit.register(edit_capture.stop)

    recorder = SoundDeviceAdapter(
        sample_rate=config.get("audio.sample_rate", 16000),
    )

    # The GUI thread is the only thread that owns a window and a
    # message pump, so Win32 clipboard / SendInput calls have to run
    # there. The dispatcher's queue is drained by a poller on the Tk
    # ``root`` (the main thread), and it is injected everywhere output
    # handlers are constructed.
    gui_dispatcher = TkGuiThreadDispatcher(root)
    output_mode = _resolve_output_mode(config)
    pipeline_mode = _resolve_pipeline_mode(config)
    output = _build_output(
        output_mode, gui_dispatcher, paths.dictionary_path,
        pipeline_mode, paths.models_dir,
        _delivery_reporter(edit_capture, pipeline_mode, output_mode),
    )
    overlay = (
        RecordingOverlay(
            root=root,
            event_bus=event_bus,
            dispatcher=gui_dispatcher,
            level_provider=lambda: recorder.current_rms,
        )
        if config.get("overlay.enabled", True)
        else None
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
        max_recording_seconds=config.get(
            "recording.max_duration_seconds", 600.0,
        ),
        streaming_controller=streaming_controller,
        is_streaming_mode=lambda: (
            config.get("transcription.mode", "batch") == "streaming"
        ),
    )

    listener = _build_hotkey_listener(
        config, record_and_transcribe_use_case, _key_observer(edit_capture),
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

    settings_window_ref: dict[str, TkSettingsWindow | None] = {"current": None}

    def open_settings() -> None:
        existing = settings_window_ref["current"]
        if existing is not None and existing.is_open():
            existing.show()
            return
        window = TkSettingsWindow(
            root,
            settings_service,
            capture_bridge=Win32CaptureBridge(),
            first_run=False,
            on_save=lambda changes: apply_settings(
                changes, config, record_and_transcribe_use_case, app,
                gui_dispatcher=gui_dispatcher,
                dictionary_path=paths.dictionary_path,
                models_dir=paths.models_dir,
                edit_capture=edit_capture,
            ),
            on_capture_state_changed=lambda recording: (
                app.pause_listener() if recording else app.resume_listener()
            ),
            schedule=gui_dispatcher.dispatch,
        )
        settings_window_ref["current"] = window
        window.show()

    app = SystemTrayApp(
        use_case=record_and_transcribe_use_case,
        state_machine=state_machine,
        event_bus=event_bus,
        hotkey_listener=listener,
        tray_factory=functools.partial(
            build_win32_tray_adapter, root=root, dispatcher=gui_dispatcher,
        ),
        on_open_settings=open_settings,
        download_manager=download_manager,
        owned_disposables=(overlay,) if overlay is not None else (),
    )

    config.save()
    combo = config.get("hotkey.combination", "ctrl+alt+m")
    mode = config.get("hotkey.mode", "toggle")
    model_name = config.get("transcription.model_size")
    _logger.info(
        "Tray online. Loading '%s' in the background. "
        "Hotkey %s (%s mode) will queue until ready.",
        model_name, combo, mode,
    )

    _spawn_engine_load_thread(
        config, paths, registry, hardware_detector, license_policy, diagnostics,
        use_case=record_and_transcribe_use_case,
        state_machine=state_machine,
        event_bus=event_bus,
    )

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
            )
        )

    if changes.hotkey_changed:
        new_listener = _build_hotkey_listener(
            config, use_case, _key_observer(edit_capture),
        )
        app.set_hotkey_listener(new_listener)
