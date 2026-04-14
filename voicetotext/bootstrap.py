import logging
import sys
import threading

from voicetotext.core.action_dispatcher import ActionDispatcher, ThreadedActionDispatcher
from voicetotext.core.debug_logger import setup_debug_logger
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.core.paths import AppPaths
from voicetotext.core.state_machine import AppStateMachine
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.hotkeys.key_parser import parse_hotkey_combo
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.pynput_adapter import PynputAdapter
from voicetotext.hotkeys.push_to_talk_strategy import PushToTalkStrategy
from voicetotext.hotkeys.toggle_strategy import ToggleStrategy
from voicetotext.output.clipboard_adapter import ClipboardAdapter
from voicetotext.output.clipboard_preserving_decorator import (
    ClipboardPreservingDecorator,
)
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from voicetotext.output.sleeper import TimeSleeper
from voicetotext.output.win32_clipboard_preserver import Win32ClipboardPreserver
from voicetotext.recording.sounddevice_adapter import SoundDeviceAdapter
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import (
    EngineHotSwapCoordinator,
    NullEngineHotSwap,
)
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.service import SettingsService
from voicetotext.settings.settings_draft import ChangeSet
from voicetotext.settings.window import SettingsWindow
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_factory import (
    create_engine,
    create_model_manager,
)
from voicetotext.transcription.model_manager import ModelManager
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase
from voicetotext.tray.app import SystemTrayApp


_logger = logging.getLogger(__name__)

_OUTPUT_MODE_RENAMES = {"auto_paste": "paste_and_clipboard"}


def migrate_output_mode(config: JsonConfigAdapter) -> None:
    current = config.get("output.mode")
    replacement = _OUTPUT_MODE_RENAMES.get(current)
    if replacement is None:
        return
    _logger.info("Migrating output mode '%s' -> '%s'.", current, replacement)
    config.set("output.mode", replacement)
    config.save()


def _build_output(config: JsonConfigAdapter) -> TextOutputHandler:
    mode = config.get("output.mode")
    if mode == "clipboard_only":
        return ClipboardAdapter()
    sleeper = TimeSleeper()
    if mode == "paste_and_clipboard":
        return PasteAndClipboardAdapter(sleeper=sleeper)
    return ClipboardPreservingDecorator(
        inner=PasteAndClipboardAdapter(sleeper=sleeper),
        preserver=Win32ClipboardPreserver(),
        sleeper=sleeper,
    )


def _build_hotkey_listener(config: JsonConfigAdapter, use_case: RecordAndTranscribeUseCase) -> HotkeyListener:
    combo_string = config.get("hotkey.combination", "ctrl+alt+m")
    keys = parse_hotkey_combo(combo_string)

    if config.get("hotkey.mode") == "push_to_talk":
        strategy = PushToTalkStrategy(keys=keys)
    else:
        strategy = ToggleStrategy(keys=keys)

    listener = PynputAdapter(strategy)
    listener.set_on_activate(use_case.on_hotkey_activated)
    listener.set_on_deactivate(use_case.on_hotkey_deactivated)
    return listener



def _load_engine_with_fallback(
    config: JsonConfigAdapter, paths: AppPaths,
) -> TranscriptionEngine:
    engine = create_engine(config, paths)
    try:
        engine.load_model()
        return engine
    except (RuntimeError, OSError) as exc:
        _logger.warning("load_engine: primary device failed: %s", exc)

    _logger.info("load_engine: retrying on CPU")
    engine = create_engine(config, paths, device_override="cpu")
    engine.load_model()
    return engine


def _build_settings_service(
    config: JsonConfigAdapter,
    paths: AppPaths,
    event_bus: SimpleEventBus,
    model_manager: ModelManager,
    use_case: RecordAndTranscribeUseCase,
    action_dispatcher: ActionDispatcher,
) -> SettingsService:
    install_cache = ModelInstallationCache(model_manager, event_bus)
    downloads = DownloadCoordinator(model_manager)
    hot_swap = EngineHotSwapCoordinator(
        use_case=use_case,
        event_bus=event_bus,
        loader=lambda: _load_engine_with_fallback(config, paths),
        action_dispatcher=action_dispatcher,
    )
    service = SettingsService(
        config=config,
        model_manager=model_manager,
        event_bus=event_bus,
        install_cache=install_cache,
        downloads=downloads,
        hot_swap=hot_swap,
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
) -> SettingsService:
    """Build a SettingsService for the first-run setup window.

    No engine is loaded yet (the user hasn't picked a model), so the
    service uses a :class:`NullEngineHotSwap` — the first-run window
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
    )


def build_app(
    config: JsonConfigAdapter,
    paths: AppPaths,
    event_bus: SimpleEventBus | None = None,
    download_manager: DownloadManager | None = None,
) -> SystemTrayApp:
    setup_debug_logger(paths.base / "debug.log", console=sys.stderr.isatty())

    from voicetotext.transcription.model_info import migrate_removed_models
    migrate_removed_models(config)
    migrate_output_mode(config)

    if event_bus is None:
        event_bus = SimpleEventBus()
    if download_manager is None:
        download_manager = DownloadManager(event_bus=event_bus)

    state_machine = AppStateMachine(event_bus)

    model_name = config.get("transcription.model_size")
    _logger.info("Loading Whisper model '%s' ...", model_name)
    engine = _load_engine_with_fallback(config, paths)
    _logger.info("Model loaded.")

    recorder = SoundDeviceAdapter(
        sample_rate=config.get("audio.sample_rate", 16000),
    )

    output = _build_output(config)

    action_dispatcher: ActionDispatcher = ThreadedActionDispatcher()

    use_case = RecordAndTranscribeUseCase(
        recorder=recorder,
        engine=engine,
        output=output,
        state_machine=state_machine,
        event_bus=event_bus,
        action_dispatcher=action_dispatcher,
    )

    listener = _build_hotkey_listener(config, use_case)

    model_manager = create_model_manager(paths.models_dir, download_manager)

    settings_service = _build_settings_service(
        config=config,
        paths=paths,
        event_bus=event_bus,
        model_manager=model_manager,
        use_case=use_case,
        action_dispatcher=action_dispatcher,
    )

    def open_settings() -> None:
        SettingsWindow(
            service=settings_service,
            first_run=False,
            on_save=lambda changes: apply_settings(changes, config, use_case, app),
        ).run()

    app = SystemTrayApp(
        use_case=use_case,
        state_machine=state_machine,
        event_bus=event_bus,
        hotkey_listener=listener,
        on_open_settings=open_settings,
        download_manager=download_manager,
    )

    config.save()
    combo = config.get("hotkey.combination", "ctrl+alt+m")
    mode = config.get("hotkey.mode", "toggle")
    _logger.info("Ready. Press %s to start recording (%s mode).", combo, mode)

    return app


def apply_settings(
    changes: ChangeSet,
    config: JsonConfigAdapter,
    use_case: RecordAndTranscribeUseCase,
    app: SystemTrayApp,
) -> None:
    """Apply settings changes that don't require a model hot-swap.

    Model swaps are started asynchronously by the settings window via
    :meth:`SettingsService.trigger_engine_swap` so they don't block the
    UI; this callback only handles the lightweight dispatches.
    """
    if changes.output_changed:
        use_case.set_output(_build_output(config))

    if changes.hotkey_changed:
        new_listener = _build_hotkey_listener(config, use_case)
        app.set_hotkey_listener(new_listener)
