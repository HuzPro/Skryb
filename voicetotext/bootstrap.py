from pathlib import Path

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.core.paths import AppPaths
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.hotkeys.key_parser import parse_hotkey_combo
from voicetotext.hotkeys.pynput_adapter import PynputAdapter
from voicetotext.hotkeys.push_to_talk_strategy import PushToTalkStrategy
from voicetotext.hotkeys.toggle_strategy import ToggleStrategy
from voicetotext.output.auto_paste_adapter import AutoPasteAdapter
from voicetotext.output.clipboard_adapter import ClipboardAdapter
from voicetotext.recording.sounddevice_adapter import SoundDeviceAdapter
from voicetotext.transcription.engine_factory import (
    create_engine,
    create_model_manager,
    resolve_engine_name,
)
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase
from voicetotext.tray.app import SystemTrayApp


def _build_output(config: JsonConfigAdapter):
    if config.get("output.mode") == "clipboard_only":
        return ClipboardAdapter()
    return AutoPasteAdapter()


def _build_hotkey_listener(config: JsonConfigAdapter, use_case: RecordAndTranscribeUseCase):
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


def _load_engine_with_fallback(engine_name, config, paths):
    engine = create_engine(engine_name, config, paths)
    try:
        engine.load_model()
        return engine
    except RuntimeError as exc:
        print(f"  Engine failed: {exc}")

    if engine_name != "cpu":
        print("  Retrying on CPU...")
        try:
            engine = create_engine(engine_name, config, paths, device_override="cpu")
            engine.load_model()
            return engine
        except RuntimeError as exc:
            print(f"  CPU fallback failed: {exc}")

    effective = engine_name if engine_name != "cpu" else "faster_whisper"
    alt = "transformers" if effective != "transformers" else "faster_whisper"
    print(f"  Falling back to {alt} engine...")
    engine = create_engine(alt, config, paths)
    engine.load_model()
    return engine


def build_app(
    config: JsonConfigAdapter,
    paths: AppPaths,
) -> SystemTrayApp:
    event_bus = SimpleEventBus()
    state_machine = AppStateMachine(event_bus)

    engine_name = resolve_engine_name(config)
    print(f"  Engine: {engine_name}")
    model_name = config.get("transcription.model_size")
    print(f"Loading Whisper model '{model_name}' ...")
    engine = _load_engine_with_fallback(engine_name, config, paths)
    print("Model loaded.")

    recorder = SoundDeviceAdapter(
        sample_rate=config.get("audio.sample_rate", 16000),
    )

    output = _build_output(config)

    use_case = RecordAndTranscribeUseCase(
        recorder=recorder,
        engine=engine,
        output=output,
        state_machine=state_machine,
        event_bus=event_bus,
    )

    listener = _build_hotkey_listener(config, use_case)

    model_manager = create_model_manager(engine_name, paths.models_dir)

    def open_settings():
        from voicetotext.settings.window import SettingsWindow

        def on_save(changes: dict):
            apply_settings(changes, config, paths, use_case, state_machine, event_bus, app)

        SettingsWindow(
            config=config,
            model_manager=model_manager,
            first_run=False,
            on_save=on_save,
            models_dir=paths.models_dir,
        ).run()

    app = SystemTrayApp(
        use_case=use_case,
        state_machine=state_machine,
        event_bus=event_bus,
        hotkey_listener=listener,
        on_open_settings=open_settings,
    )

    config.save()
    combo = config.get("hotkey.combination", "ctrl+alt+m")
    mode = config.get("hotkey.mode", "toggle")
    print(f"Ready. Press {combo} to start recording ({mode} mode).")

    return app


def apply_settings(
    changes: dict,
    config: JsonConfigAdapter,
    paths: AppPaths,
    use_case: RecordAndTranscribeUseCase,
    state_machine: AppStateMachine,
    event_bus,
    app: SystemTrayApp,
) -> None:
    if changes.get("model") or changes.get("engine"):
        old_engine = use_case._engine
        engine_name = resolve_engine_name(config)
        model_name = config.get("transcription.model_size")
        print(f"Hot-swapping to model '{model_name}' (engine: {engine_name}) ...")
        new_engine = _load_engine_with_fallback(engine_name, config, paths)
        old_engine.dispose()
        use_case.set_engine(new_engine)
        print("Model swapped.")

    if changes.get("output"):
        use_case.set_output(_build_output(config))

    if changes.get("hotkey"):
        new_listener = _build_hotkey_listener(config, use_case)
        app.set_hotkey_listener(new_listener)
