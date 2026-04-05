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
from voicetotext.transcription.faster_whisper_adapter import FasterWhisperAdapter
from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase
from voicetotext.tray.app import SystemTrayApp


def _resolve_device(configured_device: str) -> str:
    if configured_device == "auto":
        try:
            print("  Checking GPU availability...")
            import ctranslate2
            supported = ctranslate2.get_supported_compute_types("cuda")
            print(f"  CUDA supported types: {supported}")
            return "cuda"
        except Exception as e:
            print(f"  GPU not available ({e}), using CPU")
            return "cpu"
    return configured_device


def _build_engine(config: JsonConfigAdapter, paths: AppPaths) -> FasterWhisperAdapter:
    device = _resolve_device(config.get("transcription.device", "auto"))
    print(f"  Device: {device}")
    compute_type = config.get("transcription.compute_type", "float16")
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    return FasterWhisperAdapter(
        model_size=config.get("transcription.model_size", "base"),
        device=device,
        compute_type=compute_type,
        language=config.get("transcription.language"),
        beam_size=config.get("transcription.beam_size", 5),
        vad_filter=config.get("transcription.vad_filter", True),
        download_root=str(paths.models_dir),
    )


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


def build_app(
    config: JsonConfigAdapter,
    paths: AppPaths,
) -> SystemTrayApp:
    event_bus = SimpleEventBus()
    state_machine = AppStateMachine(event_bus)

    engine = _build_engine(config, paths)
    model_name = config.get("transcription.model_size")
    print(f"Loading Whisper model '{model_name}' ...")
    try:
        engine.load_model()
    except RuntimeError:
        print("  GPU failed, falling back to CPU...")
        engine = FasterWhisperAdapter(
            model_size=model_name,
            device="cpu",
            compute_type="int8",
            language=config.get("transcription.language"),
            beam_size=config.get("transcription.beam_size", 5),
            vad_filter=config.get("transcription.vad_filter", True),
            download_root=str(paths.models_dir),
        )
        engine.load_model()
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

    model_manager = HuggingFaceModelManager(paths.models_dir)

    def open_settings():
        from voicetotext.settings.window import SettingsWindow

        def on_save(changes: dict):
            apply_settings(changes, config, paths, use_case, state_machine, event_bus, app)

        SettingsWindow(
            config=config,
            model_manager=model_manager,
            first_run=False,
            on_save=on_save,
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
    if changes.get("model"):
        old_engine = use_case._engine
        new_engine = _build_engine(config, paths)
        model_name = config.get("transcription.model_size")
        print(f"Hot-swapping to model '{model_name}' ...")
        try:
            new_engine.load_model()
        except RuntimeError:
            print("  GPU failed, falling back to CPU...")
            new_engine = FasterWhisperAdapter(
                model_size=model_name,
                device="cpu",
                compute_type="int8",
                language=config.get("transcription.language"),
                beam_size=config.get("transcription.beam_size", 5),
                vad_filter=config.get("transcription.vad_filter", True),
                download_root=str(paths.models_dir),
            )
            new_engine.load_model()
        old_engine.dispose()
        use_case.set_engine(new_engine)
        print("Model swapped.")

    if changes.get("output"):
        use_case.set_output(_build_output(config))

    if changes.get("hotkey"):
        new_listener = _build_hotkey_listener(config, use_case)
        app.set_hotkey_listener(new_listener)
