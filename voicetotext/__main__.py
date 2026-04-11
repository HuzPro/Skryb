import os
import sys
import traceback

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.core.paths import AppPaths
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.transcription.model_manager import ModelManager


def _apply_torch_env_overrides() -> None:
    """Translate VoiceToText env vars into PyTorch env vars.

    Must run **before** any code imports ``torch``, because PyTorch
    reads its allocator configuration once at import time. We do not
    import torch anywhere above this point in the startup chain.

    ``VOICETOTEXT_DISABLE_CUDA_CACHE=1`` is the escape hatch for the
    model-hot-swap VRAM leak: it sets
    ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` which disables PyTorch's
    caching allocator entirely. Tensor deallocations then return memory
    to the driver immediately, at a notable perf cost. Use only if the
    normal teardown path fails to release VRAM between model swaps.
    """
    if os.environ.get("VOICETOTEXT_DISABLE_CUDA_CACHE") == "1":
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


def main() -> None:
    _apply_torch_env_overrides()
    try:
        _run()
    except Exception:
        _show_error(traceback.format_exc())
        sys.exit(1)


def _run() -> None:
    paths = AppPaths()
    paths.ensure_dirs()

    config = JsonConfigAdapter(paths.config_path)
    config.load()

    event_bus = SimpleEventBus()
    download_manager = DownloadManager(event_bus=event_bus)

    from voicetotext.transcription.engine_factory import create_model_manager
    model_manager = create_model_manager(paths.models_dir, download_manager)

    if paths.is_first_run() or not _configured_model_installed(config, model_manager):
        _run_first_time_setup(config, model_manager, event_bus)

    from voicetotext.bootstrap import build_app

    app = build_app(
        config, paths, event_bus=event_bus, download_manager=download_manager
    )
    app.run()


def _configured_model_installed(
    config: ConfigurationStore, model_manager: ModelManager
) -> bool:
    model_name = config.get("transcription.model_size", "base")
    return model_manager.is_installed(model_name)


def _run_first_time_setup(
    config: JsonConfigAdapter,
    model_manager: ModelManager,
    event_bus: SimpleEventBus,
) -> None:
    from voicetotext.bootstrap import build_first_run_settings_service
    from voicetotext.settings.window import SettingsWindow

    service = build_first_run_settings_service(
        config=config,
        event_bus=event_bus,
        model_manager=model_manager,
    )
    SettingsWindow(service=service, first_run=True).run()

    config.save()


def _show_error(message: str) -> None:
    try:
        from tkinter import messagebox
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("VoiceToText Error", message)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
