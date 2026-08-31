from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from concurrent.futures import Future
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter
    from pathlib import Path

    from voicetotext.core.config import ConfigurationStore
    from voicetotext.core.event_bus import SimpleEventBus
    from voicetotext.core.json_config_adapter import JsonConfigAdapter
    from voicetotext.core.paths import AppPaths
    from voicetotext.transcription.engine_registry import EngineRegistry
    from voicetotext.transcription.engine_types import HardwareProfile
    from voicetotext.transcription.license_policy import LicensePolicy
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

    ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`` opts into the AOTriton
    flash / mem-efficient attention kernels on AMD ROCm. The flag is a
    no-op on NVIDIA / CPU, so it is set unconditionally; ``setdefault``
    lets a user explicitly opt out by exporting ``=0`` first.
    """
    if os.environ.get("VOICETOTEXT_DISABLE_CUDA_CACHE") == "1":
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def _apply_hf_cache_env_overrides(paths: AppPaths) -> None:
    """Point ``huggingface_hub``'s default cache at our models dir.

    Must run **before** any code imports ``huggingface_hub``, because
    its ``HF_HUB_CACHE`` constant is read once at module import. Our
    ``ChunkedFileDownloader`` writes to ``paths.models_dir``, but
    third-party libraries that use ``hf_hub_download()`` (notably
    ``nano_parakeet``) call it with no ``cache_dir`` argument and fall
    back to the global default - which would otherwise be
    ``~/.cache/huggingface/hub``. Pointing the global default at our
    models dir keeps a single source of truth for cached model files.

    Uses ``setdefault`` so a user-set ``HF_HUB_CACHE`` still wins.
    """
    os.environ.setdefault("HF_HUB_CACHE", str(paths.models_dir))


def _parse_cli(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voicetotext")
    maintenance = parser.add_mutually_exclusive_group()
    maintenance.add_argument("--doctor", action="store_true")
    maintenance.add_argument("--create-shortcut", action="store_true")
    maintenance.add_argument("--enable-autostart", action="store_true")
    maintenance.add_argument("--disable-autostart", action="store_true")
    return parser.parse_args(arguments)


def main() -> None:
    from voicetotext.core.startup_log import begin_startup_timing

    begin_startup_timing()
    options = _parse_cli()

    from voicetotext.core.paths import AppPaths

    paths = AppPaths()
    _apply_torch_env_overrides()
    _apply_hf_cache_env_overrides(paths)
    maintenance_code = _run_maintenance(options)
    if maintenance_code is not None:
        raise SystemExit(maintenance_code)

    from voicetotext.core.single_instance import SingleInstanceMutex

    mutex = SingleInstanceMutex()
    if not mutex.acquire():
        _show_instance_conflict()
        raise SystemExit(0)

    try:
        paths.ensure_dirs()
        from voicetotext.core.startup_log import initialize_session_log

        session_dir = initialize_session_log(
            paths.diagnostics_dir, console=_stderr_is_console(),
        )
        paths.activate_session(session_dir)
        from voicetotext.core.torch_prewarm import start_torch_prewarm

        prewarm = start_torch_prewarm()
        _run(paths, prewarm)
    except Exception:
        logging.getLogger(__name__).exception("Startup failed")
        _show_error(traceback.format_exc())
        raise SystemExit(1)
    finally:
        mutex.release()


def _run_maintenance(options: argparse.Namespace) -> int | None:
    if options.doctor:
        from voicetotext.diagnostics.doctor import run_doctor

        print(run_doctor())
        return 0
    if not (
        options.create_shortcut
        or options.enable_autostart
        or options.disable_autostart
    ):
        return None
    from voicetotext.core.dev_install import (
        create_shortcut,
        disable_autostart,
        enable_autostart,
    )

    if options.create_shortcut:
        succeeded, message = create_shortcut()
    elif options.enable_autostart:
        succeeded, message = enable_autostart()
    else:
        succeeded, message = disable_autostart()
    output = sys.stdout if succeeded else sys.stderr
    if output is not None:
        output.write(message + "\n")
    return 0 if succeeded else 1


def _consume_torch_prewarm(prewarm: Future[ModuleType]) -> None:
    logger = logging.getLogger(__name__)
    try:
        module = prewarm.result()
    except BaseException:
        logger.exception("Torch prewarm failed")
        return
    logger.info(
        "Torch prewarm complete (version=%s)",
        getattr(module, "__version__", "unknown"),
    )


def _run(paths: AppPaths, prewarm: Future[ModuleType]) -> None:
    import tkinter

    from voicetotext.core.event_bus import SimpleEventBus
    from voicetotext.core.json_config_adapter import JsonConfigAdapter
    from voicetotext.core.startup_log import milestone
    from voicetotext.downloads.download_manager import DownloadManager
    from voicetotext.transcription.engine_factory import (
        build_engine_registry,
        create_model_manager,
    )
    from voicetotext.transcription.license_policy import LicensePolicy

    root = tkinter.Tk()
    root.withdraw()
    milestone("Tk-created")

    config = JsonConfigAdapter(paths.config_path)
    model_was_configured = config.has_persisted_value(
        "transcription.model_size",
    )
    config.load()
    milestone("config-loaded")

    # The one licence policy for the whole process. First-run setup and
    # ``build_app`` each build their own model manager, so a second
    # instance here would give them separate acceptance state.
    license_policy = LicensePolicy(config)

    from voicetotext.transcription.model_info import (
        migrate_engine_field,
        migrate_removed_models,
    )

    migrate_removed_models(config)
    migrate_engine_field(config)

    event_bus = SimpleEventBus()
    download_manager = DownloadManager(event_bus=event_bus)
    model_manager = create_model_manager(
        paths.models_dir, download_manager, license_policy,
    )
    registry = build_engine_registry()

    _consume_torch_prewarm(prewarm)
    from voicetotext.transcription.torch_hardware_detector import (
        TorchHardwareDetector,
    )

    hardware = TorchHardwareDetector().detect()
    from voicetotext.transcription.model_info import DEFAULT_MODEL

    if not model_was_configured:
        config.set("transcription.model_size", DEFAULT_MODEL)

    if paths.is_first_run() or not _configured_model_installed(
        config, model_manager,
    ):
        _run_first_time_setup(
            config, model_manager, event_bus, root, registry, hardware,
            paths.dictionary_path, license_policy,
        )

    if not _configured_model_installed(config, model_manager):
        root.destroy()
        return

    from voicetotext.bootstrap import build_app

    app = build_app(
        config,
        paths,
        root,
        event_bus=event_bus,
        download_manager=download_manager,
        registry=registry,
        hardware=hardware,
        license_policy=license_policy,
    )
    app.run()


def _configured_model_installed(
    config: ConfigurationStore,
    model_manager: ModelManager,
) -> bool:
    from voicetotext.transcription.model_info import DEFAULT_MODEL

    model_name = config.get(
        "transcription.model_size",
        DEFAULT_MODEL,
    )
    return bool(model_manager.is_installed(model_name))


def _run_first_time_setup(
    config: JsonConfigAdapter,
    model_manager: ModelManager,
    event_bus: SimpleEventBus,
    root: tkinter.Misc,
    registry: EngineRegistry,
    hardware: HardwareProfile,
    dictionary_path: Path,
    license_policy: LicensePolicy,
) -> None:
    from voicetotext.bootstrap import build_first_run_settings_service
    from voicetotext.output.tk_gui_thread_dispatcher import TkGuiThreadDispatcher
    from voicetotext.settings.tk_window import TkSettingsWindow
    from voicetotext.settings.win32_capture_bridge import Win32CaptureBridge

    service = build_first_run_settings_service(
        config=config,
        event_bus=event_bus,
        model_manager=model_manager,
        registry=registry,
        hardware=hardware,
        dictionary_path=dictionary_path,
        license_policy=license_policy,
    )
    dispatcher = TkGuiThreadDispatcher(root)
    window = TkSettingsWindow(
        root,
        service,
        capture_bridge=Win32CaptureBridge(),
        first_run=True,
        schedule=dispatcher.dispatch,
    )
    window.show()
    root.wait_window(window)
    dispatcher.cancel()
    config.save()


def _show_instance_conflict() -> None:
    _try_show_message(
        "VoiceToText",
        "VoiceToText is already running in this Windows session.",
    )


def _stderr_is_console() -> bool:
    # pythonw (GUI launches, the Start Menu shortcut) runs with no stdio.
    return sys.stderr is not None and sys.stderr.isatty()


def _show_error(message: str) -> None:
    if sys.stderr is not None:
        sys.stderr.write(message)
    _try_show_error_dialog(message)


def _try_show_message(title: str, message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    except Exception:
        pass


def _try_show_error_dialog(message: str) -> None:
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("VoiceToText Error", message)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
