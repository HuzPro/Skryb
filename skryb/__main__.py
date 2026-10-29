from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from concurrent.futures import Future
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter
    from collections.abc import Callable
    from pathlib import Path

    from skryb.core.config import ConfigurationStore
    from skryb.core.event_bus import SimpleEventBus
    from skryb.core.json_config_adapter import JsonConfigAdapter
    from skryb.core.paths import AppPaths
    from skryb.core.single_instance import SingleInstanceMutex
    from skryb.transcription.engine_preload import EnginePreload
    from skryb.transcription.engine_registry import EngineRegistry
    from skryb.transcription.engine_types import HardwareProfile
    from skryb.transcription.hardware_detector import HardwareDetector
    from skryb.transcription.license_policy import LicensePolicy
    from skryb.transcription.model_manager import ModelManager
    from skryb.ui.api_server import ApiServer
    from skryb.ui.launcher import ShellLauncher

_MB_ICONERROR = 0x10
_MB_SETFOREGROUND = 0x10000
_MB_TOPMOST = 0x40000
SETUP_NOT_FINISHED = (
    "Setup was closed before a speech model was installed, so Skryb will close. "
    "Start Skryb again to finish setting it up."
)


def _apply_torch_env_overrides() -> None:
    """Translate Skryb env vars into PyTorch env vars.

    Must run **before** any code imports ``torch``, because PyTorch
    reads its allocator configuration once at import time. We do not
    import torch anywhere above this point in the startup chain.

    ``SKRYB_DISABLE_CUDA_CACHE=1`` is the escape hatch for the
    model-hot-swap VRAM leak: it sets
    ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` which disables PyTorch's
    caching allocator entirely. Tensor deallocations then return memory
    to the driver immediately, at a notable perf cost. Use only if the
    normal teardown path fails to release VRAM between model swaps.

    ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`` opts into the AOTriton
    flash / mem-efficient attention kernels on AMD ROCm. The flag is a
    no-op on NVIDIA / CPU, so it is set unconditionally; ``setdefault``
    lets a user explicitly opt out by exporting ``=0`` first.

    ``ROCM_SDK_TARGET_FAMILY`` is answered here so no torch import ever
    runs ROCm's own ``offload-arch`` probe, which flashes a console window
    under the windowless launcher.
    """
    if os.environ.get("SKRYB_DISABLE_CUDA_CACHE") == "1":
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    from skryb.core.rocm_env import prepare_rocm_environment

    prepare_rocm_environment()


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
    parser = argparse.ArgumentParser(prog="skryb")
    maintenance = parser.add_mutually_exclusive_group()
    maintenance.add_argument("--doctor", action="store_true")
    maintenance.add_argument("--create-shortcut", action="store_true")
    maintenance.add_argument("--enable-autostart", action="store_true")
    maintenance.add_argument("--disable-autostart", action="store_true")
    # Passed by a copy restarting onto newer code to the copy replacing it.
    parser.add_argument("--restart", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(arguments)


def main() -> None:
    from skryb.core.startup_log import begin_startup_timing

    begin_startup_timing()
    _start_library_prewarm()
    options = _parse_cli()
    _move_settings_from_voicetotext()

    from skryb.core.paths import AppPaths

    paths = AppPaths()
    _apply_torch_env_overrides()
    _apply_hf_cache_env_overrides(paths)
    maintenance_code = _run_maintenance(options)
    if maintenance_code is not None:
        raise SystemExit(maintenance_code)

    from skryb.core.single_instance import SingleInstanceMutex

    mutex = SingleInstanceMutex()
    if not _take_instance_lock(mutex, paths, restarting=options.restart):
        raise SystemExit(0)

    try:
        paths.load_configured_diagnostics_root()
        paths.ensure_dirs()
        from skryb.core.startup_log import initialize_session_log

        session_dir = initialize_session_log(
            paths.diagnostics_dir, console=_stderr_is_console(),
        )
        paths.activate_session(session_dir)
        startup = _prepare_startup(paths)
        _run(paths, startup, restarted=options.restart)
    except Exception:
        logging.getLogger(__name__).exception("Startup failed")
        _show_error(traceback.format_exc())
        raise SystemExit(1)
    finally:
        mutex.release()


def _move_settings_from_voicetotext() -> None:
    from skryb.core.paths import move_voicetotext_folders

    try:
        move_voicetotext_folders()
    except OSError as error:
        _refuse_to_start_without_old_settings(error)


def _refuse_to_start_without_old_settings(error: OSError) -> None:
    _try_show_message(
        "Skryb",
        f"Skryb couldn't move your settings and models over from VoiceToText "
        f"({error}). Close VoiceToText if it's running, then start Skryb again.",
    )
    raise SystemExit(1) from error


# Third-party libraries on the path to the tray, the one needed soonest
# first. Only libraries belong here: they never import skryb, so a
# second thread importing them cannot deadlock against the main thread's
# imports (Python 3.12 raises that as an error). sounddevice stays out
# because importing it starts PortAudio on the importing thread.
_PREWARM_LIBRARIES = (
    "numpy",
    "tkinter",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "jellyfish",
    "text_to_num",
)


def _start_library_prewarm() -> None:
    """Import the tray path's libraries on a second thread.

    After a Defender definitions update or a reboot, every file the app
    opens is scanned first, which turns these imports into seconds. Started
    here, they load while the main thread parses arguments, takes the
    instance lock, opens the log and reads the config, and the main
    thread's own imports find them done.
    """
    import threading

    threading.Thread(
        target=_import_libraries,
        args=(_PREWARM_LIBRARIES,),
        daemon=True,
        name="library-prewarm",
    ).start()


def _import_libraries(names: tuple[str, ...]) -> None:
    import contextlib
    import importlib

    for name in names:
        # The main thread's own import of it raises properly, once
        # logging exists to record it.
        with contextlib.suppress(ImportError, OSError):
            importlib.import_module(name)


# How long a restarting copy waits for the one it replaces to clean up and exit.
_REPLACE_WAIT_SECONDS = 30.0
# A copy still starting answers only once its tray is built, so this covers
# a slow cold start.
_ANSWER_WAIT_SECONDS = 20.0


def _take_instance_lock(
    mutex: SingleInstanceMutex, paths: AppPaths, *, restarting: bool,
) -> bool:
    """Hold the one-copy lock, or hand this launch to the copy already running.

    The copy running answers by checking for newer code. One that never
    answers is stuck, so it is ended and this launch takes its place.
    """
    from skryb.core.single_instance import LaunchSignal, RunningCopy

    running_copy = RunningCopy(paths.config_path.with_name("running.pid"))
    if mutex.acquire(wait_seconds=_REPLACE_WAIT_SECONDS if restarting else 0.0):
        running_copy.record()
        return True
    if LaunchSignal().ask(_ANSWER_WAIT_SECONDS):
        return False
    # A copy that can't be ended may still have exited on its own meanwhile.
    ended = running_copy.end()
    if mutex.acquire(wait_seconds=_REPLACE_WAIT_SECONDS if ended else 0.0):
        running_copy.record()
        return True
    _show_instance_conflict()
    return False


def _run_maintenance(options: argparse.Namespace) -> int | None:
    if options.doctor:
        from skryb.diagnostics.doctor import run_doctor

        print(run_doctor())
        return 0
    if not (
        options.create_shortcut
        or options.enable_autostart
        or options.disable_autostart
    ):
        return None
    from skryb.core.dev_install import (
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


@dataclass(frozen=True)
class _Startup:
    """What ``main`` settles before the GUI thread has anything to build."""

    config: JsonConfigAdapter
    model_was_configured: bool
    registry: EngineRegistry
    await_torch_prewarm: Callable[[], None]


def _prepare_startup(paths: AppPaths) -> _Startup:
    """Load the config and decide about torch before the GUI work starts.

    The prewarm wants to start as early as possible so the import
    overlaps everything after it, but only the config can say whether
    this launch needs torch at all. So the config and the engine
    registry come first: together they cost about 0.2s and both are
    needed moments later regardless.
    """
    from skryb.core.json_config_adapter import JsonConfigAdapter
    from skryb.core.startup_log import milestone
    from skryb.transcription.engine_factory import build_engine_registry
    from skryb.transcription.model_info import (
        migrate_engine_field,
        migrate_removed_models,
    )

    config = JsonConfigAdapter(paths.config_path)
    model_was_configured = config.has_persisted_value(
        "transcription.model_size",
    )
    config.load()
    milestone("config-loaded")
    migrate_removed_models(config)
    migrate_engine_field(config)
    registry = build_engine_registry()
    return _Startup(
        config=config,
        model_was_configured=model_was_configured,
        registry=registry,
        await_torch_prewarm=_start_torch_prewarm_if_needed(
            config, registry, paths.models_dir,
        ),
    )


def _start_torch_prewarm_if_needed(
    config: ConfigurationStore,
    registry: EngineRegistry,
    models_dir: Path,
) -> Callable[[], None]:
    """Warm torch on a background thread, unless nothing will use it.

    Returns the call that waits for that import to finish. A launch
    whose engine does not use torch waits for nothing and never loads
    torch at all - that is the whole saving. The engine descriptor
    declares the dependency, because importing the engine to ask it
    would cost the very second this is protecting.
    """
    from skryb.transcription.engine_factory import startup_requires_torch

    if not startup_requires_torch(config, registry, models_dir, _remembered_hardware(config)):
        logging.getLogger(__name__).info(
            "Torch prewarm skipped: no engine this launch can select needs it",
        )
        return _nothing_to_await

    from skryb.core.torch_prewarm import start_torch_prewarm

    prewarm = start_torch_prewarm()
    return lambda: _consume_torch_prewarm(prewarm)


def _nothing_to_await() -> None:
    return None


def _remembered_hardware(config: ConfigurationStore) -> HardwareProfile | None:
    """The GPU probe a past launch saved, read without importing torch."""
    from skryb.transcription.cached_hardware_detector import CachedHardwareDetector
    from skryb.transcription.torch_hardware_detector import TorchHardwareDetector
    from skryb.transcription.win32_display_adapters import display_adapter_fingerprint

    return CachedHardwareDetector(TorchHardwareDetector(), config, display_adapter_fingerprint).peek()


def _run(paths: AppPaths, startup: _Startup, *, restarted: bool = False) -> None:
    import tkinter

    from skryb.core.event_bus import SimpleEventBus
    from skryb.core.startup_log import milestone
    from skryb.downloads.download_manager import DownloadManager
    from skryb.transcription.engine_factory import create_model_manager
    from skryb.transcription.license_policy import LicensePolicy

    config = startup.config
    registry = startup.registry

    # The one licence policy for the whole process. First-run setup and
    # ``build_app`` each build their own model manager, so a second
    # instance here would give them separate acceptance state.
    license_policy = LicensePolicy(config)

    event_bus = SimpleEventBus()
    # Before setup, which is served through it: one socket bind and an idle
    # thread until the shell connects.
    api_server = _start_api_server(event_bus)
    download_manager = DownloadManager(event_bus=event_bus)
    model_manager = create_model_manager(
        paths.models_dir, download_manager, license_policy,
    )

    startup.await_torch_prewarm()
    from skryb.transcription.cached_hardware_detector import (
        CachedHardwareDetector,
    )
    from skryb.transcription.torch_hardware_detector import (
        TorchHardwareDetector,
    )
    from skryb.transcription.win32_display_adapters import (
        display_adapter_fingerprint,
    )

    # Held, not called: the GPU probe imports torch, so only the code
    # that actually needs an answer gets to pay for one.
    hardware_detector = CachedHardwareDetector(
        TorchHardwareDetector(), config, display_adapter_fingerprint,
    )
    from skryb.transcription.model_info import default_model_for

    if not startup.model_was_configured:
        config.set(
            "transcription.model_size",
            default_model_for(hardware_detector.detect().kind),
        )

    needs_setup = paths.is_first_run() or not _configured_model_installed(
        config, model_manager,
    )
    # Before the window and the tray: the model takes longer to load than
    # both take to build, so it starts first and they are built meanwhile.
    preload = None if needs_setup else _start_engine_preload(
        config, paths, registry, hardware_detector, license_policy,
    )

    root = tkinter.Tk()
    root.withdraw()
    milestone("Tk-created")
    from skryb.core.single_instance import LaunchSignal, SecondLaunches

    second_launches = SecondLaunches(root, LaunchSignal())

    if needs_setup:
        from skryb.ui.launcher import ShellLauncher

        _run_first_time_setup(
            config, model_manager, event_bus, root, registry,
            hardware_detector, paths.dictionary_path, license_policy,
            api_server=api_server,
            launcher=ShellLauncher(api_server, notify=_show_setup_problem),
            diagnostics_dir=paths.diagnostics_dir,
        )

    if not _configured_model_installed(config, model_manager):
        root.destroy()
        return

    from skryb.bootstrap import build_app

    app = build_app(
        config,
        paths,
        root,
        event_bus=event_bus,
        download_manager=download_manager,
        registry=registry,
        hardware_detector=hardware_detector,
        license_policy=license_policy,
        api_server=api_server,
        preload=preload,
        second_launches=second_launches,
        # The copy it replaced has just checked.
        check_for_updates=not restarted,
    )
    app.run()


def _start_engine_preload(
    config: ConfigurationStore,
    paths: AppPaths,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    license_policy: LicensePolicy,
) -> EnginePreload | None:
    """Start loading the batch engine, or return None to leave it to the tray.

    Only the torch-free CTranslate2 engine is preloaded. Its modules are
    imported here, on the main thread, so the preload thread imports no
    skryb module: two threads importing the same modules can
    deadlock, which Python 3.12 raises as an error. A torch engine would
    import torch to pick its device, and its adapter imports more of the
    app, so it keeps loading after the tray as before.

    Also skipped where ``build_app`` would change what gets loaded first:
    streaming builds its engine elsewhere, and a legacy ``None``/``"auto"``
    language is rewritten by ``migrate_language`` before the load.
    """
    from skryb.transcription.engine_factory import (
        create_engine_for_active_model,
        startup_requires_torch,
    )

    if config.get("transcription.mode", "batch") == "streaming":
        return None
    if config.get("transcription.language") in (None, "auto"):
        return None
    if startup_requires_torch(config, registry, paths.models_dir, hardware_detector.peek()):
        return None
    import importlib

    for module in (
        "skryb.transcription.crisperwhisper_ct2_adapter",
        "skryb.transcription.ct2_conversion",
    ):
        importlib.import_module(module)
    from skryb.transcription.engine_preload import EnginePreload

    return EnginePreload.start(
        lambda: create_engine_for_active_model(
            config, paths, registry, hardware_detector,
            license_policy=license_policy,
        ),
    )


def _start_api_server(event_bus: SimpleEventBus) -> ApiServer:
    from skryb.ui.api_server import ApiServer
    from skryb.ui.event_hub import EventHub

    server = ApiServer(EventHub(event_bus))
    server.start()
    return server


def _configured_model_installed(
    config: ConfigurationStore,
    model_manager: ModelManager,
) -> bool:
    from skryb.transcription.model_info import DEFAULT_MODEL

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
    hardware_detector: HardwareDetector,
    dictionary_path: Path,
    license_policy: LicensePolicy,
    *,
    api_server: ApiServer,
    launcher: ShellLauncher,
    diagnostics_dir: Path,
) -> None:
    """Serve setup to the shell and wait, pumping Tk, until it is done or closed.

    A shell that cannot be opened has already told the user why, so this
    returns at once and the caller finds no model installed.
    """
    import functools
    import threading

    from skryb.bootstrap import build_first_run_settings_service
    from skryb.recording.input_devices import Microphones
    from skryb.hotkeys.win32_copilot_key import copilot_key_acts_as_right_ctrl
    from skryb.settings.win32_capture_bridge import Win32CaptureBridge
    from skryb.ui.api_handlers import CoreApi, ShortcutCapture
    from skryb.ui.launcher import wait_for_setup

    service = build_first_run_settings_service(
        config=config,
        event_bus=event_bus,
        model_manager=model_manager,
        registry=registry,
        hardware_detector=hardware_detector,
        dictionary_path=dictionary_path,
        license_policy=license_policy,
    )
    setup_done = threading.Event()
    CoreApi(
        service,
        api_server.hub,
        microphones=Microphones(),
        capture=ShortcutCapture(
            Win32CaptureBridge(), functools.partial(api_server.hub.publish, "capture"),
            copilot_acts_as_right_ctrl=copilot_key_acts_as_right_ctrl,
        ),
        diagnostics_dir=diagnostics_dir,
        first_run=True,
        on_setup_done=setup_done.set,
    ).serve_on(api_server)
    if launcher.open("setup"):
        wait_for_setup(root, lambda: setup_done.is_set() or not launcher.is_running())
        if not _configured_model_installed(config, model_manager):
            _show_setup_problem(SETUP_NOT_FINISHED)
    config.save()


def _show_setup_problem(message: str) -> None:
    # No tray icon exists yet to show a balloon on.
    _try_show_message("Skryb", message)


def _show_instance_conflict() -> None:
    _try_show_message(
        "Skryb",
        "Skryb is already running but not responding, and it couldn't "
        "be closed. End it in Task Manager, then start Skryb again.",
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

        # No window of ours is in front to own it, so it asks to be on top.
        ctypes.windll.user32.MessageBoxW(
            None, message, title, _MB_ICONERROR | _MB_SETFOREGROUND | _MB_TOPMOST,
        )
    except Exception:
        pass


def _try_show_error_dialog(message: str) -> None:
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Skryb Error", message)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
