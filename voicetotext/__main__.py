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


def _apply_hf_cache_env_overrides(paths: AppPaths) -> None:
    """Point ``huggingface_hub``'s default cache at our models dir.

    Must run **before** any code imports ``huggingface_hub``, because
    its ``HF_HUB_CACHE`` constant is read once at module import. Our
    ``ChunkedFileDownloader`` writes to ``paths.models_dir``, but
    third-party libraries that use ``hf_hub_download()`` (notably
    ``nano_parakeet``) call it with no ``cache_dir`` argument and fall
    back to the global default — which would otherwise be
    ``~/.cache/huggingface/hub``. Pointing the global default at our
    models dir keeps a single source of truth for cached model files.

    Uses ``setdefault`` so a user-set ``HF_HUB_CACHE`` still wins.
    """
    os.environ.setdefault("HF_HUB_CACHE", str(paths.models_dir))


def main() -> None:
    _apply_torch_env_overrides()
    try:
        _run()
    except Exception:
        _show_error(traceback.format_exc())
        sys.exit(1)


def _run() -> None:
    import sys as _sys

    from PySide6.QtWidgets import QApplication

    paths = AppPaths()
    paths.ensure_dirs()
    _apply_hf_cache_env_overrides(paths)

    # Single QApplication for the whole process. The tray must outlive
    # any transient window (settings/first-run), so disable Qt's default
    # "quit when last window closes" behaviour.
    qapp = QApplication.instance() or QApplication(_sys.argv)
    qapp.setQuitOnLastWindowClosed(False)

    config = JsonConfigAdapter(paths.config_path)
    config.load()

    event_bus = SimpleEventBus()
    download_manager = DownloadManager(event_bus=event_bus)

    from voicetotext.transcription.engine_factory import create_model_manager
    model_manager = create_model_manager(paths.models_dir, download_manager)

    if paths.is_first_run() or not _configured_model_installed(config, model_manager):
        _run_first_time_setup(config, model_manager, event_bus)

    if not _configured_model_installed(config, model_manager):
        # User closed first-run without installing a model — nothing
        # to load, so bail cleanly before build_app tries the engine.
        return

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
    from PySide6.QtCore import QEventLoop

    from voicetotext.bootstrap import build_first_run_settings_service
    from voicetotext.settings.qt_window import QtSettingsWindow

    service = build_first_run_settings_service(
        config=config,
        event_bus=event_bus,
        model_manager=model_manager,
    )

    # Block until the first-run window closes — whether via Save (which
    # calls close()) or the OS close button. QEventLoop keeps the main
    # event pump running without calling QApplication.exec(), which the
    # tray uses later.
    loop = QEventLoop()
    window = QtSettingsWindow(service=service, first_run=True)
    window.destroyed.connect(loop.quit)
    window.show()
    loop.exec()

    config.save()


def _show_error(message: str) -> None:
    sys.stderr.write(message)
    _try_show_error_dialog(message)


def _try_show_error_dialog(message: str) -> None:
    try:
        import sys as _sys

        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(_sys.argv)
        QMessageBox.critical(None, "VoiceToText Error", message)
    except Exception:
        pass


if __name__ == "__main__":
    main()
