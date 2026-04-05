import sys
import traceback

from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.core.paths import AppPaths
from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager


def main() -> None:
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

    if paths.is_first_run():
        _run_first_time_setup(config, paths)

    from voicetotext.bootstrap import build_app

    app = build_app(config, paths)
    app.run()


def _run_first_time_setup(config: JsonConfigAdapter, paths: AppPaths) -> None:
    from voicetotext.settings.window import SettingsWindow

    model_manager = HuggingFaceModelManager(paths.models_dir)

    SettingsWindow(
        config=config,
        model_manager=model_manager,
        first_run=True,
    ).run()

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
