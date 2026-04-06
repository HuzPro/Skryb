import sys
import traceback

from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.core.paths import AppPaths


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
    from voicetotext.transcription.engine_factory import (
        create_model_manager,
        resolve_engine_name,
    )

    engine_name = resolve_engine_name(config)
    model_manager = create_model_manager(engine_name, paths.models_dir)

    SettingsWindow(
        config=config,
        model_manager=model_manager,
        first_run=True,
        models_dir=paths.models_dir,
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
