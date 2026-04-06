import sys
from typing import Callable

import customtkinter as ctk
from tkinter import messagebox

from voicetotext.core.config import ConfigurationStore
from voicetotext.settings.hotkeys_tab import HotkeyTab
from voicetotext.settings.output_tab import OutputTab
from voicetotext.settings.transcription_tab import TranscriptionTab
from voicetotext.transcription.model_manager import ModelManager


class SettingsWindow:

    def __init__(
        self,
        config: ConfigurationStore,
        model_manager: ModelManager,
        first_run: bool = False,
        on_save: Callable[[dict], None] | None = None,
        models_dir=None,
    ):
        self._config = config
        self._model_manager = model_manager
        self._first_run = first_run
        self._on_save = on_save
        self._models_dir = models_dir
        self._window: ctk.CTk | None = None
        self._original_values: dict = {}
        self._warning_label: ctk.CTkLabel | None = None

    def run(self) -> None:
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self._window = ctk.CTk()
        title = "VoiceToText Setup" if self._first_run else "VoiceToText Settings"
        self._window.title(title)
        self._window.geometry("520x580")
        self._window.resizable(False, False)

        if self._first_run:
            self._window.protocol("WM_DELETE_WINDOW", self._on_close_first_run)
        else:
            self._window.protocol("WM_DELETE_WINDOW", self._on_close)

        self._snapshot_original_values()
        self._build_ui()
        self._window.mainloop()

    def _build_ui(self) -> None:
        if self._first_run:
            banner = ctk.CTkLabel(
                self._window,
                text="Welcome! Download a model to get started.\n'base' is recommended.",
                font=ctk.CTkFont(size=14),
                wraplength=460,
            )
            banner.pack(pady=(15, 5), padx=20)

        tabview = ctk.CTkTabview(self._window)
        tabview.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        tab_transcription = tabview.add("Transcription")
        tab_hotkeys = tabview.add("Hotkeys")
        tab_output = tabview.add("Output")

        self._transcription_tab = TranscriptionTab(
            tab_transcription, self._config, self._model_manager,
            models_dir=self._models_dir,
        )
        self._hotkey_tab = HotkeyTab(tab_hotkeys, self._config)
        self._output_tab = OutputTab(tab_output, self._config)

        self._warning_label = ctk.CTkLabel(
            self._window,
            text="",
            text_color="#FF4444",
            font=ctk.CTkFont(size=12),
        )
        self._warning_label.pack(pady=(0, 2), padx=15)

        button_text = "Save & Start" if self._first_run else "Save & Apply"
        save_button = ctk.CTkButton(
            self._window,
            text=button_text,
            command=self._on_save_clicked,
            height=36,
            font=ctk.CTkFont(size=14),
        )
        save_button.pack(pady=(0, 15), padx=15)

    def _has_any_model(self) -> bool:
        return any(
            self._model_manager.is_installed(m.name)
            for m in self._model_manager.list_models()
        )

    def _snapshot_original_values(self) -> None:
        self._original_values = {
            "model_size": self._config.get("transcription.model_size"),
            "device": self._config.get("transcription.device"),
            "engine": self._config.get("transcription.engine"),
            "hotkey_combination": self._config.get("hotkey.combination"),
            "hotkey_mode": self._config.get("hotkey.mode"),
            "output_mode": self._config.get("output.mode"),
        }

    def _compute_changes(self) -> dict:
        return {
            "model": (
                self._config.get("transcription.model_size")
                != self._original_values["model_size"]
                or self._config.get("transcription.device")
                != self._original_values["device"]
            ),
            "engine": (
                self._config.get("transcription.engine")
                != self._original_values["engine"]
            ),
            "hotkey": (
                self._config.get("hotkey.combination")
                != self._original_values["hotkey_combination"]
                or self._config.get("hotkey.mode")
                != self._original_values["hotkey_mode"]
            ),
            "output": (
                self._config.get("output.mode")
                != self._original_values["output_mode"]
            ),
        }

    def _on_save_clicked(self) -> None:
        self._hotkey_tab.apply()
        self._transcription_tab.apply()
        self._output_tab.apply()
        self._config.save()

        if self._first_run and not self._has_any_model():
            self._warning_label.configure(
                text="Please download at least one model before starting."
            )
            return

        changes = self._compute_changes()

        if self._on_save:
            self._on_save(changes)

        self._window.destroy()

    def _on_close(self) -> None:
        self._window.destroy()

    def _on_close_first_run(self) -> None:
        if self._has_any_model():
            self._hotkey_tab.apply()
            self._transcription_tab.apply()
            self._output_tab.apply()
            self._config.save()
            self._window.destroy()
        else:
            if messagebox.askyesno(
                "Quit?",
                "No model downloaded. The app needs at least one model to work.\n\nQuit anyway?",
            ):
                self._window.destroy()
                sys.exit(0)
