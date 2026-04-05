import customtkinter as ctk

from voicetotext.core.config import ConfigurationStore


class HotkeyTab:

    def __init__(self, parent: ctk.CTkFrame, config: ConfigurationStore):
        self._config = config
        self._build(parent)

    def _build(self, parent: ctk.CTkFrame) -> None:
        combo_label = ctk.CTkLabel(parent, text="Hotkey Combination:")
        combo_label.pack(anchor="w", padx=10, pady=(15, 2))

        self._combo_entry = ctk.CTkEntry(parent, width=300)
        self._combo_entry.insert(0, self._config.get("hotkey.combination", "ctrl+alt+m"))
        self._combo_entry.pack(anchor="w", padx=10, pady=(0, 10))

        hint = ctk.CTkLabel(
            parent,
            text="Examples: ctrl+alt+m, ctrl+shift+space, f12",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        hint.pack(anchor="w", padx=10, pady=(0, 15))

        mode_label = ctk.CTkLabel(parent, text="Hotkey Mode:")
        mode_label.pack(anchor="w", padx=10, pady=(5, 2))

        current_mode = self._config.get("hotkey.mode", "toggle")
        self._mode_var = ctk.StringVar(value=current_mode)

        toggle_radio = ctk.CTkRadioButton(
            parent,
            text="Toggle (press to start, press again to stop)",
            variable=self._mode_var,
            value="toggle",
        )
        toggle_radio.pack(anchor="w", padx=20, pady=3)

        ptt_radio = ctk.CTkRadioButton(
            parent,
            text="Push-to-talk (hold to record, release to stop)",
            variable=self._mode_var,
            value="push_to_talk",
        )
        ptt_radio.pack(anchor="w", padx=20, pady=3)

    def apply(self) -> None:
        self._config.set("hotkey.combination", self._combo_entry.get().strip())
        self._config.set("hotkey.mode", self._mode_var.get())
