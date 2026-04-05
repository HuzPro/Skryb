import customtkinter as ctk

from voicetotext.core.config import ConfigurationStore


class OutputTab:

    def __init__(self, parent: ctk.CTkFrame, config: ConfigurationStore):
        self._config = config
        self._build(parent)

    def _build(self, parent: ctk.CTkFrame) -> None:
        mode_label = ctk.CTkLabel(parent, text="Output Mode:")
        mode_label.pack(anchor="w", padx=10, pady=(15, 2))

        current_mode = self._config.get("output.mode", "auto_paste")
        self._mode_var = ctk.StringVar(value=current_mode)

        paste_radio = ctk.CTkRadioButton(
            parent,
            text="Auto-paste (copy to clipboard and simulate Ctrl+V)",
            variable=self._mode_var,
            value="auto_paste",
        )
        paste_radio.pack(anchor="w", padx=20, pady=3)

        clipboard_radio = ctk.CTkRadioButton(
            parent,
            text="Clipboard only (copy to clipboard, no paste)",
            variable=self._mode_var,
            value="clipboard_only",
        )
        clipboard_radio.pack(anchor="w", padx=20, pady=3)

    def apply(self) -> None:
        self._config.set("output.mode", self._mode_var.get())
