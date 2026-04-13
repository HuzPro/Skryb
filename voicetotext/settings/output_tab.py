import customtkinter as ctk

from voicetotext.settings.service import SettingsService
from voicetotext.settings.tk_cleanup import release_tk_variables


class OutputTab:
    """View for choosing how transcribed text is delivered to the focused window."""

    def __init__(self, parent: ctk.CTkFrame, service: SettingsService) -> None:
        self._service = service
        self._build(parent)

    def dispose(self) -> None:
        release_tk_variables(self._mode_var)

    def _build(self, parent: ctk.CTkFrame) -> None:
        draft = self._service.current_draft()

        mode_label = ctk.CTkLabel(parent, text="Output Mode")
        mode_label.pack(anchor="w", padx=16, pady=(18, 4))

        self._mode_var = ctk.StringVar(value=draft.output_mode)

        direct_paste_radio = ctk.CTkRadioButton(
            parent,
            text="Paste only (does not modify your clipboard)",
            variable=self._mode_var,
            value="direct_paste",
        )
        direct_paste_radio.pack(anchor="w", padx=24, pady=4)

        paste_and_clipboard_radio = ctk.CTkRadioButton(
            parent,
            text="Paste and copy to clipboard",
            variable=self._mode_var,
            value="paste_and_clipboard",
        )
        paste_and_clipboard_radio.pack(anchor="w", padx=24, pady=4)

        clipboard_radio = ctk.CTkRadioButton(
            parent,
            text="Clipboard only (no paste)",
            variable=self._mode_var,
            value="clipboard_only",
        )
        clipboard_radio.pack(anchor="w", padx=24, pady=4)

    def current_mode(self) -> str:
        return self._mode_var.get()
