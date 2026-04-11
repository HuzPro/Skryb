import customtkinter as ctk

from voicetotext.settings.service import SettingsService
from voicetotext.settings.tk_cleanup import release_tk_variables


class HotkeyTab:
    """View for configuring the global activation hotkey and its mode."""

    def __init__(self, parent: ctk.CTkFrame, service: SettingsService):
        self._service = service
        self._build(parent)

    def dispose(self) -> None:
        release_tk_variables(self._mode_var)

    def _build(self, parent: ctk.CTkFrame) -> None:
        draft = self._service.current_draft()

        combo_label = ctk.CTkLabel(parent, text="Hotkey Combination")
        combo_label.pack(anchor="w", padx=16, pady=(18, 4))

        self._combo_entry = ctk.CTkEntry(parent, height=32)
        self._combo_entry.insert(0, draft.hotkey_combination)
        self._combo_entry.pack(fill="x", padx=16, pady=(0, 6))

        hint = ctk.CTkLabel(
            parent,
            text="Examples: ctrl+alt+m, ctrl+shift+space, f12",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        hint.pack(anchor="w", padx=16, pady=(0, 18))

        mode_label = ctk.CTkLabel(parent, text="Activation Mode")
        mode_label.pack(anchor="w", padx=16, pady=(4, 4))

        self._mode_var = ctk.StringVar(value=draft.hotkey_mode)

        toggle_radio = ctk.CTkRadioButton(
            parent,
            text="Toggle (press to start, press again to stop)",
            variable=self._mode_var,
            value="toggle",
        )
        toggle_radio.pack(anchor="w", padx=24, pady=4)

        ptt_radio = ctk.CTkRadioButton(
            parent,
            text="Push-to-talk (hold to record, release to stop)",
            variable=self._mode_var,
            value="push_to_talk",
        )
        ptt_radio.pack(anchor="w", padx=24, pady=4)

    def current_combination(self) -> str:
        return self._combo_entry.get().strip()

    def current_mode(self) -> str:
        return self._mode_var.get()
