import tkinter
from tkinter import ttk

from voicetotext.settings.signal import Signal
from voicetotext.settings.theme import (
    CARD_MUTED_STYLE,
    CARD_PADDING,
    RADIO_TITLE_STYLE,
    card,
    set_card_selected,
)
from voicetotext.settings.transcription_mode import TranscriptionMode

_MODE_TITLES: dict[TranscriptionMode, str] = {
    TranscriptionMode.BATCH: "Batch",
    TranscriptionMode.STREAMING: "Live",
}
_MODE_HINTS: dict[TranscriptionMode, str] = {
    TranscriptionMode.BATCH: (
        "Records your dictation, then transcribes it when you stop. "
        "Works on any computer."
    ),
    TranscriptionMode.STREAMING: (
        "Types live while you speak (GPU only). "
        "Lists only streaming models."
    ),
}

_HINT_WRAP_PX = 240
_RADIO_INDENT = 22


class TkModeSelector(ttk.Frame):
    """Batch / Live as two cards side by side, above the model list.

    Fires :attr:`mode_changed` only on a *user* choice (the radio's
    ``command`` or a click on its card); programmatic :meth:`set_mode` is
    silent. Each card says what its mode does; the Live card says why it
    is off when there is no GPU for it.
    """

    def __init__(self, parent: tkinter.Misc) -> None:
        super().__init__(parent)
        self._var = tkinter.StringVar(
            master=self, value=TranscriptionMode.BATCH.value,
        )
        self._buttons: dict[TranscriptionMode, ttk.Radiobutton] = {}
        self._cards: dict[TranscriptionMode, ttk.Frame] = {}
        self._hints: dict[TranscriptionMode, ttk.Label] = {}
        self._streaming_reason = ""
        self.mode_changed = Signal()
        self._build()
        self.set_mode(TranscriptionMode.BATCH)

    def _build(self) -> None:
        self.columnconfigure((0, 1), weight=1, uniform="mode")
        for column, mode in enumerate(TranscriptionMode):
            mode_card = card(self, padding=CARD_PADDING)
            mode_card.grid(
                row=0, column=column, sticky="nsew", padx=(0, 12) if column == 0 else 0,
            )
            button = ttk.Radiobutton(
                mode_card,
                text=_MODE_TITLES[mode],
                value=mode.value,
                variable=self._var,
                command=self._on_user_toggle,
                style=RADIO_TITLE_STYLE,
            )
            button.pack(anchor="w")
            hint = ttk.Label(
                mode_card, text=_MODE_HINTS[mode], style=CARD_MUTED_STYLE,
                wraplength=_HINT_WRAP_PX, justify="left",
            )
            hint.pack(anchor="w", padx=(_RADIO_INDENT, 0), pady=(2, 0))
            for widget in (mode_card, hint):
                widget.bind("<Button-1>", lambda _event, b=button: self._click(b))
            self._buttons[mode] = button
            self._cards[mode] = mode_card
            self._hints[mode] = hint

    def _click(self, button: ttk.Radiobutton) -> None:
        if not button.instate(["disabled"]):
            button.invoke()

    def _on_user_toggle(self) -> None:
        self._refresh_selection()
        self.mode_changed.emit(self.current_mode())

    def current_mode(self) -> TranscriptionMode:
        return TranscriptionMode(self._var.get())

    def set_mode(self, mode: TranscriptionMode) -> None:
        self._var.set(mode.value)
        self._refresh_selection()

    def radio_button(self, mode: TranscriptionMode) -> ttk.Radiobutton:
        return self._buttons[mode]

    def disable_streaming(self, reason: str) -> None:
        """Disable the Live option (no compatible GPU) and show *reason* on it.

        Leaves Batch selectable. The enclosing page calls this when the
        detected hardware cannot run a streaming model.
        """
        self._streaming_reason = reason
        self._buttons[TranscriptionMode.STREAMING].state(["disabled"])
        self._hints[TranscriptionMode.STREAMING].configure(text=reason)

    def enable_streaming(self) -> None:
        self._streaming_reason = ""
        self._buttons[TranscriptionMode.STREAMING].state(["!disabled"])
        self._hints[TranscriptionMode.STREAMING].configure(
            text=_MODE_HINTS[TranscriptionMode.STREAMING],
        )

    def is_streaming_enabled(self) -> bool:
        return "disabled" not in self._buttons[TranscriptionMode.STREAMING].state()

    def streaming_reason(self) -> str:
        return self._streaming_reason

    def hint_text(self) -> str:
        return _MODE_HINTS[self.current_mode()]

    def _refresh_selection(self) -> None:
        for mode, mode_card in self._cards.items():
            set_card_selected(mode_card, mode is self.current_mode())
