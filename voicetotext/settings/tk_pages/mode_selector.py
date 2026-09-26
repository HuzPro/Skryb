import tkinter
from tkinter import ttk

from voicetotext.settings.signal import Signal
from voicetotext.settings.transcription_mode import TranscriptionMode


_MODE_HINTS: dict[TranscriptionMode, str] = {
    TranscriptionMode.BATCH: (
        "Records your dictation, then transcribes it when you stop. "
        "All batch models are available below."
    ),
    TranscriptionMode.STREAMING: (
        "Types text live while you speak (GPU only). "
        "The list below shows only streaming models."
    ),
}

_HINT_WRAP_PX = 480


class TkModeSelector(ttk.Frame):
    """Inline batch/streaming toggle shown directly above the model list.

    Fires :attr:`mode_changed` only on a *user* toggle (via the radio
    ``command``); programmatic :meth:`set_mode` is silent. A hint line
    under the buttons explains the active mode and why the model list is
    filtered.
    """

    def __init__(self, parent: tkinter.Misc) -> None:
        super().__init__(parent)
        self._var = tkinter.StringVar(
            master=self, value=TranscriptionMode.BATCH.value,
        )
        self._buttons: dict[TranscriptionMode, ttk.Radiobutton] = {}
        self._streaming_reason = ""
        self.mode_changed = Signal()
        self._build()
        self.set_mode(TranscriptionMode.BATCH)

    def _build(self) -> None:
        row = ttk.Frame(self)
        row.pack(fill="x")
        for mode in TranscriptionMode:
            button = ttk.Radiobutton(
                row,
                text=mode.display_label,
                value=mode.value,
                variable=self._var,
                command=self._on_user_toggle,
            )
            button.pack(side="left", padx=(0, 16))
            self._buttons[mode] = button

        self._hint = ttk.Label(
            self, text="", wraplength=_HINT_WRAP_PX, justify="left",
        )
        self._hint.pack(anchor="w", pady=(4, 0))

    def _on_user_toggle(self) -> None:
        self._refresh_hint()
        self.mode_changed.emit(self.current_mode())

    def current_mode(self) -> TranscriptionMode:
        return TranscriptionMode(self._var.get())

    def set_mode(self, mode: TranscriptionMode) -> None:
        self._var.set(mode.value)
        self._refresh_hint()

    def radio_button(self, mode: TranscriptionMode) -> ttk.Radiobutton:
        return self._buttons[mode]

    def disable_streaming(self, reason: str) -> None:
        """Disable the Streaming option (no compatible GPU) and record *reason*.

        Leaves Batch selectable. The enclosing page calls this when the
        detected hardware cannot run a streaming model.
        """
        self._streaming_reason = reason
        self._buttons[TranscriptionMode.STREAMING].state(["disabled"])

    def enable_streaming(self) -> None:
        self._streaming_reason = ""
        self._buttons[TranscriptionMode.STREAMING].state(["!disabled"])

    def is_streaming_enabled(self) -> bool:
        return "disabled" not in self._buttons[TranscriptionMode.STREAMING].state()

    def streaming_reason(self) -> str:
        return self._streaming_reason

    def hint_text(self) -> str:
        return self._hint.cget("text")

    def _refresh_hint(self) -> None:
        self._hint.configure(text=_MODE_HINTS[self.current_mode()])
