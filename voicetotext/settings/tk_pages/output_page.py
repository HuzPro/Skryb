import tkinter
from tkinter import ttk

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.theme import HEADING_STYLE


class TkOutputPage(ttk.Frame):
    """Page for choosing how transcribed text reaches the focused window.

    Renders one radio button per :class:`OutputMode`, all bound to a
    single ``StringVar`` holding the enum value, so the page maps a
    selection straight back to the canonical mode.
    """

    def __init__(self, parent: tkinter.Misc) -> None:
        super().__init__(parent)
        self._var = tkinter.StringVar(
            master=self, value=OutputMode.DIRECT_PASTE.value,
        )
        self._buttons: dict[OutputMode, ttk.Radiobutton] = {}
        self._build()
        self.set_mode(OutputMode.DIRECT_PASTE)

    def _build(self) -> None:
        heading = ttk.Label(self, text="Output Mode", style=HEADING_STYLE)
        heading.pack(anchor="w", pady=(0, 8))
        for mode in OutputMode:
            button = ttk.Radiobutton(
                self,
                text=mode.display_label,
                value=mode.value,
                variable=self._var,
            )
            button.pack(anchor="w", pady=2)
            self._buttons[mode] = button

    def current_mode(self) -> OutputMode:
        return OutputMode(self._var.get())

    def set_mode(self, mode: OutputMode) -> None:
        self._var.set(mode.value)

    def radio_buttons(self) -> list[ttk.Radiobutton]:
        return list(self._buttons.values())
