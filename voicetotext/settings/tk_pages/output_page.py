import tkinter
from collections.abc import Callable
from tkinter import ttk

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.theme import HEADING_STYLE


def _ignore() -> None:
    return None


class TkOutputPage(ttk.Frame):
    """Page for choosing how transcribed text reaches the focused window.

    Renders one radio button per :class:`OutputMode`, all bound to a
    single ``StringVar`` holding the enum value, so the page maps a
    selection straight back to the canonical mode.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        on_reset_pill_position: Callable[[], None] = _ignore,
    ) -> None:
        super().__init__(parent)
        self._on_reset_pill_position = on_reset_pill_position
        self._var = tkinter.StringVar(
            master=self, value=OutputMode.DIRECT_PASTE.value,
        )
        self._buttons: dict[OutputMode, ttk.Radiobutton] = {}
        self._capture_content = tkinter.BooleanVar(master=self, value=False)
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

        ttk.Label(self, text="Privacy", style=HEADING_STYLE).pack(
            anchor="w", pady=(24, 8),
        )
        ttk.Checkbutton(
            self, text="Save recordings and transcripts for diagnostics",
            variable=self._capture_content,
        ).pack(anchor="w")
        ttk.Label(
            self,
            text=(
                "Off by default. Saves speech and text in the logs folder.\n"
                "Applies when you save. Existing recordings are kept.\n"
                "Use a local folder that is not synced or shared."
            ),
            wraplength=560,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(self, text="Recording pill", style=HEADING_STYLE).pack(
            anchor="w", pady=(24, 8),
        )
        ttk.Label(
            self,
            text="Drag the pill to move it; it stays there for the next dictation.",
            wraplength=560,
        ).pack(anchor="w")
        self._reset_pill_button = ttk.Button(
            self, text="Reset pill position", command=self._reset_pill_position,
        )
        self._reset_pill_button.pack(anchor="w", pady=(8, 0))
        self._pill_status = ttk.Label(self, text="")
        self._pill_status.pack(anchor="w", pady=(4, 0))

    def _reset_pill_position(self) -> None:
        self._on_reset_pill_position()
        self._pill_status.configure(text="The pill will appear at the bottom centre.")

    def captures_content(self) -> bool:
        return self._capture_content.get()

    def set_capture_content(self, enabled: bool) -> None:
        self._capture_content.set(enabled)

    def current_mode(self) -> OutputMode:
        return OutputMode(self._var.get())

    def set_mode(self, mode: OutputMode) -> None:
        self._var.set(mode.value)

    def radio_buttons(self) -> list[ttk.Radiobutton]:
        return list(self._buttons.values())
