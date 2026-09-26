import tkinter
from collections.abc import Callable
from tkinter import ttk

from voicetotext.output.output_mode import OutputMode
from voicetotext.output.paste_method import PasteMethod, normalized_process_name
from voicetotext.settings.theme import (
    BODY_FONT,
    CARD_BODY_STYLE,
    CARD_MUTED_STYLE,
    CARD_PADDING,
    CONTROL_GAP,
    RADIO_TITLE_STYLE,
    RowGroup,
    card,
    combobox,
    page_note,
    section_heading,
    set_card_selected,
)

_PASTE_HINT = (
    "For apps that do not take Ctrl+V: type the app's .exe name, pick how to paste "
    "into it and click Add. mintty gets Shift+Insert automatically."
)
_MODE_CARDS: dict[OutputMode, tuple[str, str]] = {
    OutputMode.DIRECT_PASTE: (
        "Paste only", "Types into the app and leaves your clipboard as it was.",
    ),
    OutputMode.PASTE_AND_CLIPBOARD: (
        "Paste and copy", "Types into the app and keeps the text on the clipboard.",
    ),
    OutputMode.CLIPBOARD_ONLY: (
        "Clipboard only", "Copies the text; you paste it where you want.",
    ),
}
_MODE_WRAP = 140
_RADIO_INDENT = 22
_PILL_HINT = "Drag the pill to move it; it stays there for the next dictation."
_PILL_RESET = "Reset. The pill will appear at the bottom centre next time."


def _ignore() -> None:
    return None


class TkOutputPage(ttk.Frame):
    """Page for how transcribed text reaches the focused window.

    One radio per :class:`OutputMode`, each on its own card, all bound to
    a single ``StringVar`` holding the enum value, so the page maps a
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
        self._cards: dict[OutputMode, ttk.Frame] = {}
        self._overrides: dict[str, PasteMethod] = {}
        self._new_process = tkinter.StringVar(master=self)
        self._new_method = tkinter.StringVar(
            master=self, value=PasteMethod.CTRL_V.display_label,
        )
        self._build()
        self.set_mode(OutputMode.DIRECT_PASTE)

    def _build(self) -> None:
        section_heading(self, "Paste", first=True)
        modes = ttk.Frame(self)
        modes.pack(fill="x")
        modes.columnconfigure(tuple(range(len(OutputMode))), weight=1, uniform="mode")
        for column, mode in enumerate(OutputMode):
            self._mode_card(modes, column, mode)
        self._build_paste_methods()
        self._build_pill()

    def _mode_card(self, parent: ttk.Frame, column: int, mode: OutputMode) -> None:
        title, description = _MODE_CARDS[mode]
        mode_card = card(parent, padding=CARD_PADDING)
        mode_card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 10, 0))
        button = ttk.Radiobutton(
            mode_card, text=title, value=mode.value, variable=self._var,
            command=self._refresh_selection, style=RADIO_TITLE_STYLE,
        )
        button.pack(anchor="w")
        hint = ttk.Label(
            mode_card, text=description, style=CARD_MUTED_STYLE,
            wraplength=_MODE_WRAP, justify="left",
        )
        hint.pack(anchor="w", padx=(_RADIO_INDENT, 0), pady=(2, 0))
        for widget in (mode_card, hint):
            widget.bind("<Button-1>", lambda _event: button.invoke())
        self._buttons[mode] = button
        self._cards[mode] = mode_card

    def _build_pill(self) -> None:
        section_heading(self, "Recording pill")
        rows = RowGroup(self)
        rows.pack(fill="x")
        row = rows.add(
            "Pill position", _PILL_HINT,
            lambda parent: ttk.Button(
                parent, text="Reset position", command=self._reset_pill_position,
            ),
        )
        assert isinstance(row.control, ttk.Button)
        self._reset_pill_button = row.control
        self._pill_status = row.description

    def _build_paste_methods(self) -> None:
        section_heading(self, "Paste method per app")
        page_note(self, _PASTE_HINT).pack(anchor="w", pady=(0, CONTROL_GAP))
        table = card(self, padding=(1, 1, 1, CARD_PADDING))
        table.pack(fill="x")
        self._paste_table = ttk.Treeview(
            table, columns=("app", "method"), show="headings", height=3,
            selectmode="browse",
        )
        self._paste_table.heading("app", text="App", anchor="w")
        self._paste_table.heading("method", text="Paste method", anchor="w")
        self._paste_table.column("app", width=260, anchor="w")
        self._paste_table.column("method", width=160, anchor="w")
        self._paste_table.pack(fill="x")
        row = ttk.Frame(table, style=CARD_BODY_STYLE)
        row.pack(fill="x", padx=CARD_PADDING - 1, pady=(CARD_PADDING - 2, 0))
        ttk.Entry(row, textvariable=self._new_process, width=22, font=BODY_FONT).pack(
            side="left", fill="x", expand=True,
        )
        combobox(
            row,
            textvariable=self._new_method,
            values=[method.display_label for method in PasteMethod],
            width=14,
        ).pack(side="left", padx=CONTROL_GAP)
        ttk.Button(row, text="Add", command=self._add_override).pack(side="left")
        ttk.Button(row, text="Remove", command=self._remove_override).pack(
            side="left", padx=(CONTROL_GAP, 0),
        )

    def _add_override(self) -> None:
        process = normalized_process_name(self._new_process.get())
        if not process:
            return
        labels = {method.display_label: method for method in PasteMethod}
        self._overrides[process] = labels.get(self._new_method.get(), PasteMethod.CTRL_V)
        self._new_process.set("")
        self._render_overrides()

    def _remove_override(self) -> None:
        for process in self._paste_table.selection():
            self._overrides.pop(process, None)
        self._render_overrides()

    def _render_overrides(self) -> None:
        self._paste_table.delete(*self._paste_table.get_children())
        for process, method in sorted(self._overrides.items()):
            self._paste_table.insert(
                "", "end", iid=process, values=(process, method.display_label),
            )

    def _reset_pill_position(self) -> None:
        self._on_reset_pill_position()
        self._pill_status.configure(text=_PILL_RESET)

    def paste_method_overrides(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (process, method.value) for process, method in sorted(self._overrides.items())
        )

    def set_paste_method_overrides(
        self, overrides: tuple[tuple[str, str], ...],
    ) -> None:
        methods = {method.value: method for method in PasteMethod}
        self._overrides = {
            process: methods[value] for process, value in overrides if value in methods
        }
        self._render_overrides()

    def current_mode(self) -> OutputMode:
        return OutputMode(self._var.get())

    def set_mode(self, mode: OutputMode) -> None:
        self._var.set(mode.value)
        self._refresh_selection()

    def radio_buttons(self) -> list[ttk.Radiobutton]:
        return list(self._buttons.values())

    def _refresh_selection(self) -> None:
        for mode, mode_card in self._cards.items():
            set_card_selected(mode_card, mode is self.current_mode())
