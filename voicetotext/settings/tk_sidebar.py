import tkinter
from collections.abc import Callable, Sequence
from tkinter import ttk

from voicetotext.settings.theme import GLYPH_FONT, SIDEBAR_WIDTH

_GLYPH_COLUMN = 24


class _NavItem(ttk.Frame):
    """One sidebar entry: a glyph in a fixed column, then the section name."""

    def __init__(
        self, parent: tkinter.Misc, glyph: str, label: str, on_click: Callable[[], None],
    ) -> None:
        super().__init__(parent, style="Nav.TFrame", padding=(10, 7), cursor="hand2")
        self.columnconfigure(0, minsize=_GLYPH_COLUMN)
        self._glyph = ttk.Label(self, text=glyph, style="Nav.TLabel", font=GLYPH_FONT)
        self._glyph.grid(row=0, column=0, sticky="w")
        self._label = ttk.Label(self, text=label, style="Nav.TLabel")
        self._label.grid(row=0, column=1, sticky="w")
        self._selected = False
        self._hovered = False
        for widget in (self, self._glyph, self._label):
            widget.bind("<Button-1>", lambda _event: on_click())
            widget.bind("<Enter>", lambda _event: self._hover(True))
            widget.bind("<Leave>", lambda _event: self._hover(False))

    def label(self) -> str:
        return str(self._label.cget("text"))

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._restyle()

    def _hover(self, hovered: bool) -> None:
        self._hovered = hovered
        self._restyle()

    def _restyle(self) -> None:
        state = "NavSelected" if self._selected else "NavHover" if self._hovered else "Nav"
        self.configure(style=f"{state}.TFrame")
        self._glyph.configure(style=f"{state}.TLabel")
        self._label.configure(style=f"{state}.TLabel")


class TkSidebar(ttk.Frame):
    """The settings sections as a column of flat entries, one selected."""

    def __init__(
        self,
        parent: tkinter.Misc,
        sections: Sequence[tuple[str, str]],
        on_select: Callable[[int], None],
    ) -> None:
        super().__init__(parent, style="Sidebar.TFrame", width=SIDEBAR_WIDTH, padding=(8, 0))
        self.pack_propagate(False)
        ttk.Label(self, text="VoiceToText", style="SidebarTitle.TLabel").pack(
            anchor="w", padx=10, pady=(20, 16),
        )
        self._items = [
            _NavItem(self, glyph, label, lambda index=index: on_select(index))
            for index, (glyph, label) in enumerate(sections)
        ]
        for item in self._items:
            item.pack(fill="x", pady=1)

    def select(self, index: int) -> None:
        for position, item in enumerate(self._items):
            item.set_selected(position == index)

    def labels(self) -> list[str]:
        return [item.label() for item in self._items]
