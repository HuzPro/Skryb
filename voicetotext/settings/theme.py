"""The look every VoiceToText window shares: palette, type, spacing, widgets.

Tk draws everything flat in the ``clam`` theme, so the whole look lives in
ttk styles configured once by :func:`apply_theme`. Widgets pick a style by
name and never hold colours themselves, which is what lets a card switch
between its plain and selected look by swapping a style prefix.
"""

import base64
import io
import tkinter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from tkinter import ttk

from PIL import Image, ImageDraw

from voicetotext.core.color_scheme import ColorScheme

_FAMILY = "Segoe UI"
_SEMIBOLD = "Segoe UI Semibold"
# Pixel sizes, because Tk only takes whole points and 10.5 pt is 14 px.
TITLE_FONT = (_SEMIBOLD, -24)
SECTION_FONT = (_SEMIBOLD, -16)
ROW_TITLE_FONT = (_SEMIBOLD, -14)
BODY_FONT = (_FAMILY, -13)
_STRONG_BODY_FONT = (_SEMIBOLD, -13)
# Segoe UI Symbol draws the sidebar glyphs in one colour; plain Segoe UI
# falls back to the colour emoji font for some of them.
GLYPH_FONT = ("Segoe UI Symbol", -14)

PAGE_PADDING = 24
SECTION_GAP = 28
ROW_GAP = 12
CONTROL_GAP = 8
CARD_PADDING = 14
SIDEBAR_WIDTH = 190
CONTENT_MAX_WIDTH = 640
CONTROL_COLUMN_WIDTH = 220
# Wrap widths that fit the narrowest content column (the 860 px minimum
# window) and stay put when the column is capped at its widest.
DESCRIPTION_WRAP = 330
CARD_TEXT_WRAP = 540
PAGE_TEXT_WRAP = 580

PAGE_TITLE_STYLE = "PageTitle.TLabel"
SECTION_STYLE = "Section.TLabel"
MUTED_STYLE = "Muted.TLabel"
CARD_STYLE = "Card.TFrame"
CARD_BODY_STYLE = "CardBody.TFrame"
SELECTED_CARD_STYLE = "Selected.TFrame"
DIVIDER_STYLE = "Divider.TFrame"
CARD_TITLE_STYLE = "CardTitle.TLabel"
CARD_TEXT_STYLE = "Card.TLabel"
CARD_MUTED_STYLE = "CardMuted.TLabel"
SWITCH_STYLE = "Card.Switch.TCheckbutton"
CHECK_STYLE = "Card.TCheckbutton"
RADIO_STYLE = "Card.TRadiobutton"
RADIO_TITLE_STYLE = "CardTitle.TRadiobutton"
PRIMARY_BUTTON_STYLE = "Primary.TButton"
DANGER_BUTTON_STYLE = "Danger.TButton"
LINK_BUTTON_STYLE = "CardLink.TButton"
PAGE_LINK_BUTTON_STYLE = "Link.TButton"
ICON_BUTTON_STYLE = "CardIcon.TButton"
SMALL_BUTTON_STYLE = "Small.TButton"
KEYCAP_ENTRY_STYLE = "KeyCap.TEntry"
THIN_PROGRESS_STYLE = "Thin.Horizontal.TProgressbar"

# A card's own look and everything inside it switch together by prefix.
_CARD_PREFIX = "Card"
_SELECTED_PREFIX = "Selected"


@dataclass(frozen=True)
class Theme:
    """One colour palette; every window draws from it.

    ``background`` is the window, ``sidebar`` the slightly darker strip
    beside it, ``surface`` cards and fields. ``selection_background`` tints
    the chosen card or row; ``hover`` is the fainter tint under the pointer.
    ``strong_border`` outlines unticked boxes and switches, which the card
    ``border`` is too faint for. ``status_*`` colour the footer message.
    """

    name: str
    background: str
    sidebar: str
    surface: str
    foreground: str
    muted: str
    disabled: str
    border: str
    strong_border: str
    hover: str
    select_background: str
    select_foreground: str
    selection_background: str
    accent: str
    accent_hover: str
    accent_foreground: str
    danger: str
    success: str
    warning: str
    scrollbar_thumb: str
    status_neutral: str
    status_error: str

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


LIGHT_THEME = Theme(
    name="light",
    background="#f7f7f8",
    sidebar="#efeff1",
    surface="#ffffff",
    foreground="#18181b",
    muted="#71717a",
    disabled="#a1a1aa",
    border="#e4e4e7",
    strong_border="#c4c4cc",
    hover="#e7e7ea",
    select_background="#e8f0fe",
    select_foreground="#18181b",
    selection_background="#e8f0fe",
    accent="#2563eb",
    accent_hover="#1d4ed8",
    accent_foreground="#ffffff",
    danger="#dc2626",
    success="#16a34a",
    warning="#d97706",
    scrollbar_thumb="#d4d4d8",
    status_neutral="#71717a",
    status_error="#dc2626",
)

DARK_THEME = Theme(
    name="dark",
    background="#1e1e20",
    sidebar="#18181a",
    surface="#2a2a2e",
    foreground="#ededf0",
    muted="#a1a1aa",
    disabled="#6b6b73",
    border="#3a3a40",
    strong_border="#5a5a63",
    hover="#2c2c31",
    select_background="#25344d",
    select_foreground="#ededf0",
    selection_background="#25344d",
    accent="#7aa7ff",
    accent_hover="#9bbcff",
    # White on this light blue is 2.4:1; the dark ink reads at 8:1.
    accent_foreground="#0b1733",
    danger="#f87171",
    success="#4ade80",
    warning="#fbbf24",
    scrollbar_thumb="#4a4a52",
    status_neutral="#a1a1aa",
    status_error="#f87171",
)

_THEMES: dict[ColorScheme, Theme] = {
    ColorScheme.LIGHT: LIGHT_THEME,
    ColorScheme.DARK: DARK_THEME,
}


def theme_for(scheme: ColorScheme) -> Theme:
    return _THEMES[scheme]


def apply_theme(container: tkinter.Misc, theme: Theme) -> None:
    """Recolour the UI under *container* for *theme*.

    Switches ttk to the recolourable ``clam`` theme (the native Windows
    themes ignore colour options), configures every style the windows use,
    points the combobox dropdown list at the theme via the option database,
    and recolours the plain-Tk widgets (``Listbox``/``Canvas``/``Text``)
    that ttk styling does not reach. ``container`` must be a plain-Tk
    container (``Toplevel``/``Tk``/``tk.Frame``); ttk frames reject ``bg``.
    """
    style = ttk.Style(container)
    style.theme_use("clam")
    _install_images(container, style, theme)
    _configure_styles(style, theme)
    _configure_combobox_popdown(container, theme)
    container.configure(bg=theme.background)
    _recolour_plain_widgets(container, theme)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


def card(parent: tkinter.Misc, *, padding: int | tuple[int, ...] = CARD_PADDING) -> ttk.Frame:
    return ttk.Frame(parent, style=CARD_STYLE, padding=padding)


def divider(parent: tkinter.Misc) -> ttk.Frame:
    return ttk.Frame(parent, style=DIVIDER_STYLE, height=1)


def set_card_selected(widget: ttk.Frame, selected: bool) -> None:
    """Give a card, and everything drawn on it, the selected or plain look."""
    prefix = _SELECTED_PREFIX if selected else _CARD_PREFIX
    _swap_surface(widget, prefix)


def is_card_selected(widget: ttk.Frame) -> bool:
    return str(widget.cget("style")).startswith(_SELECTED_PREFIX)


def _swap_surface(widget: tkinter.Misc, prefix: str) -> None:
    if isinstance(widget, ttk.Widget):
        style = str(widget.cget("style"))
        for old in (_CARD_PREFIX, _SELECTED_PREFIX):
            if style.startswith(old) and old != prefix:
                widget.configure(style=prefix + style[len(old):])
    for child in widget.winfo_children():
        _swap_surface(child, prefix)


def section_heading(parent: tkinter.Misc, text: str, *, first: bool = False) -> ttk.Label:
    label = ttk.Label(parent, text=text, style=SECTION_STYLE)
    label.pack(anchor="w", pady=(0 if first else SECTION_GAP, CONTROL_GAP))
    return label


def page_note(parent: tkinter.Misc, text: str = "", **options: object) -> ttk.Label:
    """A muted line on the page background, under or above a card."""
    return ttk.Label(
        parent, text=text, style=MUTED_STYLE, wraplength=PAGE_TEXT_WRAP,
        justify="left", **options,
    )


def hide_while_empty(label: ttk.Label, variable: tkinter.StringVar) -> None:
    """Keep a gridded message *label* out of the layout while it has nothing to say."""

    def sync(*_args: object) -> None:
        if variable.get():
            label.grid()
        else:
            label.grid_remove()

    variable.trace_add("write", sync)
    sync()


ControlFactory = Callable[[tkinter.Misc], tkinter.Widget]


class SettingRow(ttk.Frame):
    """Title and description on the left, the control in a fixed right column.

    The control is built by *control* with the row as its parent. It sits
    in a column of the same width on every row, so controls line up down
    the page; ``fill`` stretches it across that column (combos, entries).
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        title: str,
        description: str = "",
        control: ControlFactory | None = None,
        *,
        fill: bool = False,
        column_width: int = CONTROL_COLUMN_WIDTH,
    ) -> None:
        super().__init__(parent, style=CARD_BODY_STYLE)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, minsize=column_width)
        self.title = ttk.Label(self, text=title, style=CARD_TITLE_STYLE)
        self.title.grid(row=0, column=0, sticky="w")
        self.description = ttk.Label(
            self, text="", style=CARD_MUTED_STYLE,
            wraplength=DESCRIPTION_WRAP if control else CARD_TEXT_WRAP,
            justify="left",
        )
        self.set_description(description)
        self.control = control(self) if control is not None else None
        if self.control is not None:
            self.control.grid(
                row=0, column=1, rowspan=2, sticky="ew" if fill else "e",
                padx=(16, 0),
            )

    def set_description(self, text: str) -> None:
        self.description.configure(text=text)
        if text:
            self.description.grid(row=1, column=0, sticky="w", pady=(2, 0))
        else:
            self.description.grid_remove()


class RowGroup(ttk.Frame):
    """A card holding setting rows with a hairline between each."""

    def __init__(self, parent: tkinter.Misc) -> None:
        super().__init__(parent, style=CARD_STYLE, padding=(CARD_PADDING, 2))
        self._rows = 0

    def add(
        self,
        title: str,
        description: str = "",
        control: ControlFactory | None = None,
        *,
        fill: bool = False,
    ) -> SettingRow:
        self._separate()
        row = SettingRow(self, title, description, control, fill=fill)
        row.pack(fill="x", pady=ROW_GAP)
        return row

    def add_widget(self, build: ControlFactory) -> tkinter.Widget:
        """A free-form block (a text box, a list) spanning the whole card."""
        self._separate()
        widget = build(self)
        widget.pack(fill="x", pady=ROW_GAP)
        return widget

    def _separate(self) -> None:
        if self._rows:
            divider(self).pack(fill="x")
        self._rows += 1


def switch(parent: tkinter.Misc, variable: tkinter.BooleanVar, **options: object) -> ttk.Checkbutton:
    return ttk.Checkbutton(parent, variable=variable, style=SWITCH_STYLE, **options)


def combobox(parent: tkinter.Misc, **options: object) -> ttk.Combobox:
    options.setdefault("state", "readonly")
    return ttk.Combobox(parent, font=BODY_FONT, **options)


def _should_show_scrollbar(content_height: int, viewport_height: int) -> bool:
    """The scrollbar is needed only when the content overflows the viewport."""
    return content_height > viewport_height


def _scroll_region_height(content_height: int, viewport_height: int) -> int:
    """Scroll region never shorter than the viewport.

    A region shorter than the viewport lets the canvas scroll into empty
    space above the content. Clamping to the viewport height makes
    "nothing to scroll" actually mean nothing moves.
    """
    return max(content_height, viewport_height)


class ScrollArea(ttk.Frame):
    """A page: a vertically scrolling column, centred and capped in width.

    The column is at most :data:`CONTENT_MAX_WIDTH` wide plus the page
    padding, like a settings page on the web, and the scrollbar appears
    only when the content is taller than the window.
    """

    def __init__(self, parent: tkinter.Misc, title: str = "") -> None:
        super().__init__(parent)
        self._canvas = tkinter.Canvas(
            self, highlightthickness=0, borderwidth=0, yscrollincrement=24,
        )
        self._scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self._canvas.yview,
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar_visible = False
        self._content_height = 0
        self.body = ttk.Frame(self._canvas, padding=PAGE_PADDING)
        self._body_id = self._canvas.create_window(0, 0, window=self.body, anchor="n")
        if title:
            ttk.Label(self.body, text=title, style=PAGE_TITLE_STYLE).pack(
                anchor="w", pady=(0, 20),
            )
        self.body.bind("<Configure>", self._on_body_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

    def scroll(self, units: int) -> None:
        if self._scrollbar_visible:
            self._canvas.yview_scroll(units, "units")

    def _on_body_configure(self, event: tkinter.Event) -> None:
        self._content_height = event.height
        self._sync(self._canvas.winfo_height())

    def _on_canvas_configure(self, event: tkinter.Event) -> None:
        width = min(event.width, CONTENT_MAX_WIDTH + 2 * PAGE_PADDING)
        self._canvas.coords(self._body_id, event.width // 2, 0)
        self._canvas.itemconfigure(self._body_id, width=width)
        self._sync(event.height)

    def _sync(self, viewport_height: int) -> None:
        self._canvas.configure(
            scrollregion=(0, 0, 0, _scroll_region_height(self._content_height, viewport_height)),
        )
        visible = _should_show_scrollbar(self._content_height, viewport_height)
        if visible == self._scrollbar_visible:
            return
        self._scrollbar_visible = visible
        # Laid over the page's right margin rather than beside it, so the
        # column stays put when moving between long and short pages.
        if visible:
            self._scrollbar.place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")
        else:
            self._scrollbar.place_forget()
            self._canvas.yview_moveto(0)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def _configure_styles(style: ttk.Style, theme: Theme) -> None:
    style.configure(
        ".",
        background=theme.background,
        foreground=theme.foreground,
        fieldbackground=theme.surface,
        bordercolor=theme.border,
        lightcolor=theme.surface,
        darkcolor=theme.surface,
        troughcolor=theme.background,
        arrowcolor=theme.muted,
        insertcolor=theme.foreground,
        selectbackground=theme.select_background,
        selectforeground=theme.select_foreground,
        # The dotted focus ring around a label reads as clutter; paint it out.
        focuscolor=theme.background,
        font=BODY_FONT,
    )
    _configure_frames(style, theme)
    _configure_labels(style, theme)
    _configure_indicators(style, theme)
    _configure_buttons(style, theme)
    _configure_fields(style, theme)
    _configure_progress(style, theme)
    _configure_slim_scrollbar(style, theme)
    _configure_table(style, theme)
    _configure_sidebar(style, theme)


def _surfaces(theme: Theme) -> tuple[tuple[str, str], ...]:
    """Each background a widget can sit on, by style prefix."""
    return (
        ("", theme.background),
        (_CARD_PREFIX, theme.surface),
        (_SELECTED_PREFIX, theme.selection_background),
    )


def _configure_frames(style: ttk.Style, theme: Theme) -> None:
    style.configure("TFrame", background=theme.background)
    for name, background, border in (
        (CARD_STYLE, theme.surface, theme.border),
        (SELECTED_CARD_STYLE, theme.selection_background, theme.accent),
    ):
        style.configure(
            name, background=background, bordercolor=border, lightcolor=border,
            darkcolor=border, relief="solid", borderwidth=1,
        )
    style.configure(CARD_BODY_STYLE, background=theme.surface)
    style.configure("SelectedBody.TFrame", background=theme.selection_background)
    style.configure(DIVIDER_STYLE, background=theme.border)


def _configure_labels(style: ttk.Style, theme: Theme) -> None:
    variants = (
        ("", theme.foreground, BODY_FONT),
        ("Title", theme.foreground, ROW_TITLE_FONT),
        ("Muted", theme.muted, BODY_FONT),
        ("Accent", theme.accent, _STRONG_BODY_FONT),
        ("Success", theme.success, _STRONG_BODY_FONT),
        ("Warning", theme.warning, _STRONG_BODY_FONT),
        ("Danger", theme.danger, _STRONG_BODY_FONT),
    )
    for prefix, background in _surfaces(theme):
        for variant, foreground, font in variants:
            style.configure(
                f"{prefix}{variant}.TLabel" if prefix or variant else "TLabel",
                background=background, foreground=foreground, font=font,
            )
    style.configure(
        PAGE_TITLE_STYLE, background=theme.background, foreground=theme.foreground,
        font=TITLE_FONT,
    )
    style.configure(
        SECTION_STYLE, background=theme.background, foreground=theme.muted,
        font=SECTION_FONT,
    )


def _configure_indicators(style: ttk.Style, theme: Theme) -> None:
    for prefix, background in _surfaces(theme):
        for widget in ("TCheckbutton", "Switch.TCheckbutton", "TRadiobutton"):
            name = f"{prefix}.{widget}" if prefix else widget
            style.configure(
                name, background=background, foreground=theme.foreground,
                focuscolor=background, font=BODY_FONT,
            )
            style.map(
                name,
                background=[("active", background)],
                foreground=[("disabled", theme.disabled)],
            )
        style.configure(
            f"{prefix}Title.TRadiobutton", background=background,
            foreground=theme.foreground, focuscolor=background, font=ROW_TITLE_FONT,
        )
        style.map(
            f"{prefix}Title.TRadiobutton",
            background=[("active", background)],
            foreground=[("disabled", theme.disabled)],
        )
    style.configure("TRadiobutton", indicatorsize=14, indicatormargin=(0, 0, 8, 0))
    style.map(
        "TRadiobutton",
        indicatorbackground=[
            ("disabled", theme.border), ("selected", theme.accent), ("!selected", theme.surface),
        ],
        indicatorforeground=[
            ("disabled selected", theme.disabled), ("selected", theme.accent_foreground),
            ("!selected", theme.surface),
        ],
        upperbordercolor=[
            ("disabled", theme.border), ("selected", theme.accent), ("!selected", theme.strong_border),
        ],
        lowerbordercolor=[
            ("disabled", theme.border), ("selected", theme.accent), ("!selected", theme.strong_border),
        ],
    )


def _configure_buttons(style: ttk.Style, theme: Theme) -> None:
    for name, background, foreground, border, hover in (
        ("TButton", theme.surface, theme.foreground, theme.strong_border, theme.hover),
        (PRIMARY_BUTTON_STYLE, theme.accent, theme.accent_foreground, theme.accent, theme.accent_hover),
        (DANGER_BUTTON_STYLE, theme.surface, theme.danger, theme.strong_border, theme.hover),
    ):
        style.configure(
            name, background=background, foreground=foreground, bordercolor=border,
            lightcolor=background, darkcolor=background, relief="solid",
            borderwidth=1, focusthickness=0, focuscolor=background,
            padding=(14, 6), anchor="center",
            font=_STRONG_BODY_FONT if name == PRIMARY_BUTTON_STYLE else BODY_FONT,
        )
        style.map(
            name,
            background=[("disabled", theme.surface), ("pressed", hover), ("active", hover)],
            lightcolor=[("disabled", theme.surface), ("pressed", hover), ("active", hover)],
            darkcolor=[("disabled", theme.surface), ("pressed", hover), ("active", hover)],
            bordercolor=[("disabled", theme.border), ("focus", theme.accent)],
            foreground=[("disabled", theme.disabled)],
            relief=[("pressed", "solid"), ("active", "solid")],
        )
    style.configure(SMALL_BUTTON_STYLE, padding=(10, 3))
    for prefix, background in _surfaces(theme):
        for suffix, foreground, hover, padding in (
            ("Link", theme.accent, theme.accent_hover, (0, 4)),
            ("Icon", theme.muted, theme.danger, (6, 2)),
        ):
            name = f"{prefix}{suffix}.TButton"
            style.configure(
                name, background=background, foreground=foreground,
                bordercolor=background, lightcolor=background, darkcolor=background,
                relief="flat", borderwidth=0, focusthickness=0, focuscolor=background,
                # Hairline arrows fringe with colour under ClearType; the
                # heavier weight keeps the icons grey.
                padding=padding, font=ROW_TITLE_FONT if suffix == "Icon" else _STRONG_BODY_FONT,
            )
            style.map(
                name,
                background=[("active", background), ("pressed", background)],
                foreground=[("disabled", theme.disabled), ("active", hover)],
            )


def _configure_fields(style: ttk.Style, theme: Theme) -> None:
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(
            name,
            fieldbackground=theme.surface,
            background=theme.surface,
            foreground=theme.foreground,
            bordercolor=theme.strong_border,
            lightcolor=theme.surface,
            darkcolor=theme.surface,
            insertcolor=theme.foreground,
            arrowcolor=theme.muted,
            arrowsize=12,
            padding=(8, 5),
            relief="flat",
        )
        style.map(
            name,
            fieldbackground=[("readonly", theme.surface), ("disabled", theme.background)],
            foreground=[("disabled", theme.disabled)],
            bordercolor=[("focus", theme.accent), ("disabled", theme.border)],
            lightcolor=[("focus", theme.surface)],
            background=[("active", theme.surface), ("pressed", theme.surface)],
            arrowcolor=[("disabled", theme.disabled)],
        )
    # A readonly combobox would otherwise paint its text as a blue selection.
    style.map(
        "TCombobox",
        selectbackground=[("readonly", theme.surface), ("focus", theme.select_background)],
        selectforeground=[("readonly", theme.foreground)],
    )
    style.configure(KEYCAP_ENTRY_STYLE, padding=(10, 5))
    style.configure(
        "ComboboxPopdownFrame", background=theme.surface, bordercolor=theme.border,
        lightcolor=theme.border, darkcolor=theme.border, relief="solid", borderwidth=1,
    )
    for prefix, background in _surfaces(theme):
        name = f"{prefix}.Horizontal.TScale" if prefix else "Horizontal.TScale"
        style.configure(name, background=background)


def _configure_progress(style: ttk.Style, theme: Theme) -> None:
    # ``clam`` draws the bar with its scrollbar thumb, so ``arrowsize`` is
    # what sets the bar's thickness.
    for name, thickness in (("Horizontal.TProgressbar", 8), (THIN_PROGRESS_STYLE, 3)):
        style.configure(
            name, background=theme.accent, troughcolor=theme.border,
            bordercolor=theme.border, lightcolor=theme.accent, darkcolor=theme.accent,
            arrowsize=thickness, borderwidth=0,
        )


def _configure_table(style: ttk.Style, theme: Theme) -> None:
    style.configure(
        "Treeview",
        background=theme.surface,
        fieldbackground=theme.surface,
        foreground=theme.foreground,
        bordercolor=theme.surface,
        lightcolor=theme.surface,
        darkcolor=theme.surface,
        borderwidth=0,
        rowheight=30,
        font=BODY_FONT,
    )
    style.map(
        "Treeview",
        background=[("selected", theme.selection_background)],
        foreground=[("selected", theme.foreground)],
    )
    style.configure(
        "Treeview.Heading",
        background=theme.surface,
        foreground=theme.muted,
        bordercolor=theme.border,
        lightcolor=theme.surface,
        darkcolor=theme.surface,
        relief="flat",
        # Lines the heading text up with the cells under it.
        padding=(2, 6),
        font=_STRONG_BODY_FONT,
    )
    style.map("Treeview.Heading", background=[("active", theme.surface)])


def _configure_slim_scrollbar(style: ttk.Style, theme: Theme) -> None:
    # A trough and a thumb, no arrow buttons. ``arrowsize`` still sets the
    # bar's thickness with the arrows gone.
    style.layout(
        "Vertical.TScrollbar",
        [
            (
                "Vertical.Scrollbar.trough",
                {
                    "sticky": "ns",
                    "children": [
                        ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"}),
                    ],
                },
            ),
        ],
    )
    style.configure(
        "Vertical.TScrollbar",
        troughcolor=theme.background,
        background=theme.scrollbar_thumb,
        bordercolor=theme.background,
        lightcolor=theme.scrollbar_thumb,
        darkcolor=theme.scrollbar_thumb,
        gripcount=0,
        borderwidth=0,
        relief="flat",
        arrowsize=10,
    )
    style.map(
        "Vertical.TScrollbar",
        background=[("active", theme.muted), ("pressed", theme.muted)],
        lightcolor=[("active", theme.muted), ("pressed", theme.muted)],
        darkcolor=[("active", theme.muted), ("pressed", theme.muted)],
    )


def _configure_sidebar(style: ttk.Style, theme: Theme) -> None:
    style.configure("Sidebar.TFrame", background=theme.sidebar)
    style.configure(
        "SidebarTitle.TLabel", background=theme.sidebar, foreground=theme.foreground,
        font=ROW_TITLE_FONT,
    )
    for state, background, foreground, font in (
        ("Nav", theme.sidebar, theme.foreground, BODY_FONT),
        ("NavHover", theme.hover, theme.foreground, BODY_FONT),
        ("NavSelected", theme.selection_background, theme.accent, _STRONG_BODY_FONT),
    ):
        style.configure(f"{state}.TFrame", background=background)
        style.configure(
            f"{state}.TLabel", background=background, foreground=foreground, font=font,
        )


def _configure_combobox_popdown(container: tkinter.Misc, theme: Theme) -> None:
    # The combobox dropdown is a separate Tk Listbox not reachable through
    # ttk styling; the option database is the only way to recolour it.
    container.option_add("*TCombobox*Listbox.background", theme.surface)
    container.option_add("*TCombobox*Listbox.foreground", theme.foreground)
    container.option_add("*TCombobox*Listbox.selectBackground", theme.selection_background)
    container.option_add("*TCombobox*Listbox.selectForeground", theme.foreground)
    container.option_add("*TCombobox*Listbox.font", BODY_FONT)
    container.option_add("*TCombobox*Listbox.borderWidth", 0)
    container.option_add("*TCombobox*Listbox.highlightThickness", 0)


def _recolour_plain_widgets(widget: tkinter.Misc, theme: Theme) -> None:
    for child in widget.winfo_children():
        if isinstance(child, tkinter.Listbox):
            child.configure(
                bg=theme.background,
                fg=theme.foreground,
                selectbackground=theme.selection_background,
                selectforeground=theme.foreground,
                highlightthickness=0,
                borderwidth=0,
                activestyle="none",
                font=BODY_FONT,
            )
        elif isinstance(child, tkinter.Canvas):
            child.configure(bg=theme.background, highlightthickness=0)
        elif isinstance(child, tkinter.Text):
            child.configure(
                bg=theme.surface,
                fg=theme.foreground,
                insertbackground=theme.foreground,
                selectbackground=theme.selection_background,
                selectforeground=theme.foreground,
                highlightthickness=1,
                highlightbackground=theme.strong_border,
                highlightcolor=theme.accent,
                relief="flat",
                borderwidth=0,
                padx=10,
                pady=8,
                font=BODY_FONT,
            )
        _recolour_plain_widgets(child, theme)


# ---------------------------------------------------------------------------
# Switch, checkbox and slider images
# ---------------------------------------------------------------------------

# ``clam`` draws a checkbox tick as an X, a slider as a bevelled box, and
# has no switch at all, so these are small images drawn here once per
# theme. The Tk images are named and refilled in place: a style element
# holds on to its image by name, so a theme change repaints every switch
# already on screen.
_IMAGES: dict[str, tkinter.PhotoImage] = {}
_SUPERSAMPLE = 4
_SWITCH_SIZE = (34, 20)
_CHECK_SIZE = 16
_THUMB_SIZE = 16
_DISABLED_ALPHA = 0.4


def _install_images(
    container: tkinter.Misc, style: ttk.Style, theme: Theme,
) -> None:
    for name, data in _theme_pngs(theme).items():
        image = _IMAGES.get(name)
        if image is not None and image.tk is container.tk:
            image.configure(data=data)
        else:
            _IMAGES[name] = tkinter.PhotoImage(name=name, data=data, format="png", master=container)
    _create_image_elements(style)
    for layout, element in (("TCheckbutton", "Check.indicator"), ("Switch.TCheckbutton", "Switch.indicator")):
        style.layout(layout, [
            ("Checkbutton.padding", {"sticky": "nswe", "children": [
                (element, {"side": "left", "sticky": ""}),
                ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [
                    ("Checkbutton.label", {"sticky": "nswe"}),
                ]}),
            ]}),
        ])
    # A thin track and a round thumb instead of clam's bevelled box.
    style.layout("Horizontal.TScale", [
        ("Track.Scale.trough", {"sticky": "nswe", "children": [
            ("Track.Scale.slider", {"side": "left", "sticky": ""}),
        ]}),
    ])


def _create_image_elements(style: ttk.Style) -> None:
    """Once per Tk interpreter; theme changes refill the images they name."""
    elements = style.element_names()
    for element, prefix, width in (
        ("Switch.indicator", "vtt_switch", _SWITCH_SIZE[0] + 8),
        ("Check.indicator", "vtt_check", _CHECK_SIZE + 8),
    ):
        if element not in elements:
            style.element_create(
                element, "image", f"{prefix}_off",
                ("disabled", "selected", f"{prefix}_on_disabled"),
                ("disabled", f"{prefix}_off_disabled"),
                ("selected", f"{prefix}_on"),
                width=width, sticky="w",
            )
    if "Track.Scale.trough" not in elements:
        style.element_create(
            "Track.Scale.trough", "image", "vtt_track",
            border=(_THUMB_SIZE // 2, 0, _THUMB_SIZE // 2, 0), sticky="ew",
        )
        style.element_create(
            "Track.Scale.slider", "image", "vtt_thumb",
            ("pressed", "vtt_thumb_active"), ("active", "vtt_thumb_active"),
        )


@cache
def _theme_pngs(theme: Theme) -> dict[str, str]:
    images: dict[str, Image.Image] = {}
    for on in (True, False):
        state = "on" if on else "off"
        images[f"vtt_switch_{state}"] = _draw_switch(theme, on)
        images[f"vtt_check_{state}"] = _draw_check(theme, on)
    for name in list(images):
        faded = images[name].copy()
        faded.putalpha(faded.getchannel("A").point(lambda value: round(value * _DISABLED_ALPHA)))
        images[f"{name}_disabled"] = faded
    images["vtt_track"] = _draw_track(theme)
    images["vtt_thumb"] = _draw_thumb(theme.accent)
    images["vtt_thumb_active"] = _draw_thumb(theme.accent_hover)
    return {name: _png_base64(image) for name, image in images.items()}


def _draw_track(theme: Theme) -> Image.Image:
    width, height = _THUMB_SIZE + 8, _THUMB_SIZE
    scaled = (width * _SUPERSAMPLE, height * _SUPERSAMPLE)
    image = Image.new("RGBA", scaled, (0, 0, 0, 0))
    thickness = 4 * _SUPERSAMPLE
    top = (scaled[1] - thickness) / 2
    ImageDraw.Draw(image).rounded_rectangle(
        (0, top, scaled[0] - 1, top + thickness - 1), radius=thickness / 2,
        fill=theme.strong_border,
    )
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _draw_thumb(colour: str) -> Image.Image:
    side = _THUMB_SIZE * _SUPERSAMPLE
    image = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((0, 0, side - 1, side - 1), fill=colour)
    return image.resize((_THUMB_SIZE, _THUMB_SIZE), Image.Resampling.LANCZOS)


def _draw_switch(theme: Theme, on: bool) -> Image.Image:
    width, height = (side * _SUPERSAMPLE for side in _SWITCH_SIZE)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=height / 2,
        fill=theme.accent if on else theme.strong_border,
    )
    margin = 3 * _SUPERSAMPLE
    knob = height - 2 * margin
    left = width - margin - knob if on else margin
    draw.ellipse((left, margin, left + knob, margin + knob), fill="#ffffff")
    return image.resize(_SWITCH_SIZE, Image.Resampling.LANCZOS)


def _draw_check(theme: Theme, on: bool) -> Image.Image:
    side = _CHECK_SIZE * _SUPERSAMPLE
    image = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    box = (0, 0, side - 1, side - 1)
    if on:
        draw.rounded_rectangle(box, radius=4 * _SUPERSAMPLE, fill=theme.accent)
        draw.line(
            [(side * 0.26, side * 0.52), (side * 0.43, side * 0.69), (side * 0.75, side * 0.33)],
            fill=theme.accent_foreground, width=round(2 * _SUPERSAMPLE), joint="curve",
        )
    else:
        draw.rounded_rectangle(
            box, radius=4 * _SUPERSAMPLE, fill=theme.surface,
            outline=theme.strong_border, width=_SUPERSAMPLE,
        )
    return image.resize((_CHECK_SIZE, _CHECK_SIZE), Image.Resampling.LANCZOS)


def _png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
