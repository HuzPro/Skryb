import tkinter
from dataclasses import dataclass
from tkinter import ttk

from voicetotext.core.color_scheme import ColorScheme

_FONT_FAMILY = "Segoe UI"
_BODY_FONT = (_FONT_FAMILY, 10)
_HEADING_FONT = (_FONT_FAMILY, 11, "bold")

# ttk style names the views opt into for non-default appearance.
HEADING_STYLE = "Heading.TLabel"
SELECTED_FRAME_STYLE = "Selected.TFrame"
SELECTED_LABEL_STYLE = "Selected.TLabel"
ROW_FRAME_STYLE = "TFrame"
ROW_LABEL_STYLE = "TLabel"


@dataclass(frozen=True)
class Theme:
    """A flat colour palette for the settings window.

    ``surface`` is for input fields and scrollable backgrounds;
    ``selection_background`` fills the chosen model row; ``accent`` matches
    the tray icon's per-state blue. ``status_*`` are the footer message
    colours (the neutral one must stay readable on ``background``, which a
    single hard-coded grey could not across both schemes).
    """

    name: str
    background: str
    surface: str
    foreground: str
    select_background: str
    select_foreground: str
    selection_background: str
    accent: str
    border: str
    scrollbar_thumb: str
    status_neutral: str
    status_error: str

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


LIGHT_THEME = Theme(
    name="light",
    background="#f4f4f4",
    surface="#ffffff",
    foreground="#1c1c1c",
    select_background="#cfe3ff",
    select_foreground="#10243d",
    selection_background="#d6e6ff",
    accent="#1f6feb",
    border="#d0d0d0",
    scrollbar_thumb="#bcbcbc",
    status_neutral="#444444",
    status_error="#c0392b",
)

DARK_THEME = Theme(
    name="dark",
    background="#252526",
    surface="#3c3f41",
    foreground="#e6e6e6",
    select_background="#2f5b8f",
    select_foreground="#ffffff",
    selection_background="#2d4a6b",
    accent="#6fa8ff",
    border="#3f3f46",
    scrollbar_thumb="#5a5a5e",
    status_neutral="#cccccc",
    status_error="#ff6b6b",
)

_THEMES: dict[ColorScheme, Theme] = {
    ColorScheme.LIGHT: LIGHT_THEME,
    ColorScheme.DARK: DARK_THEME,
}


def theme_for(scheme: ColorScheme) -> Theme:
    return _THEMES[scheme]


def apply_theme(container: tkinter.Misc, theme: Theme) -> None:
    """Recolour the settings UI under *container* for *theme*.

    Switches ttk to the recolourable ``clam`` theme (the native Windows
    themes ignore colour options), configures flat modern styles for every
    ttk widget the settings window uses, points the combobox dropdown list
    at the theme via the option database, and recolours the plain-Tk
    widgets (``Listbox``/``Canvas``) that ttk styling does not reach.
    ``container`` must be a plain-Tk container (``Toplevel``/``Tk``/
    ``tk.Frame``); ttk frames reject the ``bg`` option.
    """
    style = ttk.Style(container)
    style.theme_use("clam")
    _configure_styles(style, theme)
    _configure_combobox_popdown(container, theme)
    container.configure(bg=theme.background)
    _recolour_plain_widgets(container, theme)


def _configure_styles(style: ttk.Style, theme: Theme) -> None:
    style.configure(
        ".",
        background=theme.background,
        foreground=theme.foreground,
        fieldbackground=theme.surface,
        bordercolor=theme.border,
        lightcolor=theme.surface,
        darkcolor=theme.surface,
        troughcolor=theme.surface,
        arrowcolor=theme.foreground,
        insertcolor=theme.foreground,
        selectbackground=theme.select_background,
        selectforeground=theme.select_foreground,
        # Hide the dotted focus ring (the ugly dotted box around a selected
        # radio's label) by painting it in the background colour.
        focuscolor=theme.background,
        font=_BODY_FONT,
    )
    style.configure("TFrame", background=theme.background)
    style.configure(
        "TLabel", background=theme.background, foreground=theme.foreground,
    )
    style.configure(
        HEADING_STYLE,
        background=theme.background,
        foreground=theme.foreground,
        font=_HEADING_FONT,
    )
    # Filled row selection (replaces the old border): tint the row frame
    # and its labels together so the whole card reads as selected.
    style.configure(SELECTED_FRAME_STYLE, background=theme.selection_background)
    style.configure(
        SELECTED_LABEL_STYLE,
        background=theme.selection_background,
        foreground=theme.foreground,
    )
    style.configure(
        "TRadiobutton", background=theme.background, foreground=theme.foreground,
    )
    style.map(
        "TRadiobutton",
        background=[("active", theme.background)],
        foreground=[("disabled", theme.border)],
        indicatorcolor=[("selected", theme.accent)],
    )
    style.configure(
        "TCheckbutton", background=theme.background, foreground=theme.foreground,
    )
    style.map(
        "TCheckbutton",
        background=[("active", theme.background)],
        foreground=[("disabled", theme.border)],
        indicatorcolor=[("selected", theme.accent)],
    )
    _configure_flat_button(style, theme)
    style.configure(
        "TEntry",
        fieldbackground=theme.surface,
        foreground=theme.foreground,
        insertcolor=theme.foreground,
        borderwidth=1,
        relief="flat",
    )
    style.configure(
        "TCombobox",
        fieldbackground=theme.surface,
        foreground=theme.foreground,
        arrowcolor=theme.foreground,
        background=theme.surface,
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "TCombobox",
        fieldbackground=[
            ("readonly", theme.surface),
            ("disabled", theme.background),
        ],
        foreground=[("disabled", theme.border)],
        selectbackground=[("readonly", theme.surface)],
        selectforeground=[("readonly", theme.foreground)],
    )
    style.configure(
        "TProgressbar",
        background=theme.accent,
        troughcolor=theme.surface,
        bordercolor=theme.surface,
        lightcolor=theme.accent,
        darkcolor=theme.accent,
    )
    _configure_slim_scrollbar(style, theme)
    _configure_table(style, theme)


def _configure_table(style: ttk.Style, theme: Theme) -> None:
    style.configure(
        "Treeview",
        background=theme.surface,
        fieldbackground=theme.surface,
        foreground=theme.foreground,
        borderwidth=0,
    )
    style.map(
        "Treeview",
        background=[("selected", theme.select_background)],
        foreground=[("selected", theme.select_foreground)],
    )
    style.configure(
        "Treeview.Heading",
        background=theme.background,
        foreground=theme.foreground,
        relief="flat",
    )


def _configure_flat_button(style: ttk.Style, theme: Theme) -> None:
    style.configure(
        "TButton",
        background=theme.surface,
        foreground=theme.foreground,
        bordercolor=theme.border,
        relief="flat",
        borderwidth=0,
        focusthickness=0,
        padding=(12, 6),
    )
    style.map(
        "TButton",
        background=[
            ("pressed", theme.select_background),
            ("active", theme.select_background),
            ("disabled", theme.background),
        ],
        foreground=[("disabled", theme.border)],
        relief=[("pressed", "flat"), ("active", "flat")],
    )


def _configure_slim_scrollbar(style: ttk.Style, theme: Theme) -> None:
    # Drop the up/down arrow buttons for a clean modern bar: just a trough
    # and an expanding thumb. ``arrowsize`` still sets the bar thickness
    # even with the arrows removed, so keep it wide enough to grab.
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
        borderwidth=0,
        relief="flat",
        arrowsize=16,
    )
    style.map(
        "Vertical.TScrollbar",
        background=[("active", theme.accent), ("pressed", theme.accent)],
    )


def _configure_combobox_popdown(container: tkinter.Misc, theme: Theme) -> None:
    # The combobox dropdown is a separate Tk Listbox not reachable through
    # ttk styling; the option database is the only way to recolour it.
    container.option_add("*TCombobox*Listbox.background", theme.surface)
    container.option_add("*TCombobox*Listbox.foreground", theme.foreground)
    container.option_add(
        "*TCombobox*Listbox.selectBackground", theme.select_background,
    )
    container.option_add(
        "*TCombobox*Listbox.selectForeground", theme.select_foreground,
    )
    container.option_add("*TCombobox*Listbox.font", _BODY_FONT)


def _recolour_plain_widgets(widget: tkinter.Misc, theme: Theme) -> None:
    for child in widget.winfo_children():
        if isinstance(child, tkinter.Listbox):
            child.configure(
                bg=theme.background,
                fg=theme.foreground,
                selectbackground=theme.select_background,
                selectforeground=theme.select_foreground,
                highlightthickness=0,
                borderwidth=0,
                activestyle="none",
                font=_BODY_FONT,
            )
        elif isinstance(child, tkinter.Canvas):
            child.configure(bg=theme.background, highlightthickness=0)
        _recolour_plain_widgets(child, theme)
