import tkinter
from dataclasses import dataclass
from tkinter import ttk

from voicetotext.core.color_scheme import ColorScheme


@dataclass(frozen=True)
class Theme:
    """A flat colour palette for the settings window.

    ``surface`` is for input fields and scrollable backgrounds;
    ``accent`` matches the tray icon's per-state blue. ``status_*`` are
    the footer message colours (the neutral one must stay readable on
    ``background``, which a single hard-coded grey could not across both
    schemes).
    """

    name: str
    background: str
    surface: str
    foreground: str
    select_background: str
    select_foreground: str
    accent: str
    border: str
    status_neutral: str
    status_error: str


LIGHT_THEME = Theme(
    name="light",
    background="#f4f4f4",
    surface="#ffffff",
    foreground="#1c1c1c",
    select_background="#cfe3ff",
    select_foreground="#10243d",
    accent="#1f6feb",
    border="#c4c4c4",
    status_neutral="#333333",
    status_error="#c0392b",
)

DARK_THEME = Theme(
    name="dark",
    background="#2b2b2b",
    surface="#3c3f41",
    foreground="#e6e6e6",
    select_background="#2f5b8f",
    select_foreground="#ffffff",
    accent="#6fa8ff",
    border="#555555",
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
    themes ignore colour options), configures every ttk widget style the
    settings window uses, points the combobox dropdown list at the theme
    via the option database, and recolours the plain-Tk widgets
    (``Listbox``/``Canvas``) that ttk styling does not reach. ``container``
    must be a plain-Tk container (``Toplevel``/``Tk``/``tk.Frame``); ttk
    frames reject the ``bg`` option.
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
        troughcolor=theme.surface,
        arrowcolor=theme.foreground,
        insertcolor=theme.foreground,
        selectbackground=theme.select_background,
        selectforeground=theme.select_foreground,
    )
    # ``bordercolor`` on TFrame paints the selected model row's solid
    # border (the only frames with a non-zero borderwidth) in the accent.
    style.configure("TFrame", background=theme.background, bordercolor=theme.accent)
    style.configure(
        "TLabel", background=theme.background, foreground=theme.foreground,
    )
    style.configure(
        "TRadiobutton", background=theme.background, foreground=theme.foreground,
    )
    style.map(
        "TRadiobutton",
        background=[("active", theme.background)],
        foreground=[("disabled", theme.border)],
    )
    style.configure(
        "TButton", background=theme.surface, foreground=theme.foreground,
    )
    style.map(
        "TButton",
        background=[
            ("active", theme.select_background),
            ("disabled", theme.background),
        ],
        foreground=[("disabled", theme.border)],
    )
    style.configure(
        "TEntry",
        fieldbackground=theme.surface,
        foreground=theme.foreground,
        insertcolor=theme.foreground,
    )
    style.configure(
        "TCombobox",
        fieldbackground=theme.surface,
        foreground=theme.foreground,
        arrowcolor=theme.foreground,
        background=theme.surface,
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
        bordercolor=theme.border,
    )
    style.configure(
        "Vertical.TScrollbar",
        background=theme.surface,
        troughcolor=theme.background,
        arrowcolor=theme.foreground,
        bordercolor=theme.border,
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


def _recolour_plain_widgets(widget: tkinter.Misc, theme: Theme) -> None:
    for child in widget.winfo_children():
        if isinstance(child, tkinter.Listbox):
            child.configure(
                bg=theme.surface,
                fg=theme.foreground,
                selectbackground=theme.select_background,
                selectforeground=theme.select_foreground,
                highlightthickness=0,
                borderwidth=0,
            )
        elif isinstance(child, tkinter.Canvas):
            child.configure(bg=theme.background, highlightthickness=0)
        _recolour_plain_widgets(child, theme)
