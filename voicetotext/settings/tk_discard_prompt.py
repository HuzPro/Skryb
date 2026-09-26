import tkinter
from collections.abc import Callable, Sequence
from tkinter import ttk

from voicetotext.settings.theme import (
    CONTROL_GAP,
    DANGER_BUTTON_STYLE,
    MUTED_STYLE,
    PRIMARY_BUTTON_STYLE,
    Theme,
)
from voicetotext.settings.win32_titlebar import show_with_titlebar

# "save", "discard" or "cancel".
UnsavedChangesPrompt = Callable[[tkinter.Toplevel, Theme], str]

_CANCEL = "cancel"
_MESSAGE_WRAP = 420
# (label, answer, style), rightmost button first.
_Choice = tuple[str, str, str]


def ask_save_discard_cancel(parent: tkinter.Toplevel, theme: Theme) -> str:
    """Ask whether to keep edits the user is closing the window on.

    The stock message box can only offer Yes / No / Cancel. Closing it or
    pressing Escape cancels.
    """
    return _ask(
        parent,
        theme,
        "Unsaved changes",
        "Discard changes?",
        "You changed settings in this window and have not saved them.",
        (
            ("Save", "save", PRIMARY_BUTTON_STYLE),
            ("Discard", "discard", DANGER_BUTTON_STYLE),
            ("Cancel", _CANCEL, "TButton"),
        ),
        on_return="save",
    )


def confirm_deletion(
    parent: tkinter.Tk | tkinter.Toplevel, theme: Theme, heading: str, message: str,
) -> bool:
    """Ask before deleting something that cannot be brought back.

    Only the Delete button deletes: Return, Escape and closing all cancel.
    """
    return _ask(
        parent,
        theme,
        heading,
        heading,
        message,
        (("Delete", "delete", DANGER_BUTTON_STYLE), ("Cancel", _CANCEL, "TButton")),
        on_return=_CANCEL,
    ) == "delete"


def _ask(
    parent: tkinter.Tk | tkinter.Toplevel,
    theme: Theme,
    title: str,
    heading: str,
    message: str,
    choices: Sequence[_Choice],
    *,
    on_return: str,
) -> str:
    """A small modal in the window's own theme, centred over *parent*."""
    dialog = tkinter.Toplevel(parent)
    dialog.withdraw()
    dialog.title(title)
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.configure(bg=theme.background)
    answer = [_CANCEL]

    def choose(choice: str) -> None:
        answer[0] = choice
        dialog.destroy()

    frame = ttk.Frame(dialog, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=heading, style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        frame, text=message, style=MUTED_STYLE, wraplength=_MESSAGE_WRAP, justify="left",
    ).pack(anchor="w", pady=(4, 20))
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")
    for index, (label, choice, style) in enumerate(choices):
        ttk.Button(
            buttons, text=label, style=style, command=lambda chosen=choice: choose(chosen),
        ).pack(side="right", padx=(0, CONTROL_GAP if index else 0))
    dialog.protocol("WM_DELETE_WINDOW", lambda: choose(_CANCEL))
    dialog.bind("<Escape>", lambda _event: choose(_CANCEL))
    dialog.bind("<Return>", lambda _event: choose(on_return))

    dialog.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_reqwidth()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_reqheight()) // 3
    dialog.geometry(f"+{x}+{y}")
    show_with_titlebar(dialog, dark=theme.is_dark)
    dialog.grab_set()
    dialog.focus_force()
    parent.wait_window(dialog)
    return answer[0]
