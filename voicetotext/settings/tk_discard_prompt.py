import tkinter
from collections.abc import Callable
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


def ask_save_discard_cancel(parent: tkinter.Toplevel, theme: Theme) -> str:
    """Ask whether to keep edits the user is closing the window on.

    A small modal in the window's own theme, because the stock message box
    can only offer Yes / No / Cancel. Closing it or pressing Escape cancels.
    """
    dialog = tkinter.Toplevel(parent)
    dialog.withdraw()
    dialog.title("Unsaved changes")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.configure(bg=theme.background)
    answer = ["cancel"]

    def choose(choice: str) -> None:
        answer[0] = choice
        dialog.destroy()

    frame = ttk.Frame(dialog, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Discard changes?", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        frame,
        text="You changed settings in this window and have not saved them.",
        style=MUTED_STYLE,
    ).pack(anchor="w", pady=(4, 20))
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")
    ttk.Button(
        buttons, text="Save", style=PRIMARY_BUTTON_STYLE, command=lambda: choose("save"),
    ).pack(side="right")
    ttk.Button(
        buttons, text="Discard", style=DANGER_BUTTON_STYLE,
        command=lambda: choose("discard"),
    ).pack(side="right", padx=CONTROL_GAP)
    ttk.Button(buttons, text="Cancel", command=lambda: choose("cancel")).pack(side="right")
    dialog.protocol("WM_DELETE_WINDOW", lambda: choose("cancel"))
    dialog.bind("<Escape>", lambda _event: choose("cancel"))
    dialog.bind("<Return>", lambda _event: choose("save"))

    dialog.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_reqwidth()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_reqheight()) // 3
    dialog.geometry(f"+{x}+{y}")
    show_with_titlebar(dialog, dark=theme.is_dark)
    dialog.grab_set()
    dialog.focus_force()
    parent.wait_window(dialog)
    return answer[0]
