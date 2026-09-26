import tkinter
from collections.abc import Callable
from tkinter import ttk

from voicetotext.core.color_scheme import detect_app_color_scheme
from voicetotext.history.retention import RETENTION_CHOICES
from voicetotext.settings.theme import (
    CONTROL_GAP,
    DANGER_BUTTON_STYLE,
    RowGroup,
    combobox,
    page_note,
    section_heading,
    switch,
    theme_for,
)
from voicetotext.settings.tk_discard_prompt import confirm_deletion


def _retention_label(days: int) -> str:
    return "Forever" if days <= 0 else f"{days} days"


def _where_files_live(logs_folder: str) -> str:
    place = logs_folder or "the app's own data folder"
    return (
        f"History is kept in your logs folder ({place}). Turning it on applies "
        "when you save, and recordings already there are kept. Use a local "
        "folder that is not synced or shared; Logs folder... below moves it."
    )


class TkHistoryPage(ttk.Frame):
    """Settings > History: whether dictations are kept, for how long, and wiping them.

    ``delete_all_history`` runs after the user confirms; without it the
    button is left out.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        *,
        logs_folder: str = "",
        delete_all_history: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._capture_content = tkinter.BooleanVar(master=self, value=False)
        self._retention = tkinter.StringVar(master=self, value=_retention_label(0))
        self._retention_days: dict[str, int] = {}
        self._delete_all_history = delete_all_history
        self._build(logs_folder)
        self.set_history_retention_days(0)

    def _build(self, logs_folder: str) -> None:
        section_heading(self, "Saved dictations", first=True)
        rows = RowGroup(self)
        rows.pack(fill="x")
        rows.add(
            "Save history",
            "Keeps the text and audio of each dictation so you can find, replay "
            "or retry it from the tray's History window.",
            lambda parent: switch(parent, self._capture_content),
        )
        row = rows.add(
            "Keep history for",
            "Older dictations are deleted automatically.",
            lambda parent: combobox(parent, textvariable=self._retention),
            fill=True,
        )
        assert isinstance(row.control, ttk.Combobox)
        self._retention_box = row.control
        if self._delete_all_history is not None:
            rows.add(
                "Delete all history",
                "Removes every saved dictation's text and recording. This cannot be undone.",
                lambda parent: ttk.Button(
                    parent, text="Delete all", style=DANGER_BUTTON_STYLE,
                    command=self._confirm_delete_all,
                ),
            )
        page_note(self, _where_files_live(logs_folder)).pack(
            anchor="w", pady=(CONTROL_GAP + 4, 0),
        )

    def _confirm_delete_all(self) -> None:
        if self._delete_all_history is None:
            return
        if confirm_deletion(
            self.winfo_toplevel(),
            theme_for(detect_app_color_scheme()),
            "Delete all history",
            "Delete every saved dictation's text and recording? This cannot be undone.",
        ):
            self._delete_all_history()

    def captures_content(self) -> bool:
        return self._capture_content.get()

    def set_capture_content(self, enabled: bool) -> None:
        self._capture_content.set(enabled)

    def history_retention_days(self) -> int:
        return self._retention_days.get(self._retention.get(), 0)

    def set_history_retention_days(self, days: int) -> None:
        choices = sorted({*RETENTION_CHOICES, max(days, 0)})
        self._retention_days = {_retention_label(choice): choice for choice in choices}
        self._retention_box.configure(values=list(self._retention_days))
        self._retention.set(_retention_label(days))
