import tkinter
from collections.abc import Callable
from tkinter import messagebox, ttk

from voicetotext.history.retention import RETENTION_CHOICES
from voicetotext.output.output_mode import OutputMode
from voicetotext.output.paste_method import PasteMethod, normalized_process_name
from voicetotext.settings.theme import HEADING_STYLE

_PASTE_HINT = "Leave empty to use Ctrl+V everywhere; mintty gets Shift+Insert automatically."


def _retention_label(days: int) -> str:
    return "Forever" if days <= 0 else f"{days} days"


class TkOutputPage(ttk.Frame):
    """Page for how transcribed text reaches the focused window, and history.

    Renders one radio button per :class:`OutputMode`, all bound to a
    single ``StringVar`` holding the enum value, so the page maps a
    selection straight back to the canonical mode. ``delete_all_history``
    runs after the user confirms; without it the button is left out.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        delete_all_history: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._var = tkinter.StringVar(
            master=self, value=OutputMode.DIRECT_PASTE.value,
        )
        self._buttons: dict[OutputMode, ttk.Radiobutton] = {}
        self._capture_content = tkinter.BooleanVar(master=self, value=False)
        self._retention = tkinter.StringVar(master=self, value=_retention_label(0))
        self._retention_days: dict[str, int] = {}
        self._delete_all_history = delete_all_history
        self._overrides: dict[str, PasteMethod] = {}
        self._new_process = tkinter.StringVar(master=self)
        self._new_method = tkinter.StringVar(
            master=self, value=PasteMethod.CTRL_V.display_label,
        )
        self._build()
        self.set_mode(OutputMode.DIRECT_PASTE)
        self.set_history_retention_days(0)

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
        self._build_paste_methods()
        self._build_history()

    def _build_paste_methods(self) -> None:
        ttk.Label(self, text="Paste method per app", style=HEADING_STYLE).pack(
            anchor="w", pady=(24, 8),
        )
        self._paste_table = ttk.Treeview(
            self, columns=("app", "method"), show="headings", height=3,
            selectmode="browse",
        )
        self._paste_table.heading("app", text="App", anchor="w")
        self._paste_table.heading("method", text="Paste method", anchor="w")
        self._paste_table.column("app", width=240, anchor="w")
        self._paste_table.column("method", width=160, anchor="w")
        self._paste_table.pack(anchor="w", fill="x")
        row = ttk.Frame(self)
        row.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Entry(row, textvariable=self._new_process, width=24).pack(side="left")
        ttk.Combobox(
            row,
            textvariable=self._new_method,
            values=[method.display_label for method in PasteMethod],
            state="readonly",
            width=16,
        ).pack(side="left", padx=8)
        ttk.Button(row, text="Add", command=self._add_override).pack(side="left")
        ttk.Button(row, text="Remove", command=self._remove_override).pack(
            side="left", padx=(8, 0),
        )
        ttk.Label(self, text=_PASTE_HINT, wraplength=560).pack(
            anchor="w", pady=(4, 0),
        )

    def _build_history(self) -> None:
        ttk.Label(self, text="History", style=HEADING_STYLE).pack(
            anchor="w", pady=(24, 8),
        )
        ttk.Checkbutton(
            self, text="Save dictation history (text and audio)",
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
        row = ttk.Frame(self)
        row.pack(anchor="w", fill="x", pady=(8, 0))
        ttk.Label(row, text="Keep history for").pack(side="left")
        self._retention_box = ttk.Combobox(
            row, textvariable=self._retention, state="readonly", width=12,
        )
        self._retention_box.pack(side="left", padx=8)
        if self._delete_all_history is not None:
            ttk.Button(
                row, text="Delete all history", command=self._confirm_delete_all,
            ).pack(side="left", padx=(16, 0))

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

    def _confirm_delete_all(self) -> None:
        if self._delete_all_history is None:
            return
        if messagebox.askyesno(
            "Delete all history",
            "Delete every saved dictation's text and recording? "
            "This cannot be undone.",
            parent=self,
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

    def radio_buttons(self) -> list[ttk.Radiobutton]:
        return list(self._buttons.values())
