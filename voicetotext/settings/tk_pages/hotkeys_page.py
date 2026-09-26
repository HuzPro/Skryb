import tkinter
from collections.abc import Callable, Sequence
from tkinter import ttk

from voicetotext.hotkeys.bindings import (
    Binding,
    BindingAction,
    BindingError,
    BindingMode,
    binding_for,
    validate_bindings,
)
from voicetotext.settings.signal import Signal
from voicetotext.settings.theme import (
    CARD_BODY_STYLE,
    CARD_MUTED_STYLE,
    CARD_TEXT_WRAP,
    CARD_TITLE_STYLE,
    CONTROL_GAP,
    ICON_BUTTON_STYLE,
    LINK_BUTTON_STYLE,
    RowGroup,
    combobox,
    page_note,
    section_heading,
    switch,
)
from voicetotext.settings.tk_hotkey_capture import (
    CaptureBridge,
    TkHotkeyCapture,
)

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]

_MODE_LABELS = {BindingMode.TOGGLE: "Toggle", BindingMode.HOLD: "Hold"}
_MODES_BY_LABEL = {label: mode for mode, label in _MODE_LABELS.items()}

_MODES_HINT = (
    "Toggle: press to start, press again to stop. "
    "Hold: hold to talk, let go to stop."
)
_DOUBLE_TAP_HINT = "Two quick presses keep it recording; the next press stops it."
_CAPTURE_HINT = (
    "Type a shortcut such as ctrl+alt+m, f13 or mouse4, or click Record, "
    "press the keys, then click Stop Recording. Leave a shortcut empty to "
    "turn it off."
)
# Every shortcut row lays its parts out on the same columns, so the key
# fields, the mode boxes and the remove buttons line up down the page.
_CAPTURE_COLUMN = 300
_MODE_COLUMN = 92
_REMOVE_COLUMN = 30


def _run_now(action: Callable[[], None]) -> None:
    action()


class ShortcutRow(ttk.Frame):
    """One shortcut field, with a Toggle/Hold choice and a remove button when
    the section needs them."""

    def __init__(
        self,
        parent: tkinter.Misc,
        capture_bridge: CaptureBridge,
        schedule: ScheduleOnGuiThread,
        *,
        with_mode: bool,
        on_remove: Callable[["ShortcutRow"], None] | None = None,
    ) -> None:
        super().__init__(parent, style=CARD_BODY_STYLE)
        self.columnconfigure(0, minsize=_CAPTURE_COLUMN)
        self.columnconfigure(1, minsize=_MODE_COLUMN)
        self.columnconfigure(2, minsize=_REMOVE_COLUMN)
        self.capture = TkHotkeyCapture(self, capture_bridge, schedule)
        self.capture.grid(row=0, column=0, sticky="new")
        self._mode_box = combobox(self, values=list(_MODES_BY_LABEL), width=7)
        self.set_mode(BindingMode.TOGGLE)
        if with_mode:
            self._mode_box.grid(row=0, column=1, sticky="new", padx=(CONTROL_GAP, 0))
        self.remove_button: ttk.Button | None = None
        if on_remove is not None:
            self.remove_button = ttk.Button(
                self, text="×", width=2, style=ICON_BUTTON_STYLE,
                command=lambda: on_remove(self),
            )
            self.remove_button.grid(row=0, column=2, sticky="e", padx=(CONTROL_GAP, 0))

    def mode(self) -> BindingMode:
        return _MODES_BY_LABEL.get(self._mode_box.get(), BindingMode.TOGGLE)

    def set_mode(self, mode: BindingMode) -> None:
        self._mode_box.set(_MODE_LABELS.get(mode, _MODE_LABELS[BindingMode.TOGGLE]))


class TkHotkeysPage(ttk.Frame):
    """Page for every global shortcut: any number of dictation shortcuts,
    the double-tap lock, and the optional edit, paste-last and copy-last
    shortcuts. Exposes getters the enclosing window uses to build a
    :class:`SettingsDraft`, plus :meth:`validation_error` for Save.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        capture_bridge: CaptureBridge,
        schedule: ScheduleOnGuiThread = _run_now,
    ) -> None:
        super().__init__(parent)
        self._bridge = capture_bridge
        self._schedule = schedule
        self._dictation_rows: list[ShortcutRow] = []
        self._double_tap = tkinter.BooleanVar(master=self, value=True)
        self.recording_state_changed = Signal()
        self._build()

    def _build(self) -> None:
        section_heading(self, "Dictation", first=True)
        rows = RowGroup(self)
        rows.pack(fill="x")
        dictate = self._block(rows, "Dictate", f"Start and stop dictation. {_MODES_HINT}")
        self._rows_frame = ttk.Frame(dictate, style=CARD_BODY_STYLE)
        self._rows_frame.pack(fill="x")
        ttk.Button(
            dictate, text="Add another", style=LINK_BUTTON_STYLE,
            command=self.add_dictation_row,
        ).pack(anchor="w")
        rows.add(
            "Double-tap to lock",
            f"For a Hold shortcut. {_DOUBLE_TAP_HINT}",
            lambda parent: switch(parent, self._double_tap),
        )

        section_heading(self, "More shortcuts")
        more = RowGroup(self)
        more.pack(fill="x")
        self._edit_row = self._optional(
            more, "Edit selected text by voice",
            "Select text, press this and say how to change it.", with_mode=True,
        )
        self._paste_row = self._optional(
            more, "Paste last dictation", "Paste it again where the cursor is.",
            with_mode=False,
        )
        self._copy_row = self._optional(
            more, "Copy last dictation", "Put it on the clipboard.", with_mode=False,
        )
        page_note(self, _CAPTURE_HINT).pack(anchor="w", pady=(CONTROL_GAP, 0))
        self.add_dictation_row()

    @staticmethod
    def _block(group: RowGroup, title: str, description: str) -> ttk.Frame:
        """A titled block whose shortcut fields sit under the text, full width.

        Beside the text the fields would leave too little room for a long
        title at the narrowest window.
        """

        def build(parent: tkinter.Misc) -> ttk.Frame:
            block = ttk.Frame(parent, style=CARD_BODY_STYLE)
            ttk.Label(block, text=title, style=CARD_TITLE_STYLE).pack(anchor="w")
            ttk.Label(
                block, text=description, style=CARD_MUTED_STYLE,
                wraplength=CARD_TEXT_WRAP, justify="left",
            ).pack(anchor="w", pady=(2, CONTROL_GAP + 2))
            return block

        block = group.add_widget(build)
        assert isinstance(block, ttk.Frame)
        return block

    def _optional(
        self, group: RowGroup, title: str, description: str, *, with_mode: bool,
    ) -> ShortcutRow:
        row = self._row(self._block(group, title, description), with_mode=with_mode)
        row.pack(anchor="w")
        return row

    def _row(
        self,
        parent: tkinter.Misc,
        *,
        with_mode: bool,
        on_remove: Callable[[ShortcutRow], None] | None = None,
    ) -> ShortcutRow:
        row = ShortcutRow(
            parent, self._bridge, self._schedule,
            with_mode=with_mode, on_remove=on_remove,
        )
        row.capture.recording_started.connect(
            lambda: self._on_capture_started(row.capture),
        )
        row.capture.recording_stopped.connect(self._on_capture_stopped)
        return row

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_dictation_row(self) -> ShortcutRow:
        row = self._row(self._rows_frame, with_mode=True, on_remove=self.remove_dictation_row)
        row.pack(anchor="w", pady=(0, CONTROL_GAP))
        self._dictation_rows.append(row)
        self._refresh_remove_buttons()
        return row

    def remove_dictation_row(self, row: ShortcutRow) -> None:
        if len(self._dictation_rows) <= 1 or row not in self._dictation_rows:
            return
        self._dictation_rows.remove(row)
        row.destroy()
        self._refresh_remove_buttons()

    def dictation_rows(self) -> list[ShortcutRow]:
        return list(self._dictation_rows)

    def edit_row(self) -> ShortcutRow:
        return self._edit_row

    def paste_row(self) -> ShortcutRow:
        return self._paste_row

    def copy_row(self) -> ShortcutRow:
        return self._copy_row

    def bindings(self) -> tuple[Binding, ...]:
        return tuple(
            binding_for(row.capture.combination(), row.mode(), action)
            for row, action in self._rows_with_actions()
            if row.capture.combination().strip()
        )

    def set_bindings(self, bindings: Sequence[Binding]) -> None:
        rows, self._dictation_rows = self._dictation_rows, []
        for row in rows:
            row.destroy()
        for binding in [b for b in bindings if b.action is BindingAction.DICTATE]:
            self._show(self.add_dictation_row(), binding)
        if not self._dictation_rows:
            self.add_dictation_row()
        for row, action in self._optional_rows():
            self._show(row, next((b for b in bindings if b.action is action), None))

    def double_tap_locks_hold(self) -> bool:
        return self._double_tap.get()

    def set_double_tap_locks_hold(self, enabled: bool) -> None:
        self._double_tap.set(enabled)

    def validation_error(self) -> str:
        """Why these shortcuts cannot be saved, or "" when they can."""
        for row, _action in self._rows_with_actions():
            row.capture.commit_typed_text()
            if row.capture.has_validation_error():
                return (
                    f"Shortcut {row.capture.display_text()!r} is not valid: "
                    f"{row.capture.validation_error()}."
                )
        try:
            validate_bindings(self.bindings())
        except BindingError as exc:
            return str(exc)
        return ""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rows_with_actions(self) -> list[tuple[ShortcutRow, BindingAction]]:
        dictation = [(row, BindingAction.DICTATE) for row in self._dictation_rows]
        return dictation + self._optional_rows()

    def _optional_rows(self) -> list[tuple[ShortcutRow, BindingAction]]:
        return [
            (self._edit_row, BindingAction.EDIT_SELECTION),
            (self._paste_row, BindingAction.PASTE_LAST),
            (self._copy_row, BindingAction.COPY_LAST),
        ]

    @staticmethod
    def _show(row: ShortcutRow, binding: Binding | None) -> None:
        row.capture.set_combination(binding.combo if binding else "")
        row.set_mode(binding.mode if binding else BindingMode.TOGGLE)

    def _refresh_remove_buttons(self) -> None:
        state = ["!disabled"] if len(self._dictation_rows) > 1 else ["disabled"]
        for row in self._dictation_rows:
            if row.remove_button is not None:
                row.remove_button.state(state)

    def _on_capture_started(self, started: TkHotkeyCapture) -> None:
        for row, _action in self._rows_with_actions():
            if row.capture is not started and row.capture.is_recording():
                row.capture.stop_recording()
        self.recording_state_changed.emit(True)

    def _on_capture_stopped(self) -> None:
        if not any(row.capture.is_recording() for row, _ in self._rows_with_actions()):
            self.recording_state_changed.emit(False)
