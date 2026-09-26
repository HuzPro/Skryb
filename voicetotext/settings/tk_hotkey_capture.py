import tkinter
from collections.abc import Callable
from tkinter import ttk
from typing import Protocol

from voicetotext.hotkeys.key_parser import parse_hotkey_combo
from voicetotext.settings.signal import Signal
from voicetotext.settings.theme import (
    CARD_BODY_STYLE,
    KEYCAP_ENTRY_STYLE,
    ROW_TITLE_FONT,
    SMALL_BUTTON_STYLE,
)

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]

_RECORD_LABEL = "Record"
_STOP_LABEL = "Stop Recording"
_ESCAPE_NAME = "esc"


class CaptureBridge(Protocol):
    """Native key-capture source the recorder drives during recording."""

    key_pressed: Signal

    def start(self) -> None: ...
    def stop(self) -> None: ...


def _run_now(action: Callable[[], None]) -> None:
    action()


def shown_combination(combo: str) -> str:
    """``"ctrl+shift+m"`` as the key caps read: ``"Ctrl + Shift + M"``."""
    return " + ".join(
        part.upper() if len(part) == 1 else part[:1].upper() + part[1:]
        for part in combo.split("+")
        if part
    )


def _typed_combination(text: str) -> str:
    """What the user typed, in the form the config stores: ``"ctrl+m"``."""
    return "+".join(part.strip().lower() for part in text.strip().split("+"))


class TkHotkeyCapture(ttk.Frame):
    """A key-cap field + Record/Stop control for capturing a global hotkey.

    The field shows the combination the way the keys read ("Ctrl + Alt +
    M"); what it stores is the config form ("ctrl+alt+m"). Recording is
    driven by an injected :class:`CaptureBridge` (the native Win32 hook in
    production, a fake in tests). The bridge emits canonical key names on
    its worker thread; ``schedule`` marshals each onto the GUI thread
    before it touches the field. Typed text is validated against
    :func:`parse_hotkey_combo`.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        capture_bridge: CaptureBridge,
        schedule: ScheduleOnGuiThread = _run_now,
    ) -> None:
        super().__init__(parent, style=CARD_BODY_STYLE)
        self._bridge = capture_bridge
        self._schedule = schedule
        self._combination = ""
        self._recording = False
        self._recorded_parts: list[str] = []
        self._validation_error = ""
        self.combination_changed = Signal()
        self.recording_started = Signal()
        self.recording_stopped = Signal()
        self._text_var = tkinter.StringVar(master=self)
        self._build()
        self._bridge.key_pressed.connect(self._on_bridge_key_raw)

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self._entry = ttk.Entry(
            self, textvariable=self._text_var, style=KEYCAP_ENTRY_STYLE,
            font=ROW_TITLE_FONT, width=12,
        )
        self._entry.grid(row=0, column=0, sticky="ew")
        self._entry.bind("<Return>", self._on_entry_commit)
        self._entry.bind("<FocusOut>", self._on_entry_commit)

        self._record_button = ttk.Button(
            self, text=_RECORD_LABEL, width=13, style=SMALL_BUTTON_STYLE,
            command=self._toggle_record,
        )
        self._record_button.grid(row=0, column=1, sticky="ns", padx=(6, 0))

        self._error_label = ttk.Label(
            self, text="", style="CardDanger.TLabel", wraplength=240, justify="left",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combination(self) -> str:
        return self._combination

    def set_combination(self, combo: str) -> None:
        self._combination = combo
        self._text_var.set(shown_combination(combo))
        self._clear_error()

    def is_recording(self) -> bool:
        return self._recording

    def display_text(self) -> str:
        return self._text_var.get()

    def record_button_text(self) -> str:
        return self._record_button.cget("text")

    def has_validation_error(self) -> bool:
        return bool(self._validation_error)

    def validation_error(self) -> str:
        return self._validation_error

    def start_recording(self) -> None:
        if self._recording:
            return
        self._recording = True
        self._recorded_parts = []
        self._clear_error()
        self._entry.state(["readonly"])
        self._text_var.set("")
        self._record_button.configure(text=_STOP_LABEL)
        self._bridge.start()
        self.recording_started.emit()

    def stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._bridge.stop()
        self._entry.state(["!readonly"])
        self._record_button.configure(text=_RECORD_LABEL)
        if self._recorded_parts:
            self._set_combination_internal("+".join(self._recorded_parts))
        else:
            self._text_var.set(shown_combination(self._combination))
        self.recording_stopped.emit()

    def destroy(self) -> None:
        # The capture hooks swallow every key system-wide, so a window
        # closed mid-recording must not leave them installed.
        try:
            self.stop_recording()
        finally:
            self._bridge.key_pressed.disconnect(self._on_bridge_key_raw)
            super().destroy()

    def set_typed_text(self, text: str) -> None:
        self._text_var.set(text)

    def commit_typed_text(self) -> None:
        self._on_text_committed()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _toggle_record(self) -> None:
        if self._recording:
            self.stop_recording()
        else:
            self.start_recording()

    def _cancel_recording(self) -> None:
        self._recorded_parts = []
        self.stop_recording()

    def _on_bridge_key_raw(self, name: str) -> None:
        self._schedule(lambda: self._on_bridge_key(name))

    def _on_bridge_key(self, name: str) -> None:
        if not self._recording:
            return
        if name == _ESCAPE_NAME:
            self._cancel_recording()
            return
        if self._recorded_parts and self._recorded_parts[-1] == name:
            return
        self._recorded_parts.append(name)
        self._text_var.set(shown_combination("+".join(self._recorded_parts)))

    def _on_entry_commit(self, _event: tkinter.Event) -> None:
        self._on_text_committed()

    def _on_text_committed(self) -> None:
        if self._recording:
            return
        text = _typed_combination(self._text_var.get())
        if text == self._combination:
            self._text_var.set(shown_combination(text))
            self._clear_error()
            return
        if not text:
            self._set_combination_internal("")
            self._clear_error()
            return
        try:
            parse_hotkey_combo(text)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._set_combination_internal(text)
        self._clear_error()

    def _set_combination_internal(self, combo: str) -> None:
        self._combination = combo
        self._text_var.set(shown_combination(combo))
        self.combination_changed.emit(combo)

    def _show_error(self, message: str) -> None:
        self._validation_error = message
        self._error_label.configure(text=f"Invalid combination: {message}")
        self._error_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _clear_error(self) -> None:
        self._validation_error = ""
        self._error_label.configure(text="")
        self._error_label.grid_remove()
