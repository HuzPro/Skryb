import tkinter
from collections.abc import Callable
from tkinter import ttk

from voicetotext.settings.signal import Signal
from voicetotext.settings.theme import HEADING_STYLE
from voicetotext.settings.tk_hotkey_capture import (
    CaptureBridge,
    TkHotkeyCapture,
)

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]

_TOGGLE_MODE = "toggle"
_PUSH_TO_TALK_MODE = "push_to_talk"
_ALLOWED_MODES = {_TOGGLE_MODE, _PUSH_TO_TALK_MODE}

_HINT = (
    "Type a combination, or click Record and press your keys, "
    "then click Stop."
)


def _run_now(action: Callable[[], None]) -> None:
    action()


class TkHotkeysPage(ttk.Frame):
    """Page for configuring the global hotkey combination and activation mode.

    Combines :class:`TkHotkeyCapture` for key-chord capture with a radio
    group for toggle / push-to-talk. Exposes getters the enclosing window
    uses to build a :class:`SettingsDraft`.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        capture_bridge: CaptureBridge,
        schedule: ScheduleOnGuiThread = _run_now,
    ) -> None:
        super().__init__(parent)
        self._combination = ""
        self._mode_var = tkinter.StringVar(master=self, value=_TOGGLE_MODE)
        self._mode_buttons: dict[str, ttk.Radiobutton] = {}
        self.recording_state_changed = Signal()
        self._build(capture_bridge, schedule)

    def _build(
        self, capture_bridge: CaptureBridge, schedule: ScheduleOnGuiThread,
    ) -> None:
        ttk.Label(self, text="Hotkey Combination", style=HEADING_STYLE).pack(
            anchor="w",
        )

        self._capture = TkHotkeyCapture(self, capture_bridge, schedule)
        self._capture.combination_changed.connect(self._on_combination_changed)
        self._capture.recording_started.connect(self._on_recording_started)
        self._capture.recording_stopped.connect(self._on_recording_stopped)
        self._capture.pack(fill="x", pady=(2, 0))

        ttk.Label(self, text=_HINT, wraplength=480, justify="left").pack(
            anchor="w", pady=(2, 0),
        )

        ttk.Label(self, text="Activation Mode", style=HEADING_STYLE).pack(
            anchor="w", pady=(12, 0),
        )
        self._add_mode_radio(
            _TOGGLE_MODE, "Toggle (press to start, press again to stop)",
        )
        self._add_mode_radio(
            _PUSH_TO_TALK_MODE,
            "Push-to-talk (hold to record, release to stop)",
        )

    def _add_mode_radio(self, value: str, label: str) -> None:
        button = ttk.Radiobutton(
            self, text=label, value=value, variable=self._mode_var,
        )
        button.pack(anchor="w")
        self._mode_buttons[value] = button

    def _on_combination_changed(self, combo: str) -> None:
        self._combination = combo

    def _on_recording_started(self) -> None:
        self.recording_state_changed.emit(True)

    def _on_recording_stopped(self) -> None:
        self.recording_state_changed.emit(False)

    def hotkey_capture(self) -> TkHotkeyCapture:
        return self._capture

    def current_combination(self) -> str:
        return self._combination

    def set_combination(self, combo: str) -> None:
        self._combination = combo
        self._capture.set_combination(combo)

    def current_mode(self) -> str:
        return self._mode_var.get()

    def set_mode(self, mode: str) -> None:
        if mode not in _ALLOWED_MODES:
            return
        self._mode_var.set(mode)
