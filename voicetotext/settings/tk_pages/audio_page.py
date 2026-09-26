import math
import tkinter
from dataclasses import replace
from tkinter import ttk

from voicetotext.recording.input_devices import (
    DEFAULT_LABEL as _DEFAULT_LABEL,
    SYSTEM_DEFAULT,
    Microphones,
)
from voicetotext.recording.preferences import RecordingPreferences
from voicetotext.settings.theme import HEADING_STYLE

_METER_POLL_MS = 50
_METER_FALL_PER_POLL = 0.06
_LIMIT_MINUTES = (1.0, 60.0)
_WARN_SECONDS = (0.0, 300.0)
_DUCK_CHOICES = (
    ("off", "Leave other audio alone"),
    ("lower", "Lower other apps"),
    ("mute", "Mute other apps"),
)
_NOT_CONNECTED = " (not connected)"


class TkAudioPage(ttk.Frame):
    """Page for the microphone, the sounds and the recording limit.

    The level meter only listens while the page is on screen: an open
    stream on a microphone is a microphone in use, and Windows says so.
    """

    def __init__(self, parent: tkinter.Misc, microphones: Microphones) -> None:
        super().__init__(parent)
        self._microphones = microphones
        self._choice = tkinter.StringVar(master=self, value=_DEFAULT_LABEL)
        self._boost = tkinter.BooleanVar(master=self, value=True)
        self._warn_silent = tkinter.BooleanVar(master=self, value=True)
        self._cues = tkinter.BooleanVar(master=self, value=True)
        self._cue_volume = tkinter.DoubleVar(master=self, value=0.6)
        self._duck = tkinter.StringVar(master=self, value="off")
        self._limit_minutes = tkinter.StringVar(master=self, value="10")
        self._warn_seconds = tkinter.StringVar(master=self, value="30")
        self._preferences = RecordingPreferences()
        self._preferred: list[str] = []
        self._labels: dict[str, str] = {}
        self._names_by_label: dict[str, str] = {_DEFAULT_LABEL: SYSTEM_DEFAULT}
        self._chosen = SYSTEM_DEFAULT
        self._latest_level = 0.0
        self._shown_level = 0.0
        self._poll_id: str | None = None
        self._build()
        self._fill(rescan=False)

    def _build(self) -> None:
        ttk.Label(self, text="Microphone", style=HEADING_STYLE).pack(
            anchor="w", pady=(0, 8),
        )
        row = ttk.Frame(self)
        row.pack(anchor="w", fill="x")
        self._picker = ttk.Combobox(
            row, textvariable=self._choice, state="readonly", width=52,
        )
        self._picker.pack(side="left")
        self._picker.bind("<<ComboboxSelected>>", self._on_picked)
        ttk.Button(row, text="Refresh", command=lambda: self._fill(rescan=True)).pack(
            side="left", padx=(8, 0),
        )
        ttk.Label(self, text="Level").pack(anchor="w", pady=(16, 4))
        self._meter = ttk.Progressbar(self, maximum=1.0, length=400)
        self._meter.pack(anchor="w")
        self._meter_status = ttk.Label(self, text="Speak to test the microphone.")
        self._meter_status.pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(
            self, text="Warn when the microphone goes silent",
            variable=self._warn_silent,
        ).pack(anchor="w", pady=(8, 0))
        self._build_preferred()

        ttk.Label(self, text="Quiet speech", style=HEADING_STYLE).pack(
            anchor="w", pady=(16, 8),
        )
        ttk.Checkbutton(
            self, text="Boost quiet speech before transcribing",
            variable=self._boost,
        ).pack(anchor="w")
        ttk.Label(
            self,
            text=(
                "Helps when you are far from the microphone. Raises the volume\n"
                "up to 10x without distorting it. Saved recordings stay as heard."
            ),
            wraplength=560,
        ).pack(anchor="w", pady=(4, 0))
        self._build_sounds()
        self._build_limit()

    def _build_preferred(self) -> None:
        ttk.Label(self, text="Preferred microphones", style=HEADING_STYLE).pack(
            anchor="w", pady=(16, 4),
        )
        ttk.Label(self, text="The first connected microphone in this list is used.").pack(
            anchor="w", pady=(0, 4),
        )
        row = ttk.Frame(self)
        row.pack(anchor="w", fill="x")
        border = ttk.Frame(row, relief="solid", borderwidth=1)
        border.pack(side="left")
        self._preferred_list = tkinter.Listbox(
            border, height=4, width=52, exportselection=False, activestyle="none",
        )
        self._preferred_list.pack()
        buttons = ttk.Frame(row)
        buttons.pack(side="left", padx=(8, 0), anchor="n")
        for position, (text, command) in enumerate((
            ("Up", lambda: self._move_preferred(-1)),
            ("Down", lambda: self._move_preferred(1)),
            ("Remove", self._remove_preferred),
            ("Add selected", self._add_selected),
        )):
            ttk.Button(buttons, text=text, command=command).grid(
                row=position // 2, column=position % 2, sticky="ew",
                padx=(0, 4), pady=(0, 4),
            )

    def _build_sounds(self) -> None:
        ttk.Label(self, text="Sounds", style=HEADING_STYLE).pack(
            anchor="w", pady=(16, 8),
        )
        row = ttk.Frame(self)
        row.pack(anchor="w", fill="x")
        ttk.Checkbutton(
            row, text="Play start, stop and ready sounds", variable=self._cues,
        ).pack(side="left")
        ttk.Label(row, text="Volume").pack(side="left", padx=(16, 6))
        ttk.Scale(
            row, from_=0.0, to=1.0, variable=self._cue_volume, length=160,
        ).pack(side="left")
        row = ttk.Frame(self)
        row.pack(anchor="w", fill="x", pady=(8, 0))
        ttk.Label(row, text="While recording:").pack(side="left", padx=(0, 8))
        for value, text in _DUCK_CHOICES:
            ttk.Radiobutton(row, text=text, value=value, variable=self._duck).pack(
                side="left", padx=(0, 12),
            )

    def _build_limit(self) -> None:
        ttk.Label(self, text="Recording limit", style=HEADING_STYLE).pack(
            anchor="w", pady=(16, 8),
        )
        row = ttk.Frame(self)
        row.pack(anchor="w", fill="x")
        ttk.Label(row, text="Stop recording after").pack(side="left")
        ttk.Spinbox(
            row, from_=_LIMIT_MINUTES[0], to=_LIMIT_MINUTES[1], increment=1,
            width=4, textvariable=self._limit_minutes,
        ).pack(side="left", padx=6)
        ttk.Label(row, text="minutes. Warn").pack(side="left")
        ttk.Spinbox(
            row, from_=_WARN_SECONDS[0], to=_WARN_SECONDS[1], increment=5,
            width=4, textvariable=self._warn_seconds,
        ).pack(side="left", padx=6)
        ttk.Label(row, text="seconds before.").pack(side="left")

    def recording_preferences(self) -> RecordingPreferences:
        limit = _number_in(self._limit_minutes.get(), *_LIMIT_MINUTES)
        warn = _number_in(self._warn_seconds.get(), *_WARN_SECONDS)
        return replace(
            self._preferences,
            warn_when_silent=self._warn_silent.get(),
            preferred_devices=tuple(self._preferred),
            cues_enabled=self._cues.get(),
            cue_volume=round(min(max(self._cue_volume.get(), 0.0), 1.0), 2),
            duck_other_audio=self._duck.get(),
            max_duration_seconds=(
                self._preferences.max_duration_seconds if limit is None else limit * 60
            ),
            warn_before_cap_seconds=(
                self._preferences.warn_before_cap_seconds if warn is None else warn
            ),
        )

    def set_recording_preferences(self, preferences: RecordingPreferences) -> None:
        self._preferences = preferences
        self._warn_silent.set(preferences.warn_when_silent)
        self._preferred = list(preferences.preferred_devices)
        self._cues.set(preferences.cues_enabled)
        self._cue_volume.set(preferences.cue_volume)
        self._duck.set(preferences.duck_other_audio)
        self._limit_minutes.set(f"{preferences.max_duration_seconds / 60:g}")
        self._warn_seconds.set(f"{preferences.warn_before_cap_seconds:g}")
        self._show_preferred()

    def _add_selected(self) -> None:
        if self._chosen == SYSTEM_DEFAULT:
            return
        full = self._labels.get(self._chosen, self._chosen)
        if full not in self._preferred and self._chosen not in self._preferred:
            self._preferred.append(full)
        self._show_preferred(select=len(self._preferred) - 1)

    def _remove_preferred(self) -> None:
        index = self._selected_preferred()
        if index is None:
            return
        del self._preferred[index]
        self._show_preferred(select=min(index, len(self._preferred) - 1))

    def _move_preferred(self, step: int) -> None:
        index = self._selected_preferred()
        if index is None or not 0 <= index + step < len(self._preferred):
            return
        order = self._preferred
        order[index], order[index + step] = order[index + step], order[index]
        self._show_preferred(select=index + step)

    def _selected_preferred(self) -> int | None:
        selection = self._preferred_list.curselection()
        return int(selection[0]) if selection else None

    def _show_preferred(self, select: int | None = None) -> None:
        connected = {*self._labels, *self._labels.values()}
        self._preferred_list.delete(0, "end")
        for name in self._preferred:
            self._preferred_list.insert(
                "end", name if name in connected else name + _NOT_CONNECTED,
            )
        if select is not None and select >= 0:
            self._preferred_list.selection_set(select)

    def current_device(self) -> str:
        return self._chosen

    def set_device(self, name: str) -> None:
        self._chosen = name
        self._fill(rescan=False)

    def boosts_quiet_speech(self) -> bool:
        return self._boost.get()

    def set_boost_quiet_speech(self, enabled: bool) -> None:
        self._boost.set(enabled)

    def on_shown(self) -> None:
        self._restart_meter()

    def on_hidden(self) -> None:
        self._microphones.stop_meter()
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        self._latest_level = self._shown_level = 0.0
        self._meter["value"] = 0.0

    def destroy(self) -> None:
        self.on_hidden()
        super().destroy()

    def _fill(self, rescan: bool) -> None:
        names = self._microphones.names(rescan=rescan)
        labels = self._microphones.labels(names)
        self._labels = {name: labels.get(name, name) for name in names}
        self._names_by_label = {_DEFAULT_LABEL: SYSTEM_DEFAULT}
        for name in names:
            self._names_by_label[labels.get(name, name)] = name
        # A chosen microphone that is unplugged stays listed, so saving
        # without touching the picker does not quietly forget it.
        if self._chosen and self._chosen not in names:
            self._names_by_label[f"{self._chosen} (not connected)"] = self._chosen
        self._picker["values"] = list(self._names_by_label)
        self._choice.set(self._label_for(self._chosen))
        self._show_preferred()
        if self._poll_id is not None:
            self._restart_meter()

    def _label_for(self, name: str) -> str:
        for label, candidate in self._names_by_label.items():
            if candidate == name:
                return label
        return _DEFAULT_LABEL

    def _on_picked(self, _event: tkinter.Event) -> None:
        self._chosen = self._names_by_label.get(self._choice.get(), SYSTEM_DEFAULT)
        self._restart_meter()

    def _restart_meter(self) -> None:
        opened = self._microphones.start_meter(self._chosen, self._on_level)
        self._meter_status.configure(
            text="Speak to test the microphone."
            if opened else "This microphone could not be opened.",
        )
        if self._poll_id is None:
            self._poll()

    def _on_level(self, level: float) -> None:
        # Audio thread: only store it; the Tk thread picks it up in _poll.
        self._latest_level = max(self._latest_level, level)

    def _poll(self) -> None:
        level, self._latest_level = self._latest_level, 0.0
        self._shown_level = max(level, self._shown_level - _METER_FALL_PER_POLL)
        self._meter["value"] = self._shown_level
        self._poll_id = self.after(_METER_POLL_MS, self._poll)


def _number_in(text: str, low: float, high: float) -> float | None:
    """``text`` as a number held to ``low``..``high``; None if it is not one."""
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return min(max(value, low), high)
