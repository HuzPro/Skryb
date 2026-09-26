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
from voicetotext.settings.theme import (
    BODY_FONT,
    CARD_BODY_STYLE,
    CARD_MUTED_STYLE,
    CARD_PADDING,
    CARD_TEXT_STYLE,
    CARD_TITLE_STYLE,
    CONTROL_GAP,
    ICON_BUTTON_STYLE,
    PAGE_LINK_BUTTON_STYLE,
    THIN_PROGRESS_STYLE,
    RowGroup,
    card,
    combobox,
    divider,
    page_note,
    section_heading,
    set_card_selected,
    switch,
)

_METER_POLL_MS = 50
_METER_FALL_PER_POLL = 0.06
_METER_LENGTH = 110
_LIMIT_MINUTES = (1.0, 60.0)
_WARN_SECONDS = (0.0, 300.0)
_DUCK_CHOICES = (
    ("off", "Leave other audio alone"),
    ("lower", "Lower other apps"),
    ("mute", "Mute other apps"),
)
_DUCK_LABELS = dict(_DUCK_CHOICES)
_DUCK_VALUES = {label: value for value, label in _DUCK_CHOICES}
_NOT_CONNECTED = " (not connected)"
_SPEAK_TO_TEST = "Speak to test the microphone."
_WILL_NOT_OPEN = "This microphone could not be opened."
_DEFAULT_DESCRIPTION = "Uses the microphone Windows is set to."
_NO_PREFERRED = "None yet. The microphone chosen above is used."


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
        self._duck_label = tkinter.StringVar(master=self, value=_DUCK_LABELS["off"])
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
        self._meter: ttk.Progressbar
        self._meter_status: ttk.Label
        self._duck.trace_add(
            "write",
            lambda *_args: self._duck_label.set(_DUCK_LABELS.get(self._duck.get(), "")),
        )
        self._build()
        self._fill(rescan=False)

    def _build(self) -> None:
        section_heading(self, "Microphone", first=True)
        self._microphone_list = ttk.Frame(self)
        self._microphone_list.pack(fill="x")
        ttk.Button(
            self, text="Look for new microphones", style=PAGE_LINK_BUTTON_STYLE,
            command=lambda: self._fill(rescan=True),
        ).pack(anchor="w", pady=(4, 0))
        self._build_preferred()

        section_heading(self, "Recording")
        rows = RowGroup(self)
        rows.pack(fill="x")
        rows.add(
            "Boost quiet speech",
            "Helps when you are far from the microphone. Raises the volume up to "
            "10x without distorting it. Saved recordings stay as heard.",
            lambda parent: switch(parent, self._boost),
        )
        rows.add(
            "Warn when the microphone goes silent",
            "The pill says so when no sound arrives while you record.",
            lambda parent: switch(parent, self._warn_silent),
        )
        self._build_sounds()
        self._build_limit()

    def _build_preferred(self) -> None:
        section_heading(self, "Preferred microphones")
        page_note(
            self, "When one of these is plugged in, the first one in the list is used.",
        ).pack(anchor="w", pady=(0, CONTROL_GAP))
        self._preferred_card = card(self, padding=(CARD_PADDING, 4))
        self._preferred_card.pack(fill="x")
        ttk.Button(
            self, text="Add the chosen microphone", style=PAGE_LINK_BUTTON_STYLE,
            command=self._add_selected,
        ).pack(anchor="w", pady=(4, 0))

    def _build_sounds(self) -> None:
        section_heading(self, "Sounds")
        rows = RowGroup(self)
        rows.pack(fill="x")
        rows.add(
            "Play sounds",
            "A short tone when recording starts and stops, and when the app is ready.",
            self._sound_controls,
        )
        rows.add(
            "While recording",
            "What happens to music and calls from other apps.",
            lambda parent: self._duck_box(parent),
            fill=True,
        )

    def _sound_controls(self, parent: tkinter.Misc) -> ttk.Frame:
        controls = ttk.Frame(parent, style=CARD_BODY_STYLE)
        ttk.Scale(
            controls, from_=0.0, to=1.0, variable=self._cue_volume, length=110,
            style="Card.Horizontal.TScale",
        ).pack(side="left", padx=(0, 12))
        switch(controls, self._cues).pack(side="left")
        return controls

    def _duck_box(self, parent: tkinter.Misc) -> ttk.Combobox:
        box = combobox(
            parent, textvariable=self._duck_label, values=[label for _, label in _DUCK_CHOICES],
        )
        box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._duck.set(_DUCK_VALUES.get(self._duck_label.get(), "off")),
        )
        return box

    def _build_limit(self) -> None:
        section_heading(self, "Recording limit")
        rows = RowGroup(self)
        rows.pack(fill="x")
        rows.add(
            "Stop recording after",
            "The pill counts down the last seconds before it stops.",
            self._limit_controls,
        )

    def _limit_controls(self, parent: tkinter.Misc) -> ttk.Frame:
        controls = ttk.Frame(parent, style=CARD_BODY_STYLE)
        for variable, bounds, step, unit in (
            (self._limit_minutes, _LIMIT_MINUTES, 1, "min"),
            (self._warn_seconds, _WARN_SECONDS, 5, "s warning"),
        ):
            ttk.Spinbox(
                controls, from_=bounds[0], to=bounds[1], increment=step, width=4,
                textvariable=variable, font=BODY_FONT,
            ).pack(side="left")
            ttk.Label(controls, text=unit, style=CARD_TEXT_STYLE).pack(
                side="left", padx=(6, 14 if unit == "min" else 0),
            )
        return controls

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

    # -- preferred microphones -----------------------------------------------

    def _add_selected(self) -> None:
        if self._chosen == SYSTEM_DEFAULT:
            return
        full = self._labels.get(self._chosen, self._chosen)
        if full not in self._preferred and self._chosen not in self._preferred:
            self._preferred.append(full)
        self._show_preferred()

    def _remove_preferred(self, index: int) -> None:
        if 0 <= index < len(self._preferred):
            del self._preferred[index]
        self._show_preferred()

    def _move_preferred(self, index: int, step: int) -> None:
        if not (0 <= index < len(self._preferred) and 0 <= index + step < len(self._preferred)):
            return
        order = self._preferred
        order[index], order[index + step] = order[index + step], order[index]
        self._show_preferred()

    def preferred_labels(self) -> list[str]:
        connected = self._connected_names()
        return [
            name if name in connected else name + _NOT_CONNECTED
            for name in self._preferred
        ]

    def _connected_names(self) -> set[str]:
        return {*self._labels, *self._labels.values()}

    def _show_preferred(self) -> None:
        for child in self._preferred_card.winfo_children():
            child.destroy()
        if not self._preferred:
            ttk.Label(
                self._preferred_card, text=_NO_PREFERRED, style=CARD_MUTED_STYLE,
            ).pack(anchor="w", pady=10)
            return
        connected = self._connected_names()
        last = len(self._preferred) - 1
        for index, name in enumerate(self._preferred):
            if index:
                divider(self._preferred_card).pack(fill="x")
            self._preferred_row(index, name, name in connected, last).pack(fill="x", pady=6)

    def _preferred_row(self, index: int, name: str, connected: bool, last: int) -> ttk.Frame:
        row = ttk.Frame(self._preferred_card, style=CARD_BODY_STYLE)
        ttk.Label(row, text=f"{index + 1}.", style=CARD_MUTED_STYLE, width=3).pack(side="left")
        ttk.Label(row, text=name, style=CARD_TEXT_STYLE).pack(side="left")
        if not connected:
            ttk.Label(row, text="not connected", style=CARD_MUTED_STYLE).pack(
                side="left", padx=(8, 0),
            )
        for text, command, enabled in (
            ("×", lambda: self._remove_preferred(index), True),
            ("↓", lambda: self._move_preferred(index, 1), index < last),
            ("↑", lambda: self._move_preferred(index, -1), index > 0),
        ):
            button = ttk.Button(row, text=text, style=ICON_BUTTON_STYLE, width=2, command=command)
            button.state(["!disabled"] if enabled else ["disabled"])
            button.pack(side="right")
        return row

    # -- choosing the microphone ---------------------------------------------

    def current_device(self) -> str:
        return self._chosen

    def set_device(self, name: str) -> None:
        self._chosen = name
        self._fill(rescan=False)

    def microphone_labels(self) -> list[str]:
        return list(self._names_by_label)

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
            self._names_by_label[f"{self._chosen}{_NOT_CONNECTED}"] = self._chosen
        self._choice.set(self._label_for(self._chosen))
        self._show_microphones()
        self._show_preferred()
        if self._poll_id is not None:
            self._restart_meter()

    def _label_for(self, name: str) -> str:
        for label, candidate in self._names_by_label.items():
            if candidate == name:
                return label
        return _DEFAULT_LABEL

    def _show_microphones(self) -> None:
        for child in self._microphone_list.winfo_children():
            child.destroy()
        for label in self._names_by_label:
            self._microphone_card(label).pack(fill="x", pady=(0, CONTROL_GAP))

    def _microphone_card(self, label: str) -> ttk.Frame:
        chosen = label == self._choice.get()
        mic_card = card(self._microphone_list)
        mic_card.columnconfigure(0, weight=1)
        mic_card.configure(cursor="hand2")
        title = ttk.Label(mic_card, text=label, style=CARD_TITLE_STYLE)
        title.grid(row=0, column=0, sticky="w")
        description = _DEFAULT_DESCRIPTION if label == _DEFAULT_LABEL else ""
        if label.endswith(_NOT_CONNECTED):
            description = "Not connected right now."
        if chosen:
            self._meter_status = ttk.Label(mic_card, text=_SPEAK_TO_TEST, style="CardAccent.TLabel")
            self._meter_status.grid(row=1, column=0, sticky="w", pady=(2, 0))
            self._meter = ttk.Progressbar(
                mic_card, maximum=1.0, length=_METER_LENGTH, style=THIN_PROGRESS_STYLE,
            )
            self._meter.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))
        elif description:
            ttk.Label(mic_card, text=description, style=CARD_MUTED_STYLE).grid(
                row=1, column=0, sticky="w", pady=(2, 0),
            )
        for widget in (mic_card, *mic_card.winfo_children()):
            widget.bind("<Button-1>", lambda _event, choice=label: self._choose(choice))
        set_card_selected(mic_card, chosen)
        return mic_card

    def _choose(self, label: str) -> None:
        self._choice.set(label)
        self._chosen = self._names_by_label.get(label, SYSTEM_DEFAULT)
        self._show_microphones()
        self._restart_meter()

    def _restart_meter(self) -> None:
        opened = self._microphones.start_meter(self._chosen, self._on_level)
        self._meter_status.configure(
            text=_SPEAK_TO_TEST if opened else _WILL_NOT_OPEN,
            style="SelectedAccent.TLabel" if opened else "SelectedDanger.TLabel",
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
