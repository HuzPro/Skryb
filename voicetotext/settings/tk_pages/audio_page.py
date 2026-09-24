import tkinter
from tkinter import ttk

from voicetotext.recording.input_devices import SYSTEM_DEFAULT, Microphones
from voicetotext.settings.theme import HEADING_STYLE

_DEFAULT_LABEL = "Windows default"
_METER_POLL_MS = 50
_METER_FALL_PER_POLL = 0.06


class TkAudioPage(ttk.Frame):
    """Page for choosing the microphone and testing it.

    The level meter only listens while the page is on screen: an open
    stream on a microphone is a microphone in use, and Windows says so.
    """

    def __init__(self, parent: tkinter.Misc, microphones: Microphones) -> None:
        super().__init__(parent)
        self._microphones = microphones
        self._choice = tkinter.StringVar(master=self, value=_DEFAULT_LABEL)
        self._boost = tkinter.BooleanVar(master=self, value=True)
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

        ttk.Label(self, text="Quiet speech", style=HEADING_STYLE).pack(
            anchor="w", pady=(24, 8),
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
        self._names_by_label = {_DEFAULT_LABEL: SYSTEM_DEFAULT}
        for name in names:
            self._names_by_label[labels.get(name, name)] = name
        # A chosen microphone that is unplugged stays listed, so saving
        # without touching the picker does not quietly forget it.
        if self._chosen and self._chosen not in names:
            self._names_by_label[f"{self._chosen} (not connected)"] = self._chosen
        self._picker["values"] = list(self._names_by_label)
        self._choice.set(self._label_for(self._chosen))
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
