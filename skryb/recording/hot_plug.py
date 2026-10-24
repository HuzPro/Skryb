"""Notices microphones plugged in while the app runs.

Windows tells every top-level window when any device comes or goes, often
several times for one plug, so the notice waits for the dust to settle,
then looks again off the Tk thread: re-scanning restarts PortAudio, which
takes longer than a frame.
"""

import logging
import tkinter
from collections.abc import Callable, Mapping, Sequence

from skryb.core.event_bus import EventBus
from skryb.core.events import MicrophoneConnectedEvent
from skryb.recording.input_devices import (
    SYSTEM_DEFAULT,
    Microphones,
    preferred_microphone,
)

_logger = logging.getLogger(__name__)

SETTLE_MS = 1000


def plugged_changes(
    before: Sequence[str], after: Sequence[str],
) -> tuple[list[str], list[str]]:
    """The microphones newly connected and those gone, each once, in order."""
    old = {name for name in before if name.strip()}
    new = {name for name in after if name.strip()}
    added = [name for name in dict.fromkeys(after) if name in new - old]
    removed = [name for name in dict.fromkeys(before) if name in old - new]
    return added, removed


def microphone_in_use(
    preferred: Sequence[str],
    chosen: str,
    connected: Sequence[str],
    labels: Mapping[str, str],
) -> str:
    """What the next recording opens: the first connected preferred
    microphone, else the chosen one if connected, else Windows' default."""
    wanted = preferred_microphone(preferred, connected, labels)
    if wanted is not None:
        return wanted
    return chosen if chosen in connected else SYSTEM_DEFAULT


class MicrophoneHotPlug:
    def __init__(
        self,
        root: tkinter.Misc,
        microphones: Microphones,
        preferred: Callable[[], Sequence[str]],
        chosen: Callable[[], str],
        event_bus: EventBus,
        run_in_background: Callable[[Callable[[], None]], None],
    ) -> None:
        self._root = root
        self._microphones = microphones
        self._preferred = preferred
        self._chosen = chosen
        self._event_bus = event_bus
        self._run_in_background = run_in_background
        self._pending: str | None = None

    def devices_changed(self) -> None:
        """Tk thread: some device came or went."""
        if self._pending is not None:
            self._root.after_cancel(self._pending)
        self._pending = self._root.after(SETTLE_MS, self._settled)

    def _settled(self) -> None:
        self._pending = None
        self._run_in_background(self._look_again)

    def _look_again(self) -> None:
        # PortAudio still lists what it saw at its last scan, which is the
        # world before the plug.
        before = self._microphones.names()
        before_labels = self._microphones.labels(before)
        after = self._microphones.names(rescan=True)
        after_labels = self._microphones.labels(after)
        added, removed = plugged_changes(before, after)
        if removed:
            _logger.info("Microphones unplugged: %s", removed)
        preferred, chosen = self._preferred(), self._chosen()
        was = microphone_in_use(preferred, chosen, before, before_labels)
        now = microphone_in_use(preferred, chosen, after, after_labels)
        for name in added:
            label = after_labels.get(name, name)
            if name == now != was:
                self._announce(label, will_be_used=True)
            elif name != chosen and not {name, label} & set(preferred):
                self._announce(label, will_be_used=False)

    def _announce(self, label: str, will_be_used: bool) -> None:
        _logger.info("Microphone plugged in: %r (next recording: %s).", label, will_be_used)
        self._event_bus.publish(
            MicrophoneConnectedEvent(name=label, will_be_used=will_be_used),
        )
