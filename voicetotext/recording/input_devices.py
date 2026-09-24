"""The microphones PortAudio can see, named the way the user picks them.

Devices are chosen by name, not index: PortAudio renumbers them whenever a
device is plugged in or out, so a saved index would quietly point at a
different microphone. Only the default host API is listed, because every
API repeats the same hardware under its own name and not all of them can
open at the 16 kHz the recorder asks for.
"""

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

_logger = logging.getLogger(__name__)

SYSTEM_DEFAULT = ""

# Host APIs' own names for "whatever Windows' default is", which the
# picker already offers as the default.
_DEFAULT_ALIASES = frozenset({
    "Microsoft Sound Mapper - Input",
    "Primary Sound Capture Driver",
})

# Returns ``sd.query_devices()`` and the default host API's index.
DeviceQuery = Callable[[], tuple[Sequence[dict[str, Any]], int]]


def query_portaudio() -> tuple[Sequence[dict[str, Any]], int]:
    import sounddevice as sd

    return sd.query_devices(), sd.default.hostapi


def refresh_portaudio() -> None:
    """Re-scan the hardware; PortAudio only looks once, at start-up.

    Restarting PortAudio closes every open stream, so call it only while
    nothing is recording or playing.
    """
    import sounddevice as sd

    try:
        sd._terminate()  # the only way to see hot-plugged devices
        sd._initialize()
    except Exception:  # a failed re-scan keeps the old list
        _logger.warning("Could not re-scan audio devices.", exc_info=True)


def input_device_names(query: DeviceQuery = query_portaudio) -> list[str]:
    """Every microphone on the default host API, once each, in PortAudio order."""
    devices, host_api = query()
    names: list[str] = []
    for device in devices:
        name = str(device.get("name", "")).strip()
        if (
            name
            and name not in _DEFAULT_ALIASES
            and device.get("hostapi") == host_api
            and device.get("max_input_channels", 0) > 0
            and name not in names
        ):
            names.append(name)
    return names


def full_names(names: Sequence[str], query: DeviceQuery = query_portaudio) -> dict[str, str]:
    """Each name as the user would recognise it.

    The default host API on Windows (MME) cuts names at 31 characters;
    another API usually lists the same microphone in full, and the full
    name is the longest one that starts with the cut one.
    """
    devices, _host_api = query()
    inputs = [
        str(device.get("name", "")).strip()
        for device in devices
        if device.get("max_input_channels", 0) > 0
    ]
    return {
        name: max((full for full in inputs if full.startswith(name)), key=len, default=name)
        for name in names
    }


def input_device_index(
    name: str, query: DeviceQuery = query_portaudio,
) -> int | None:
    """PortAudio's index for the microphone called ``name``, or None."""
    if name == SYSTEM_DEFAULT:
        return None
    devices, host_api = query()
    for index, device in enumerate(devices):
        if (
            str(device.get("name", "")).strip() == name
            and device.get("hostapi") == host_api
            and device.get("max_input_channels", 0) > 0
        ):
            return index
    return None


def _never_busy() -> bool:
    return False


class Microphones:
    """What the settings page and the tray menu need to offer a microphone.

    ``is_busy`` guards the re-scan: restarting PortAudio mid-recording
    would cut the recording off, so a busy app lists what it already knows.
    """

    def __init__(
        self,
        is_busy: Callable[[], bool] = _never_busy,
        query: DeviceQuery = query_portaudio,
        refresh: Callable[[], None] = refresh_portaudio,
    ) -> None:
        self._is_busy = is_busy
        self._query = query
        self._refresh = refresh
        self._meter: Any = None

    def labels(self, names: Sequence[str]) -> dict[str, str]:
        try:
            return full_names(names, self._query)
        except Exception:  # fall back to the names as the device reports them
            _logger.debug("Looking up full microphone names failed.", exc_info=True)
            return {name: name for name in names}

    def names(self, rescan: bool = False) -> list[str]:
        if rescan and not self._is_busy() and self._meter is None:
            self._refresh()
        try:
            return input_device_names(self._query)
        except Exception:  # an audio stack that will not answer lists nothing
            _logger.warning("Listing microphones failed.", exc_info=True)
            return []

    def start_meter(self, name: str, on_level: Callable[[float], None]) -> bool:
        """Report the loudness of ``name`` (0..1) until ``stop_meter``."""
        self.stop_meter()
        import sounddevice as sd

        def callback(indata, _frames, _time, _status) -> None:
            rms = float((indata.astype("float64") ** 2).mean() ** 0.5)
            on_level(loudness(rms))

        try:
            self._meter = sd.InputStream(
                device=input_device_index(name, self._query),
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=callback,
            )
            self._meter.start()
        except Exception:  # a microphone that will not open just shows no level
            _logger.warning("Opening the level meter on %r failed.", name, exc_info=True)
            self._meter = None
            return False
        return True

    def stop_meter(self) -> None:
        meter, self._meter = self._meter, None
        if meter is None:
            return
        try:
            meter.stop()
            meter.close()
        except Exception:
            _logger.debug("Closing the level meter failed.", exc_info=True)


_METER_FLOOR_DB = -60.0


def loudness(rms: float) -> float:
    """``rms`` on a decibel scale from 0 (-60 dBFS or quieter) to 1 (full scale)."""
    if not math.isfinite(rms) or rms <= 0:
        return 0.0
    decibels = 20 * math.log10(rms)
    return min(max(1 - decibels / _METER_FLOOR_DB, 0.0), 1.0)
