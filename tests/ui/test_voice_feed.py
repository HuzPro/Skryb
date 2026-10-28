"""The voice level streamed to the Settings window while recording."""

from skryb.core.event_bus import SimpleEventBus
from skryb.core.events import StateChangedEvent
from skryb.ui.voice_feed import VoiceFeed


class ManualTicker:
    """Stands in for the sampling thread: the test calls the action itself."""

    def __init__(self) -> None:
        self.action = None
        self.running = False

    def __call__(self, _seconds, action):
        self.action = action
        self.running = True

        def stop() -> None:
            self.running = False

        return stop


def _feed(level: float):
    bus = SimpleEventBus()
    published: list[tuple[str, object]] = []
    ticker = ManualTicker()
    rms = [level]
    VoiceFeed(bus, lambda name, data: published.append((name, data)), lambda: rms[0], ticker=ticker)
    return bus, published, ticker, rms


def _record(bus) -> None:
    bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))


def test_nothing_is_streamed_until_a_recording_starts():
    _bus, published, ticker, _rms = _feed(0.1)

    assert ticker.running is False
    assert published == []


def test_a_recording_streams_voice_levels_between_zero_and_one():
    bus, published, ticker, rms = _feed(0.0)
    _record(bus)

    for loudness in (0.0, 0.001, 0.05, 0.3, 1.0, float("nan"), 0.02):
        rms[0] = loudness
        ticker.action()

    levels = [data["level"] for name, data in published if name == "voice"]
    assert len(levels) == 7
    assert all(0.0 <= level <= 1.0 for level in levels)
    assert max(levels) > 0.5


def test_speaking_reads_louder_than_silence():
    bus, published, ticker, rms = _feed(0.0)
    _record(bus)
    for _ in range(40):
        ticker.action()
    quiet = published[-1][1]["level"]

    rms[0] = 0.05
    ticker.action()

    assert published[-1][1]["level"] > quiet


def test_stopping_the_recording_stops_the_stream_and_drops_the_bars():
    bus, published, ticker, _rms = _feed(0.2)
    _record(bus)
    ticker.action()

    bus.publish(StateChangedEvent(old_state="recording", new_state="transcribing"))

    assert ticker.running is False
    assert published[-1] == ("voice", {"level": 0.0})


def test_other_state_changes_leave_the_stream_alone():
    bus, published, ticker, _rms = _feed(0.2)

    bus.publish(StateChangedEvent(old_state="transcribing", new_state="idle"))

    assert ticker.running is False
    assert published == []
