import numpy as np

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    RecordingStartedEvent,
    RecordingStoppedEvent,
)
from voicetotext.recording.recording_cue_player import (
    CueBeeper,
    RecordingCuePlayer,
    tone_to_samples,
)


class _FakeCuePlayer(RecordingCuePlayer):
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    def play_start(self) -> None:
        self.starts += 1

    def play_stop(self) -> None:
        self.stops += 1


class TestToneToSamples:
    def test_should_return_float32(self):
        assert tone_to_samples(880.0).dtype == np.float32

    def test_should_have_the_requested_length(self):
        samples = tone_to_samples(880.0, seconds=0.1, sample_rate=16000)

        assert len(samples) == 1600

    def test_should_be_audible(self):
        assert float(np.max(np.abs(tone_to_samples(880.0)))) > 0.0

    def test_should_stay_within_unit_amplitude(self):
        assert float(np.max(np.abs(tone_to_samples(880.0)))) <= 1.0

    def test_should_differ_between_frequencies(self):
        assert not np.array_equal(tone_to_samples(880.0), tone_to_samples(440.0))


class TestCueBeeper:
    def test_should_play_start_cue_on_recording_started(self):
        bus = SimpleEventBus()
        player = _FakeCuePlayer()
        CueBeeper(bus, player)

        bus.publish(RecordingStartedEvent())

        assert player.starts == 1
        assert player.stops == 0

    def test_should_play_stop_cue_on_recording_stopped(self):
        bus = SimpleEventBus()
        player = _FakeCuePlayer()
        CueBeeper(bus, player)

        bus.publish(RecordingStoppedEvent())

        assert player.stops == 1
        assert player.starts == 0
