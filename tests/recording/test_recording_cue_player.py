import numpy as np
import pytest

from skryb.core.event_bus import SimpleEventBus
from skryb.core.events import (
    EngineLoadCompletedEvent,
    RecordingCapWarningEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    StateChangedEvent,
)
from skryb.core.state_machine import AppState
from skryb.recording.recording_cue_player import (
    CueBeeper,
    EngineReadyBeeper,
    ReadyCuePlayer,
    RecordingCuePlayer,
    START_FREQUENCY_HZ,
    ready_chime_to_samples,
    tone_to_samples,
    warning_to_samples,
)
from skryb.recording.sounddevice_cue_player import SoundDeviceCuePlayer


class _FakeCuePlayer(RecordingCuePlayer):
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self.warnings = 0

    def play_start(self) -> None:
        self.starts += 1

    def play_stop(self) -> None:
        self.stops += 1

    def play_warning(self) -> None:
        self.warnings += 1


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


class _FakeReadyCuePlayer(ReadyCuePlayer):
    def __init__(self) -> None:
        self.readies = 0

    def play_ready(self) -> None:
        self.readies += 1


class TestReadyChimeToSamples:
    def test_should_return_float32(self):
        assert ready_chime_to_samples().dtype == np.float32

    def test_should_be_two_tones_long(self):
        single = len(tone_to_samples(START_FREQUENCY_HZ))

        assert len(ready_chime_to_samples()) == 2 * single

    def test_should_be_audible(self):
        assert float(np.max(np.abs(ready_chime_to_samples()))) > 0.0


class TestEngineReadyBeeper:
    def test_should_play_ready_when_initial_load_finishes(self):
        bus = SimpleEventBus()
        player = _FakeReadyCuePlayer()
        EngineReadyBeeper(bus, player)

        bus.publish(StateChangedEvent(
            old_state=AppState.LOADING.value, new_state=AppState.IDLE.value,
        ))

        assert player.readies == 1

    def test_should_play_ready_on_model_hot_swap_completion(self):
        bus = SimpleEventBus()
        player = _FakeReadyCuePlayer()
        EngineReadyBeeper(bus, player)

        bus.publish(EngineLoadCompletedEvent(model_name="nemotron-speech-streaming-en-0.6b"))

        assert player.readies == 1

    def test_should_not_play_when_a_failed_load_settles_into_idle(self):
        bus = SimpleEventBus()
        player = _FakeReadyCuePlayer()
        EngineReadyBeeper(bus, player)

        bus.publish(StateChangedEvent(
            old_state=AppState.ERROR.value, new_state=AppState.IDLE.value,
        ))

        assert player.readies == 0

    def test_should_not_play_on_transitions_other_than_load_finished(self):
        bus = SimpleEventBus()
        player = _FakeReadyCuePlayer()
        EngineReadyBeeper(bus, player)

        bus.publish(StateChangedEvent(
            old_state=AppState.RECORDING.value, new_state=AppState.IDLE.value,
        ))
        bus.publish(StateChangedEvent(
            old_state=AppState.IDLE.value, new_state=AppState.RECORDING.value,
        ))

        assert player.readies == 0


class TestRecordingLimitWarningCue:
    def test_a_limit_warning_plays_the_warning_cue_and_nothing_else(self):
        bus = SimpleEventBus()
        player = _FakeCuePlayer()
        CueBeeper(bus, player)

        bus.publish(RecordingCapWarningEvent(remaining_seconds=30.0))

        assert (player.warnings, player.starts, player.stops) == (1, 0, 0)

    def test_the_warning_is_softer_than_the_start_and_stop_cues(self):
        warning = float(np.max(np.abs(warning_to_samples())))
        start = float(np.max(np.abs(tone_to_samples(START_FREQUENCY_HZ))))

        assert 0.0 < warning < start


class _PlayedCues(SoundDeviceCuePlayer):
    def __init__(self, enabled: bool = True, volume: float = 0.6) -> None:
        self.enabled = enabled
        self.volume = volume
        super().__init__(enabled=lambda: self.enabled, volume=lambda: self.volume)
        self.played: list[np.ndarray] = []

    def _emit(self, samples: np.ndarray) -> None:
        self.played.append(samples)


class TestCueSettings:
    def test_the_default_volume_plays_the_cues_exactly_as_loud_as_before(self):
        player = _PlayedCues()

        player.play_start()

        np.testing.assert_array_equal(player.played[0], tone_to_samples(START_FREQUENCY_HZ))

    def test_turning_sounds_off_silences_every_cue_from_the_next_one_on(self):
        player = _PlayedCues()
        player.play_start()

        player.enabled = False
        player.play_stop()
        player.play_ready()
        player.play_warning()

        assert len(player.played) == 1

    def test_volume_scales_each_cue_as_it_is_played(self):
        player = _PlayedCues(volume=0.3)
        player.play_stop()
        player.volume = 1.0
        player.play_stop()

        quiet, loud = (float(np.max(np.abs(samples))) for samples in player.played)
        assert loud == pytest.approx(quiet * (1.0 / 0.3), rel=1e-5)
        assert loud <= 1.0

    @pytest.mark.parametrize("volume", [0.0, -3.0, float("nan")])
    def test_a_volume_of_nothing_or_nonsense_plays_nothing(self, volume):
        player = _PlayedCues(volume=volume)

        player.play_start()

        assert player.played == []
