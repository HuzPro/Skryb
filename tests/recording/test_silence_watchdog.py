import threading

import pytest

from tests.fakes.events import RecordingEventBus
from voicetotext.core.events import (
    MicrophoneAudioResumedEvent,
    MicrophoneSilentEvent,
    RecordingPausedEvent,
    RecordingResumedEvent,
    StateChangedEvent,
)
from voicetotext.recording.silence_watchdog import (
    SAMPLE_SECONDS,
    SilenceWatchdog,
    run_every,
)

_SPEECH = 0.02
_ROOM_NOISE = 0.0004


class ManualTicker:
    """Stands in for the sampling thread; the test decides when time passes."""

    def __init__(self) -> None:
        self.action = None
        self.running = False

    def __call__(self, seconds, action):
        assert seconds == SAMPLE_SECONDS
        self.action = action
        self.running = True
        return self._stop

    def _stop(self) -> None:
        self.running = False


class Microphone:
    def __init__(self, enabled: bool = True) -> None:
        self.bus = RecordingEventBus()
        self.now = 50.0
        self.rms = 0.0
        self.enabled = enabled
        self.ticker = ManualTicker()
        SilenceWatchdog(
            self.bus,
            level=lambda: self.rms,
            device_name=lambda: "Microphone (Maono PD300X Microphone)",
            enabled=lambda: self.enabled,
            clock=lambda: self.now,
            ticker=self.ticker,
        )

    def start(self) -> None:
        self.bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))

    def stop(self) -> None:
        self.bus.publish(StateChangedEvent(old_state="recording", new_state="transcribing"))

    def hear(self, rms: float, seconds: float) -> None:
        self.rms = rms
        for _ in range(round(seconds / SAMPLE_SECONDS)):
            self.now += SAMPLE_SECONDS
            if self.ticker.running:
                self.ticker.action()

    def silent_events(self) -> list[MicrophoneSilentEvent]:
        return self.bus.events_of(MicrophoneSilentEvent)

    def notices(self) -> list[str]:
        return [
            type(event).__name__
            for event in self.bus.history
            if isinstance(event, MicrophoneSilentEvent | MicrophoneAudioResumedEvent)
        ]


class TestAMutedMicrophone:
    def test_four_seconds_of_digital_silence_from_the_start_raises_one_warning(self):
        mic = Microphone()
        mic.start()

        mic.hear(0.0, 3.75)
        assert mic.silent_events() == []
        mic.hear(0.0, 60.0)

        [event] = mic.silent_events()
        assert event.device_name == "Microphone (Maono PD300X Microphone)"
        assert event.seconds_silent == pytest.approx(4.0)

    def test_values_just_under_the_floor_still_count_as_silence(self):
        mic = Microphone()
        mic.start()

        mic.hear(9e-6, 4.0)

        assert len(mic.silent_events()) == 1

    def test_the_warning_is_given_again_on_the_next_recording(self):
        mic = Microphone()
        for _ in range(3):
            mic.start()
            mic.hear(0.0, 5.0)
            mic.stop()

        assert len(mic.silent_events()) == 3


class TestAWorkingMicrophone:
    def test_a_minute_of_speech_and_room_noise_never_raises_a_warning(self):
        mic = Microphone()
        mic.start()

        for _ in range(30):
            mic.hear(_SPEECH, 1.0)
            mic.hear(_ROOM_NOISE, 1.0)

        assert mic.notices() == []

    def test_a_short_pause_in_speech_is_not_a_dead_microphone(self):
        mic = Microphone()
        mic.start()
        mic.hear(_SPEECH, 2.0)

        mic.hear(0.0, 5.5)

        assert mic.notices() == []


class TestADropout:
    def test_six_seconds_of_silence_mid_recording_warns_then_says_it_is_back(self):
        mic = Microphone()
        mic.start()
        mic.hear(_SPEECH, 3.0)

        mic.hear(0.0, 6.0)
        mic.hear(_SPEECH, 1.0)

        assert mic.notices() == ["MicrophoneSilentEvent", "MicrophoneAudioResumedEvent"]
        assert mic.silent_events()[0].seconds_silent == pytest.approx(6.0)

    def test_a_second_dropout_warns_again(self):
        mic = Microphone()
        mic.start()
        for _ in range(2):
            mic.hear(_SPEECH, 1.0)
            mic.hear(0.0, 7.0)
        mic.hear(_SPEECH, 1.0)

        assert mic.notices() == [
            "MicrophoneSilentEvent", "MicrophoneAudioResumedEvent",
        ] * 2


class TestPausing:
    def test_paused_time_does_not_count_towards_the_silence(self):
        mic = Microphone()
        mic.start()
        mic.hear(0.0, 3.0)

        mic.bus.publish(RecordingPausedEvent())
        mic.hear(0.0, 120.0)
        mic.bus.publish(RecordingResumedEvent())
        mic.hear(0.0, 0.75)
        assert mic.silent_events() == []
        mic.hear(0.0, 0.25)

        assert len(mic.silent_events()) == 1

    def test_a_pause_between_two_samples_is_left_out_too(self):
        mic = Microphone()
        mic.start()
        mic.hear(0.0, 3.5)
        mic.bus.publish(RecordingPausedEvent())
        mic.now += 40.0
        mic.bus.publish(RecordingResumedEvent())

        mic.hear(0.0, 0.25)

        assert mic.silent_events() == []


class TestStopping:
    def test_sampling_stops_when_the_recording_ends_however_it_ends(self):
        mic = Microphone()
        for ending in ("transcribing", "idle", "error"):
            mic.start()
            mic.bus.publish(StateChangedEvent(old_state="recording", new_state=ending))

            assert mic.ticker.running is False

    def test_a_sample_that_races_the_end_of_the_recording_says_nothing(self):
        mic = Microphone()
        mic.start()
        mic.hear(0.0, 3.75)
        late_sample = mic.ticker.action
        mic.stop()

        mic.now += SAMPLE_SECONDS
        late_sample()

        assert mic.notices() == []

    def test_turned_off_it_does_not_watch_at_all(self):
        mic = Microphone(enabled=False)
        mic.start()

        assert mic.ticker.action is None

    def test_turning_it_off_applies_from_the_next_recording(self):
        mic = Microphone()
        mic.start()
        mic.stop()
        mic.enabled = False

        mic.start()
        mic.hear(0.0, 10.0)

        assert mic.ticker.running is False
        assert mic.notices() == []


def test_the_sampling_thread_runs_until_stopped():
    ticks = threading.Semaphore(0)

    stop = run_every(0.001, ticks.release)
    try:
        assert ticks.acquire(timeout=2.0)
        assert ticks.acquire(timeout=2.0)
    finally:
        stop()
