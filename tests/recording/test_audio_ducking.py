import threading
import time

import numpy as np
import pytest

from skryb.core.event_bus import SimpleEventBus
from skryb.core.events import (
    RecordingPausedEvent,
    RecordingResumedEvent,
    StateChangedEvent,
)
from skryb.recording.audio_ducking import (
    AudioDucking,
    AudioSession,
    SessionDucker,
)

_OUR_PID = 4242


class Mixer:
    """Windows' volume mixer as the ducker sees it: sessions by key, volumes
    kept as 32-bit floats the way Core Audio stores them."""

    def __init__(self) -> None:
        self.volumes: dict[str, float] = {}
        self.pids: dict[str, int] = {}
        self.closed: set[str] = set()
        self.lock = threading.Lock()

    def play(self, key: str, pid: int, volume: float) -> None:
        with self.lock:
            self.pids[key] = pid
            self.volumes[key] = float(np.float32(volume))

    def user_sets(self, key: str, volume: float) -> None:
        self.play(key, self.pids[key], volume)

    def close(self, key: str) -> None:
        self.closed.add(key)

    def sessions(self) -> list[AudioSession]:
        with self.lock:
            return [self._session(key) for key in list(self.volumes)]

    def _session(self, key: str) -> AudioSession:
        def volume() -> float:
            self._check_open(key)
            return self.volumes[key]

        def set_volume(level: float) -> None:
            self._check_open(key)
            self.volumes[key] = float(np.float32(level))

        return AudioSession(key, self.pids[key], volume, set_volume)

    def _check_open(self, key: str) -> None:
        if key in self.closed:
            raise OSError("AUDCLNT_E_DEVICE_INVALIDATED")


@pytest.fixture
def mixer() -> Mixer:
    mixer = Mixer()
    mixer.play("spotify", 1000, 0.8)
    mixer.play("chrome-tab-1", 1001, 1.0)
    mixer.play("chrome-tab-2", 1001, 0.33)
    mixer.play("system-sounds", 0, 0.5)
    mixer.play("skryb-cues", _OUR_PID, 1.0)
    return mixer


class TestSessionDucker:
    def test_lowering_turns_every_other_app_down_to_the_share_it_keeps(self, mixer):
        ducker = SessionDucker(mixer.sessions, _OUR_PID)

        ducker.duck(0.2)

        assert mixer.volumes["spotify"] == pytest.approx(0.16, abs=1e-6)
        assert mixer.volumes["chrome-tab-1"] == pytest.approx(0.2, abs=1e-6)
        assert mixer.volumes["chrome-tab-2"] == pytest.approx(0.066, abs=1e-6)
        assert mixer.volumes["system-sounds"] == pytest.approx(0.1, abs=1e-6)

    def test_our_own_sounds_are_never_lowered(self, mixer):
        ducker = SessionDucker(mixer.sessions, _OUR_PID)

        ducker.duck(0.0)

        assert mixer.volumes["skryb-cues"] == 1.0

    def test_restoring_puts_every_volume_back_exactly(self, mixer):
        before = dict(mixer.volumes)
        ducker = SessionDucker(mixer.sessions, _OUR_PID)

        ducker.duck(0.2)
        ducker.restore()

        assert mixer.volumes == before

    def test_a_volume_the_user_moved_while_lowered_is_left_where_they_put_it(self, mixer):
        ducker = SessionDucker(mixer.sessions, _OUR_PID)
        ducker.duck(0.2)

        mixer.user_sets("spotify", 0.45)
        ducker.restore()

        assert mixer.volumes["spotify"] == pytest.approx(0.45)
        assert mixer.volumes["chrome-tab-1"] == pytest.approx(1.0)

    def test_an_app_that_starts_playing_mid_recording_is_lowered_too(self, mixer):
        ducker = SessionDucker(mixer.sessions, _OUR_PID)
        ducker.duck(0.0)

        mixer.play("teams-call", 2000, 0.9)
        ducker.duck_new_sessions()
        ducker.duck_new_sessions()
        ducker.restore()

        assert mixer.volumes["teams-call"] == pytest.approx(0.9)

    def test_polling_again_never_lowers_an_already_lowered_app_twice(self, mixer):
        ducker = SessionDucker(mixer.sessions, _OUR_PID)
        ducker.duck(0.5)

        for _ in range(5):
            ducker.duck_new_sessions()

        assert mixer.volumes["spotify"] == pytest.approx(0.4)

    def test_an_app_that_closed_meanwhile_does_not_stop_the_rest_coming_back(self, mixer):
        ducker = SessionDucker(mixer.sessions, _OUR_PID)
        ducker.duck(0.2)

        mixer.close("chrome-tab-1")
        ducker.restore()

        assert mixer.volumes["spotify"] == pytest.approx(0.8)
        assert mixer.volumes["system-sounds"] == pytest.approx(0.5)

    def test_hundreds_of_sessions_come_back_as_they_were(self):
        mixer = Mixer()
        for index in range(600):
            mixer.play(f"app-{index}", 3000 + index % 50, (index % 101) / 100)
        before = dict(mixer.volumes)
        ducker = SessionDucker(mixer.sessions, _OUR_PID)

        ducker.duck(0.2)
        ducker.restore()

        assert mixer.volumes == before

    def test_nothing_is_polled_once_restored(self, mixer):
        ducker = SessionDucker(mixer.sessions, _OUR_PID)
        ducker.duck(0.2)
        ducker.restore()

        mixer.play("late-app", 3000, 0.7)
        ducker.duck_new_sessions()

        assert ducker.active is False
        assert mixer.volumes["late-app"] == pytest.approx(0.7)


def eventually(condition) -> None:
    """The ducker works on a thread of its own; give it a moment."""
    deadline = time.monotonic() + 5.0
    while not condition():
        assert time.monotonic() < deadline, "the ducker never got there"
        time.sleep(0.005)


class Recording:
    def __init__(self, mixer: Mixer, mode: str = "lower", level: float = 0.2) -> None:
        self.bus = SimpleEventBus()
        self.mode = mode
        self.level = level
        self.listings = 0
        self.failures: list[Exception] = []
        self.mixer = mixer
        self.ducking = AudioDucking(
            self.bus,
            mode=lambda: self.mode,
            level=lambda: self.level,
            list_sessions=self._list,
            own_pid=_OUR_PID,
        )

    def _list(self) -> list[AudioSession]:
        self.listings += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.mixer.sessions()

    def state(self, old: str, new: str) -> None:
        self.bus.publish(StateChangedEvent(old_state=old, new_state=new))


def _lowered(mixer: Mixer, keep: float = 0.2) -> bool:
    return mixer.volumes["spotify"] == pytest.approx(0.8 * keep, abs=1e-6)


def _restored(mixer: Mixer) -> bool:
    return mixer.volumes["spotify"] == pytest.approx(0.8)


class TestDuckingDuringARecording:
    def test_other_apps_are_lowered_while_recording_and_restored_after(self, mixer):
        recording = Recording(mixer)

        recording.state("idle", "recording")
        eventually(lambda: _lowered(mixer))
        recording.state("recording", "transcribing")

        eventually(lambda: _restored(mixer))

    def test_mute_takes_other_apps_to_silence_but_not_our_own_sounds(self, mixer):
        recording = Recording(mixer, mode="mute")

        recording.state("idle", "recording")

        eventually(lambda: mixer.volumes["spotify"] == 0.0)
        assert mixer.volumes["skryb-cues"] == 1.0
        recording.ducking.dispose()

    def test_pausing_gives_other_apps_their_volume_back_until_resumed(self, mixer):
        recording = Recording(mixer)
        recording.state("idle", "recording")
        eventually(lambda: _lowered(mixer))

        recording.bus.publish(RecordingPausedEvent())
        eventually(lambda: _restored(mixer))
        recording.bus.publish(RecordingResumedEvent())

        eventually(lambda: _lowered(mixer))
        recording.ducking.dispose()

    def test_leave_other_audio_alone_never_touches_the_mixer(self, mixer):
        recording = Recording(mixer, mode="off")

        recording.state("idle", "recording")
        recording.bus.publish(RecordingPausedEvent())
        recording.bus.publish(RecordingResumedEvent())
        recording.state("recording", "idle")
        recording.ducking.dispose()

        assert recording.listings == 0

    def test_quitting_mid_recording_puts_every_volume_back(self, mixer):
        before = dict(mixer.volumes)
        recording = Recording(mixer, mode="mute")
        recording.state("idle", "recording")

        recording.ducking.dispose()

        assert mixer.volumes == before

    def test_a_setting_changed_between_recordings_applies_to_the_next(self, mixer):
        recording = Recording(mixer, mode="off")
        recording.state("idle", "recording")
        recording.state("recording", "idle")

        recording.mode = "lower"
        recording.level = 0.5
        recording.state("idle", "recording")

        eventually(lambda: _lowered(mixer, keep=0.5))
        recording.ducking.dispose()

    def test_a_broken_audio_stack_does_not_stop_the_next_recording_ducking(self, mixer):
        recording = Recording(mixer)
        recording.failures.append(OSError("CoCreateInstance failed"))

        recording.state("idle", "recording")
        recording.state("recording", "idle")
        recording.state("idle", "recording")

        eventually(lambda: _lowered(mixer))
        recording.ducking.dispose()
        assert _restored(mixer)
