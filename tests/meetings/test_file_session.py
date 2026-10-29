"""A video or audio file transcribed as a meeting, with fake decoding and fake models: no Windows, no GPU."""

import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile

from skryb.meetings.capture import RATE
from skryb.meetings.file_session import FileSession
from skryb.meetings.media_file import MediaFileError
from skryb.meetings.transcript import Event, Transcript
from tests.meetings.fakes import PHRASES, FakeModels, silence, voice

_WAIT_SECONDS = 10.0


def _two_people(turns: int) -> np.ndarray:
    """Two voices taking turns, each turn 2 s of speech and 1 s of quiet."""
    return np.concatenate([np.concatenate([voice(turn % 2, 2), silence(1)]) for turn in range(turns)])


class Rig:
    def __init__(self, tmp_path: Path, audio: np.ndarray | Exception, models: FakeModels | None = None,
                 decode_progress_calls: int = 3, gate: threading.Event | None = None) -> None:
        self.events: list[Event] = []
        self.done = threading.Event()
        self.models = models or FakeModels()
        self.source = tmp_path / "Q3 planning – ٹیم.mp4"
        self.source.write_bytes(b"not read: the decoder is a fake")
        self._audio, self._calls, self._gate = audio, decode_progress_calls, gate
        self.session = FileSession(
            tmp_path / "Meetings" / "2026-10-29 14-05 Q3 planning", datetime(2026, 10, 29, 14, 5),
            source=self.source, models=self.models, vocabulary=tmp_path / "vocabulary.txt",
            publish=self.events.append, on_done=lambda _session: self.done.set(), decode=self._decode,
        )

    def _decode(self, path: Path, progress: Callable[[float], None], cancelled: Callable[[], bool]) -> np.ndarray:
        assert path == self.source
        for call in range(self._calls):
            progress((call + 1) / self._calls)
        if isinstance(self._audio, Exception):
            raise self._audio
        if self._gate is not None:
            self._gate.wait(_WAIT_SECONDS)
        return self._audio

    def run(self, wait: float = _WAIT_SECONDS) -> None:
        self.session.start()
        self.session.begin()
        assert self.done.wait(wait)

    def statuses(self) -> list[tuple[str, float | None]]:
        return [(e["status"], e["progress"]) for e in self.events if e["type"] == "status"]


def test_a_file_with_two_voices_becomes_a_meeting_with_a_line_per_turn(tmp_path: Path) -> None:
    rig = Rig(tmp_path, _two_people(4))

    rig.run()

    saved = Transcript.load(rig.session.folder)
    assert [(e.key, e.text) for e in saved.entries] == [
        ("them:0", PHRASES[0]), ("them:1", PHRASES[1]), ("them:0", PHRASES[0]), ("them:1", PHRASES[1]),
    ]
    sound, rate = soundfile.read(rig.session.folder / "them.flac")
    assert rate == RATE and len(sound) == 12 * RATE
    assert rig.session.status == "Saved" and rig.session.warning is None
    assert rig.models.released == 1


def test_progress_goes_from_reading_through_transcribing_to_saved(tmp_path: Path) -> None:
    rig = Rig(tmp_path, _two_people(4))

    rig.run()

    statuses = rig.statuses()
    order = [status for status, _ in statuses]
    assert order[0] == "Reading the file" and order[-1] == "Saved"
    assert order.index("Loading models") < order.index("Transcribing")
    transcribing = [progress for status, progress in statuses if status == "Transcribing"]
    assert transcribing == sorted(transcribing) and transcribing[-1] == 1.0
    assert all(e["source"] == "Q3 planning – ٹیم.mp4" for e in rig.events if e["type"] == "status")


def test_an_hour_long_file_publishes_progress_in_steps_not_for_every_block(tmp_path: Path) -> None:
    rig = Rig(tmp_path, _two_people(1200), decode_progress_calls=5000)  # an hour, 1,200 turns

    rig.run(wait=120)

    statuses = rig.statuses()
    assert len(Transcript.load(rig.session.folder).entries) == 1200
    assert len([s for s in statuses if s[0] == "Reading the file"]) <= 102
    assert len([s for s in statuses if s[0] == "Transcribing"]) <= 102


def test_cancelling_keeps_the_lines_so_far_and_says_so(tmp_path: Path) -> None:
    rig = Rig(tmp_path, _two_people(400))
    original = rig.models.transcribe
    count = 0

    def slow(audio: np.ndarray, context: str | None) -> str:
        nonlocal count
        count += 1
        if count == 5:
            threading.Thread(target=rig.session.stop).start()
        return original(audio, context)

    rig.models.transcribe = slow
    rig.run()

    lines = Transcript.load(rig.session.folder).entries
    assert 5 <= len(lines) < 400
    assert rig.session.status == "Saved"
    assert rig.session.warning == "Cancelled: the lines so far are saved."
    assert rig.models.released == 1


def test_cancelling_while_the_file_is_read_saves_an_empty_meeting_without_loading_the_models(tmp_path: Path) -> None:
    gate = threading.Event()
    rig = Rig(tmp_path, _two_people(3), gate=gate)
    rig.session.start()
    rig.session.begin()
    stopper = threading.Thread(target=rig.session.stop)
    stopper.start()
    gate.set()
    stopper.join(_WAIT_SECONDS)

    assert rig.done.wait(_WAIT_SECONDS)
    assert rig.models.loaded == []
    assert Transcript.load(rig.session.folder).entries == []


@pytest.mark.parametrize(("audio", "warning"), [
    (MediaFileError("Windows stopped partway through reading the sound."), "Windows stopped partway through reading the sound."),
    (np.zeros(0, np.float32), "Q3 planning – ٹیم.mp4 has no sound to transcribe."),
])
def test_a_file_that_cant_be_transcribed_ends_saved_with_the_reason(tmp_path: Path, audio, warning: str) -> None:
    rig = Rig(tmp_path, audio)

    rig.run()

    assert rig.session.status == "Saved"
    assert rig.session.warning == warning
    assert rig.models.released == 1
    assert rig.events[-1]["warning"] == warning


def test_a_model_that_fails_to_load_is_reported_and_the_sound_is_still_saved(tmp_path: Path) -> None:
    rig = Rig(tmp_path, _two_people(2), models=FakeModels(fails=RuntimeError("hipMalloc: out of memory")))

    rig.run()

    assert rig.session.warning == "Transcription stopped: hipMalloc: out of memory"
    assert (rig.session.folder / "them.flac").exists()


def test_the_state_names_the_file_and_its_progress(tmp_path: Path) -> None:
    rig = Rig(tmp_path, _two_people(2))
    rig.run()

    state = rig.session.state()

    assert state["source"] == "Q3 planning – ٹیم.mp4"
    assert state["status"] == "Saved" and state["id"] == "2026-10-29 14-05 Q3 planning"
