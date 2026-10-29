import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from skryb.meetings.recorder import MeetingError, MeetingRecorder
from skryb.meetings.storage import MeetingSettings
from skryb.meetings.transcript import Event
from tests.meetings.fakes import FakeCapture, FakeModels, silence, voice

_WAIT_SECONDS = 10.0


class Rig:
    def __init__(self, tmp_path: Path, *, missing: list[str] | None = None, capture_fails: Exception | None = None) -> None:
        self.root = tmp_path / "Meetings"
        self.events: list[Event] = []
        self.changes: list[bool] = []
        self.captures: list[FakeCapture] = []
        self.models: list[FakeModels] = []
        self.settings = MeetingSettings(self.root, record_microphone=True, language="ur")
        self.asked_for: list[MeetingSettings] = []
        self.models_checked_for: list[str] = []
        self._missing = list(missing or [])
        self._capture_fails = capture_fails
        self.recorder = MeetingRecorder(
            settings=lambda: self.settings,
            missing_models=self._missing_models,
            make_models=self._models,
            make_capture=self._capture,
            publish=self.events.append,
            on_change=self.changes.append,
            clock=lambda: datetime(2026, 10, 29, 14, 5, 33),
        )

    def _missing_models(self, language: str) -> list[str]:
        self.models_checked_for.append(language)
        return self._missing

    def _models(self, settings: MeetingSettings) -> FakeModels:
        self.asked_for.append(settings)
        self.models.append(FakeModels())
        return self.models[-1]

    def _capture(self, record_microphone: bool) -> FakeCapture:
        capture = FakeCapture(np.concatenate([voice(0, 2), silence(1)]), fails=self._capture_fails)
        self.captures.append(capture)
        return capture


def test_starting_records_into_a_new_folder_named_by_the_minute_and_tells_the_tray(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    rig.recorder.start()

    assert rig.recorder.is_recording()
    assert rig.recorder.recording_started() == datetime(2026, 10, 29, 14, 5, 33)
    assert (rig.root / "2026-10-29 14-05" / "them.flac").exists()
    assert (rig.root / "vocabulary.txt").exists()
    assert rig.changes == [True]
    assert rig.events[-1]["type"] == "started"
    assert rig.asked_for[0].language == "ur"
    rig.recorder.stop()


def test_starting_twice_records_one_meeting(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.recorder.start()
    rig.recorder.start()

    assert len(rig.captures) == 1
    rig.recorder.stop()


def test_a_saved_meeting_leaves_the_live_page_ready_for_the_next_one(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.recorder.start()
    assert rig.captures[0].them.done.wait(_WAIT_SECONDS)

    rig.recorder.stop()

    assert not rig.recorder.is_recording()
    assert rig.changes == [True, False]
    assert rig.events[-1] == {"type": "saved", "id": "2026-10-29 14-05"}
    assert rig.recorder.live() is None
    # Renaming a speaker right after saving still reaches the transcript it is holding.
    assert rig.recorder.live_session("2026-10-29 14-05") is not None
    assert rig.recorder.live_session("2026-10-16 Ops Onboarding") is None


def test_without_the_models_a_meeting_does_not_start_and_says_which_to_download(tmp_path: Path) -> None:
    rig = Rig(tmp_path, missing=["Speaker labels", "Qwen3-ASR speech-to-text"])

    with pytest.raises(MeetingError, match="Download the meeting models first: Speaker labels, Qwen3-ASR speech-to-text."):
        rig.recorder.start()

    assert not rig.recorder.is_recording()
    assert not rig.root.exists()
    assert rig.changes == []


def test_when_the_speakers_cannot_be_opened_the_reason_is_given_and_nothing_records(tmp_path: Path) -> None:
    rig = Rig(tmp_path, capture_fails=RuntimeError("No speaker output found to record."))

    with pytest.raises(MeetingError, match="Could not start recording: No speaker output found to record."):
        rig.recorder.start()

    assert not rig.recorder.is_recording()
    assert rig.recorder.live() is None
    assert not (rig.root / "2026-10-29 14-05").exists()


def test_stopping_when_nothing_records_does_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.recorder.stop()
    assert rig.events == [] and rig.changes == []


def test_the_tray_toggle_starts_then_stops(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.recorder.toggle()
    assert rig.recorder.is_recording()
    rig.recorder.toggle()
    assert not rig.recorder.is_recording()


def test_a_start_and_a_stop_from_two_places_at_once_never_leave_two_meetings(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    threads = [threading.Thread(target=rig.recorder.toggle) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_WAIT_SECONDS)

    starts = rig.changes.count(True)
    recording = rig.recorder.is_recording()
    assert rig.changes == ([True, False] * starts)[:2 * starts - recording]
    assert len(rig.captures) == starts
    rig.recorder.stop()


def test_quitting_while_recording_closes_the_files_without_waiting_for_the_transcript(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.recorder.start()

    rig.recorder.dispose()

    assert rig.captures[0].stopped.wait(_WAIT_SECONDS)


class FileRig(Rig):
    """The recorder with a fake file reader: *audio* is what any file decodes to."""

    def __init__(self, tmp_path: Path, audio: np.ndarray | None = None, unreadable: str | None = None, **kwargs) -> None:
        super().__init__(tmp_path, **kwargs)
        from skryb.meetings.media_file import MediaFileError

        self.audio = np.concatenate([voice(0, 2), silence(1), voice(1, 2), silence(1)]) if audio is None else audio
        self.gate = threading.Event()
        self.gate.set()
        self.video_ok = tmp_path / "Sprint review.mp4"
        self.video_ok.write_bytes(b"fake")

        def check(path: Path) -> None:
            if unreadable:
                raise MediaFileError(unreadable)

        def decode(path: Path, progress, cancelled) -> np.ndarray:
            self.gate.wait(_WAIT_SECONDS)
            return self.audio

        self.recorder._check_file = check
        self.recorder._decode_file = decode

    def wait_saved(self) -> None:
        deadline = threading.Event()
        for _ in range(int(_WAIT_SECONDS * 20)):
            if any(e["type"] == "saved" for e in self.events):
                return
            deadline.wait(0.05)
        raise AssertionError("the file never finished")


def test_a_file_becomes_a_meeting_named_after_it_in_the_language_asked_for(tmp_path: Path) -> None:
    rig = FileRig(tmp_path)

    rig.recorder.transcribe_file(rig.video_ok, "en")
    rig.wait_saved()

    folder = rig.root / "2026-10-29 14-05 Sprint review"
    assert (folder / "them.flac").exists() and (folder / "transcript.json").exists()
    assert rig.asked_for[0].language == "en"
    assert rig.events[0]["type"] == "started" and rig.events[0]["source"] == "Sprint review.mp4"
    assert rig.events[-1] == {"type": "saved", "id": "2026-10-29 14-05 Sprint review"}
    assert not rig.recorder.is_recording() and rig.recorder.live() is None
    assert rig.changes == []  # the tray's recording light stays off


def test_while_a_file_is_transcribed_a_meeting_cant_start_from_the_window_or_the_tray(tmp_path: Path) -> None:
    rig = FileRig(tmp_path)
    rig.gate.clear()
    rig.recorder.transcribe_file(rig.video_ok, "ur")

    for start in (rig.recorder.start, rig.recorder.toggle, lambda: rig.recorder.transcribe_file(rig.video_ok, "en")):
        with pytest.raises(MeetingError, match="Skryb is transcribing a file. Cancel it on the Meetings page first."):
            start()

    assert rig.recorder.transcribing_file()
    rig.gate.set()
    rig.wait_saved()


def test_a_file_cant_be_transcribed_while_a_meeting_records(tmp_path: Path) -> None:
    rig = FileRig(tmp_path)
    rig.recorder.start()

    with pytest.raises(MeetingError, match="A meeting is being recorded. Stop it before transcribing a file."):
        rig.recorder.transcribe_file(rig.video_ok, "en")

    rig.recorder.stop()


def test_cancelling_a_file_saves_it_without_touching_the_tray(tmp_path: Path) -> None:
    rig = FileRig(tmp_path)
    rig.gate.clear()
    rig.recorder.transcribe_file(rig.video_ok, "en")

    threading.Timer(0.2, rig.gate.set).start()
    rig.recorder.stop()

    assert rig.events[-1]["type"] == "saved"
    assert not rig.recorder.is_recording()
    assert rig.changes == []


@pytest.mark.parametrize(("language", "unreadable", "missing", "message"), [
    ("fr", None, None, "Pick English or Urdu for the file."),
    ("en", "Skryb can't read .ogg files yet.", None, "Skryb can't read .ogg files yet."),
    ("en", None, ["Speaker labels"], "Download the meeting models first: Speaker labels."),
])
def test_a_file_that_cant_start_says_why_and_leaves_no_folder(
    tmp_path: Path, language: str, unreadable: str | None, missing: list[str] | None, message: str,
) -> None:
    rig = FileRig(tmp_path, unreadable=unreadable, missing=missing)

    with pytest.raises(MeetingError, match=message):
        rig.recorder.transcribe_file(rig.video_ok, language)

    assert not rig.root.exists() or not any(rig.root.iterdir())
    assert rig.events == []


def test_a_meeting_checks_the_models_its_language_needs(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    rig.recorder.start()
    rig.recorder.stop()

    assert rig.models_checked_for == ["ur"]


def test_a_file_checks_the_models_the_files_language_needs_not_the_meeting_settings(tmp_path: Path) -> None:
    rig = FileRig(tmp_path, missing=["Roman Urdu speech-to-text"])
    rig.settings = MeetingSettings(rig.root, record_microphone=True, language="en")

    with pytest.raises(MeetingError, match="Download the meeting models first: Roman Urdu speech-to-text."):
        rig.recorder.transcribe_file(rig.video_ok, "ur")

    assert rig.models_checked_for == ["ur"]
