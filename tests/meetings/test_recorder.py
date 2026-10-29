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
        self._capture_fails = capture_fails
        self.recorder = MeetingRecorder(
            settings=lambda: self.settings,
            missing_models=lambda: list(missing or []),
            make_models=self._models,
            make_capture=self._capture,
            publish=self.events.append,
            on_change=self.changes.append,
            clock=lambda: datetime(2026, 10, 29, 14, 5, 33),
        )

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


def test_stopping_saves_the_meeting_and_keeps_it_on_show(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.recorder.start()
    assert rig.captures[0].them.done.wait(_WAIT_SECONDS)

    rig.recorder.stop()

    assert not rig.recorder.is_recording()
    assert rig.changes == [True, False]
    assert rig.events[-1] == {"type": "saved", "id": "2026-10-29 14-05"}
    live = rig.recorder.live()
    assert live is not None and live["recording"] is False and live["status"] == "Saved"
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
