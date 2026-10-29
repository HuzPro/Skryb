"""A whole meeting through the real session, on fake devices and fake models."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile

from skryb.meetings.capture import RATE
from skryb.meetings.session import Session
from skryb.meetings.transcript import Event
from tests.meetings.fakes import PHRASES, FakeCapture, FakeModels, silence, voice

_WAIT_SECONDS = 10.0
STARTED = datetime(2026, 10, 29, 14, 5)


def _run(
    tmp_path: Path, capture: FakeCapture, models: FakeModels, *, record_microphone: bool = True,
) -> tuple[Session, list[Event]]:
    events: list[Event] = []
    session = Session(
        tmp_path / "2026-10-29 14-05", STARTED,
        capture=capture, models=models, vocabulary=tmp_path / "vocabulary.txt",
        publish=events.append, record_microphone=record_microphone,
    )
    session.start()
    assert capture.them.done.wait(_WAIT_SECONDS)
    session.stop()
    return session, events


def _saved(session: Session) -> dict[str, object]:
    return json.loads((session.folder / "transcript.json").read_text(encoding="utf-8"))


def _call() -> FakeCapture:
    """Two remote voices taking turns, and the user answering between them."""
    them = np.concatenate([voice(0, 3), silence(2.5), voice(1, 2.5), silence(1)])
    me = np.concatenate([silence(3.5), voice(2, 1.5), silence(4)])
    return FakeCapture(them, me)


def test_a_meeting_writes_each_voice_as_its_own_speaker_and_the_microphone_as_me(tmp_path: Path) -> None:
    session, _events = _run(tmp_path, _call(), FakeModels())

    saved = _saved(session)
    lines = [(e["track"], e["speaker"], e["text"]) for e in saved["entries"]]  # type: ignore[index]
    assert lines == [("them", 0, PHRASES[0]), ("me", 0, PHRASES[2]), ("them", 1, PHRASES[1])]
    assert saved["names"] == {"me:0": "Me", "them:0": "Speaker 1", "them:1": "Speaker 2"}
    assert saved["started"] == "2026-10-29T14:05:00"
    markdown = (session.folder / "transcript.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Meeting 2026-10-29 14:05")
    assert "**Me** [00:00:03]" in markdown


def test_both_tracks_are_saved_as_16k_flac_as_long_as_the_meeting(tmp_path: Path) -> None:
    session, _events = _run(tmp_path, _call(), FakeModels())

    for track in ("them", "me"):
        audio, rate = soundfile.read(session.folder / f"{track}.flac", dtype="float32")
        assert rate == RATE
        assert len(audio) == 9 * RATE
    them, _rate = soundfile.read(session.folder / "them.flac", dtype="float32")
    assert abs(them[RATE] - 0.2) < 1e-3


def test_the_window_hears_the_status_change_from_loading_to_recording_to_saved(tmp_path: Path) -> None:
    _session, events = _run(tmp_path, _call(), FakeModels())

    statuses = [e["status"] for e in events if e["type"] == "status"]
    assert statuses[0] in ("Recording", "Loading models")
    assert "Loading models" in statuses
    assert statuses[-2:] == ["Finishing", "Saved"]
    assert all(e["id"] == "2026-10-29 14-05" for e in events)
    assert {e["type"] for e in events} >= {"status", "entry"}


def test_stopping_waits_for_every_line_already_heard_and_then_lets_go_of_the_models(tmp_path: Path) -> None:
    capture = _call()
    models = FakeModels()

    session, _events = _run(tmp_path, capture, models)

    assert len(session.transcript.entries) == 3
    assert models.released == 1
    assert models.speakers is not None and models.speakers.finished
    assert capture.stopped.is_set()


def test_the_microphones_speech_detection_loads_after_the_speaker_models_never_beside_them(tmp_path: Path) -> None:
    models = FakeModels()

    session, _events = _run(tmp_path, _call(), models)

    assert models.loaded == ["speaker labels", "speech detection"]
    assert any(e.track == "me" for e in session.transcript.entries)


def test_with_the_microphone_off_only_the_speakers_are_recorded(tmp_path: Path) -> None:
    models = FakeModels()

    session, _events = _run(tmp_path, _call(), models, record_microphone=False)

    assert not (session.folder / "me.flac").exists()
    assert models.voice_asked == 0
    assert {e.track for e in session.transcript.entries} == {"them"}


def test_long_clips_get_the_vocabulary_and_names_given_mid_meeting(tmp_path: Path) -> None:
    (tmp_path / "vocabulary.txt").write_text("Northwind\nBRIGHTLINE\n", encoding="utf-8")
    them = np.concatenate([voice(0, 3), silence(1), voice(1, 1), silence(1)])
    models = FakeModels()

    _run(tmp_path, FakeCapture(them), models, record_microphone=False)

    assert models.contexts == ["Names and terms that may come up: Northwind, BRIGHTLINE.", None]


def test_when_speech_to_text_fails_the_audio_is_still_saved_and_the_window_is_told(tmp_path: Path) -> None:
    models = FakeModels(fails=RuntimeError("out of GPU memory"))

    session, events = _run(tmp_path, _call(), models)

    assert (session.folder / "them.flac").is_file()
    warnings = [e["warning"] for e in events if e["type"] == "status" and e["warning"]]
    assert warnings and "out of GPU memory" in str(warnings[-1]) and "Audio is still being saved" in str(warnings[-1])
    assert models.released == 1
    assert session.transcript.entries == []


def test_a_warning_from_the_models_reaches_the_window(tmp_path: Path) -> None:
    models = FakeModels(warning="No GPU found, so speakers are not told apart.")

    session, _events = _run(tmp_path, _call(), models)

    assert session.warning == "No GPU found, so speakers are not told apart."


def test_when_nothing_can_be_recorded_no_meeting_folder_is_left_behind(tmp_path: Path) -> None:
    session = Session(
        tmp_path / "2026-10-29 14-05", STARTED,
        capture=FakeCapture(silence(1), fails=RuntimeError("No speaker output found to record.")),
        models=FakeModels(), vocabulary=tmp_path / "vocabulary.txt", publish=lambda _event: None,
    )

    with pytest.raises(RuntimeError, match="No speaker output"):
        session.start()

    assert not session.folder.exists()


def test_quitting_mid_meeting_closes_the_audio_files_and_keeps_the_lines_so_far(tmp_path: Path) -> None:
    capture = FakeCapture(np.concatenate([voice(0, 2), silence(1)] * 20))
    session = Session(
        tmp_path / "2026-10-29 14-05", STARTED,
        capture=capture, models=FakeModels(), vocabulary=tmp_path / "vocabulary.txt", publish=lambda _event: None,
    )
    session.start()

    session.close_quickly()

    assert capture.stopped.wait(_WAIT_SECONDS)
    audio, _rate = soundfile.read(session.folder / "them.flac", dtype="float32")
    assert len(audio) > 0
