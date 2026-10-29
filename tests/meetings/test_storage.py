import json
from datetime import datetime
from pathlib import Path

import pytest

from skryb.meetings.segmenter import Segment
from skryb.meetings.storage import (
    default_meetings_folder,
    list_meetings,
    meeting_folder,
    meeting_settings_from,
    new_meeting_folder,
    rename_meeting,
    vocabulary_file,
)
from skryb.meetings.transcript import VOCABULARY_TEMPLATE, Transcript
from tests.fakes.config import FakeConfig


def _meeting(root: Path, name: str, started: datetime, lines: list[tuple[str, int, float, str]]) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    transcript = Transcript(folder, started)
    for track, speaker, start, text in lines:
        transcript.add(track, Segment(start, start + 1.5, speaker), text)
    return folder


def test_a_pc_with_a_d_drive_keeps_meetings_in_d_meetings() -> None:
    assert default_meetings_folder(lambda _drive: True) == Path("D:/Meetings")


def test_a_pc_without_a_d_drive_keeps_them_in_documents(tmp_path: Path) -> None:
    assert default_meetings_folder(lambda _drive: False, tmp_path) == tmp_path / "Documents" / "Skryb Meetings"


def test_each_meeting_gets_a_folder_named_by_the_minute_it_started(tmp_path: Path) -> None:
    started = datetime(2026, 10, 29, 9, 5, 41)
    assert new_meeting_folder(tmp_path, started) == tmp_path / "2026-10-29 09-05"


def test_two_meetings_in_the_same_minute_get_numbered_folders(tmp_path: Path) -> None:
    started = datetime(2026, 10, 29, 9, 5)
    (tmp_path / "2026-10-29 09-05").mkdir()
    (tmp_path / "2026-10-29 09-05 (2)").mkdir()
    assert new_meeting_folder(tmp_path, started) == tmp_path / "2026-10-29 09-05 (3)"


def test_the_vocabulary_file_is_created_with_its_example_once_and_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "Meetings"
    path = vocabulary_file(root)
    assert path.read_text(encoding="utf-8") == VOCABULARY_TEMPLATE
    path.write_text("Northwind\nBRIGHTLINE\n", encoding="utf-8")
    assert vocabulary_file(root).read_text(encoding="utf-8") == "Northwind\nBRIGHTLINE\n"


def test_saved_meetings_are_listed_newest_first_with_their_length_speakers_and_words(tmp_path: Path) -> None:
    _meeting(tmp_path, "2026-10-16 Ops Onboarding", datetime(2026, 10, 16, 20, 20), [
        ("them", 0, 0.0, "All right, let's jump in."),
        ("them", 1, 3.0, "Sounds good."),
        ("me", 0, 3065.0, "Thanks everyone."),
    ])
    _meeting(tmp_path, "2026-10-28 22-45 Marta email tool setup", datetime(2026, 10, 28, 22, 45), [
        ("them", 2, 10.0, "Haan theek hai, main kal tak bhej dunga."),
    ])
    (tmp_path / "2026-10-29 07-00").mkdir()  # a meeting that never got a line
    (tmp_path / "vocabulary.txt").write_text("Northwind\n", encoding="utf-8")

    meetings = list_meetings(tmp_path)

    assert [m.id for m in meetings] == ["2026-10-28 22-45 Marta email tool setup", "2026-10-16 Ops Onboarding"]
    onboarding = meetings[1]
    assert (onboarding.seconds, onboarding.speakers, onboarding.lines, onboarding.words) == (3066.5, 3, 3, 9)
    assert onboarding.started == "2026-10-16T20:20:00"
    assert onboarding.has_audio is False


def test_a_meeting_with_a_broken_transcript_is_left_out_of_the_list(tmp_path: Path) -> None:
    _meeting(tmp_path, "2026-10-20 22-53", datetime(2026, 10, 20, 22, 53), [("them", 0, 1.0, "Fine.")])
    broken = tmp_path / "Dana call"
    broken.mkdir()
    (broken / "transcript.json").write_text("{ half written", encoding="utf-8")
    other = tmp_path / "Ruth Abbott"
    other.mkdir()
    (other / "transcript.json").write_bytes(b"\xff\xfe not utf-8")

    assert [m.id for m in list_meetings(tmp_path)] == ["2026-10-20 22-53"]


def test_a_missing_meetings_folder_lists_nothing(tmp_path: Path) -> None:
    assert list_meetings(tmp_path / "nowhere") == []


def test_a_meetings_folder_with_hundreds_of_meetings_lists_them_all(tmp_path: Path) -> None:
    for day in range(300):
        started = datetime(2026, 1, 1, 9, 0).replace(day=1 + day % 28, month=1 + day // 28 % 12)
        folder = tmp_path / f"meeting {day:03}"
        folder.mkdir()
        (folder / "transcript.json").write_text(json.dumps({
            "started": started.isoformat(), "names": {},
            "entries": [{"id": n, "track": "them", "speaker": n % 3, "start": n * 2.0, "end": n * 2.0 + 1.5, "text": "ठीक है ok"} for n in range(40)],
        }, ensure_ascii=False), encoding="utf-8")

    meetings = list_meetings(tmp_path)

    assert len(meetings) == 300
    assert [m.started for m in meetings] == sorted((m.started for m in meetings), reverse=True)


@pytest.mark.parametrize("meeting_id", ["", ".", "..", "../Windows", "a/b", "a\\b", "C:", "vocabulary.txt", "missing"])
def test_only_a_meeting_folder_directly_inside_the_meetings_folder_can_be_opened(tmp_path: Path, meeting_id: str) -> None:
    (tmp_path / "vocabulary.txt").write_text("Northwind\n", encoding="utf-8")
    assert meeting_folder(tmp_path, meeting_id) is None


def test_a_meeting_named_in_any_language_can_be_opened(tmp_path: Path) -> None:
    folder = _meeting(tmp_path, "2026-10-21 میٹنگ – Harbor ✓", datetime(2026, 10, 21, 22, 55), [("them", 0, 0.0, "Hi.")])
    assert meeting_folder(tmp_path, "2026-10-21 میٹنگ – Harbor ✓") == folder


def test_meeting_settings_default_to_the_default_folder_the_microphone_and_english() -> None:
    settings = meeting_settings_from(FakeConfig({}))
    assert settings.folder == default_meetings_folder()
    assert settings.record_microphone is True
    assert settings.language == "en"


def test_saved_meeting_settings_are_read_and_an_unknown_language_falls_back_to_english() -> None:
    settings = meeting_settings_from(FakeConfig({
        "meetings.folder": "  E:\\Calls  ", "meetings.record_microphone": False, "meetings.language": "fr",
    }))
    assert settings.folder == Path("E:\\Calls")
    assert settings.record_microphone is False
    assert settings.language == "en"


@pytest.mark.parametrize(("name", "folder"), [
    ("Sprint review", "2026-10-29 14-05 Sprint review"),
    ('Q3: plan/review?* "final"', "2026-10-29 14-05 Q3 plan review final"),
    ("  ٹیم   میٹنگ\tnotes  ", "2026-10-29 14-05 ٹیم میٹنگ notes"),
    ("a" * 200, "2026-10-29 14-05 " + "a" * 80),
    ("trailing dots...", "2026-10-29 14-05 trailing dots"),
    ("", "2026-10-29 14-05"),
])
def test_a_file_meeting_is_named_after_the_file_in_a_name_windows_accepts(tmp_path: Path, name: str, folder: str) -> None:
    assert new_meeting_folder(tmp_path, datetime(2026, 10, 29, 14, 5, 33), name).name == folder


def test_a_second_file_with_the_same_name_in_the_same_minute_gets_a_number(tmp_path: Path) -> None:
    (tmp_path / "2026-10-29 14-05 Sprint review").mkdir()

    assert new_meeting_folder(tmp_path, datetime(2026, 10, 29, 14, 5), "Sprint review").name == "2026-10-29 14-05 Sprint review (2)"


_STARTED = datetime(2026, 10, 16, 20, 20)
_LINES = [("them", 0, 0.0, "Shall we start?"), ("me", 0, 2.0, "Haan, chalein.")]


@pytest.mark.parametrize(("old", "name", "new"), [
    ("2026-10-16 20-20 Weekly sync", "Planning – ٹیم میٹنگ", "2026-10-16 20-20 Planning – ٹیم میٹنگ"),
    ("2026-10-16 20-20", "Sprint review", "2026-10-16 20-20 Sprint review"),
    ("2026-10-16 20-20 Weekly sync (3)", "Retro", "2026-10-16 20-20 Retro"),
    ("2026-10-16 Weekly sync", "Retro", "2026-10-16 Retro"),
    ("Weekly sync", "Retro", "Retro"),
    ("2026-10-16 20-20 Weekly sync", 'Q3: "plan" / review?', "2026-10-16 20-20 Q3 plan review"),
    ("2026-10-16 20-20 Weekly sync", "x" * 300, "2026-10-16 20-20 " + "x" * 80),
    ("2026-10-16 20-20 Weekly sync", "   ", "2026-10-16 20-20"),
    ("Weekly sync", "", "Meeting"),
])
def test_renaming_a_meeting_keeps_its_date_and_time_and_moves_its_files(tmp_path: Path, old: str, name: str, new: str) -> None:
    folder = _meeting(tmp_path, old, _STARTED, _LINES)
    (folder / "them.flac").write_bytes(b"fLaC")

    renamed = rename_meeting(folder, name)

    assert renamed == tmp_path / new
    assert not folder.exists()
    assert (renamed / "them.flac").read_bytes() == b"fLaC"
    assert [m.id for m in list_meetings(tmp_path)] == [new]


def test_a_name_another_meeting_has_gets_the_next_free_number_even_among_a_thousand(tmp_path: Path) -> None:
    (tmp_path / "2026-10-16 20-20 Standup").mkdir()
    for n in range(2, 1001):
        (tmp_path / f"2026-10-16 20-20 Standup ({n})").mkdir()
    folder = _meeting(tmp_path, "2026-10-16 20-20 Weekly sync", _STARTED, _LINES)

    assert rename_meeting(folder, "Standup").name == "2026-10-16 20-20 Standup (1001)"


def test_renaming_a_meeting_to_its_own_name_changes_nothing(tmp_path: Path) -> None:
    folder = _meeting(tmp_path, "2026-10-16 20-20 Weekly sync", _STARTED, _LINES)

    assert rename_meeting(folder, "Weekly sync") == folder
    assert folder.is_dir()


def test_a_change_of_case_alone_still_renames_the_folder(tmp_path: Path) -> None:
    folder = _meeting(tmp_path, "2026-10-16 20-20 weekly sync", _STARTED, _LINES)

    renamed = rename_meeting(folder, "Weekly Sync")

    assert renamed.name == "2026-10-16 20-20 Weekly Sync"
    assert [entry.name for entry in tmp_path.iterdir()] == ["2026-10-16 20-20 Weekly Sync"]
