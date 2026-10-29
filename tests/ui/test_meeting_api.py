"""The Meetings window's endpoints over loopback HTTP, recording through fake devices and models."""

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from skryb.core.event_bus import SimpleEventBus
from skryb.meetings.model_files import MeetingModelFiles
from skryb.meetings.recorder import MeetingRecorder
from skryb.meetings.segmenter import Segment
from skryb.meetings.storage import MeetingSettings
from skryb.meetings.transcript import Transcript
from skryb.ui.api_server import ApiServer
from skryb.ui.event_hub import EventHub
from skryb.ui.meeting_handlers import MeetingApi
from tests.meetings.fakes import FakeCapture, FakeModels, silence, voice
from tests.ui.rig import EventStream

_TIMEOUT_SECONDS = 10.0


class MeetingRig:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "Meetings"
        self.recycle_bin = tmp_path / "Recycle Bin"
        self.recycle_refusal: OSError | None = None
        self.installed = {"nemotron-3-diarization": True, "qwen3-asr-1.7b": True}
        self.downloads: list[str] = []
        self.opened: list[Path] = []
        self.captures: list[FakeCapture] = []
        self.hub = EventHub(SimpleEventBus())
        self.server = ApiServer(self.hub, poll_seconds=0.02)
        self.server.start()
        files = MeetingModelFiles(
            diarizer_installed=lambda: self.installed["nemotron-3-diarization"],
            download_diarizer=lambda: self.downloads.append("nemotron-3-diarization"),
            is_installed=lambda name: self.installed[name],
            start_download=self.downloads.append,
        )
        self.recorder = MeetingRecorder(
            settings=lambda: MeetingSettings(self.root, record_microphone=False, language="en"),
            missing_models=files.missing,
            make_models=lambda _settings: FakeModels(),
            make_capture=self._capture,
            publish=lambda event: self.hub.publish("meeting", event),
            clock=lambda: datetime(2026, 10, 29, 14, 5),
        )
        self.server.bind(MeetingApi(
            self.recorder, self.hub, folder=lambda: self.root, model_files=files,
            open_path=self.opened.append, run_in_background=lambda action: action(), recycle=self._recycle,
        ).routes())

    def _recycle(self, folder: Path) -> None:
        if self.recycle_refusal is not None:
            raise self.recycle_refusal
        self.recycle_bin.mkdir(exist_ok=True)
        folder.rename(self.recycle_bin / folder.name)

    def _capture(self, _record_microphone: bool) -> FakeCapture:
        self.captures.append(FakeCapture(np.concatenate([voice(0, 2.5), silence(1)])))
        return self.captures[-1]

    def call(self, method: str, path: str, body: object = None) -> tuple[int, Any]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.port}{urllib.request.quote(path)}",
            data=None if body is None else json.dumps(body).encode(), method=method,
            headers={"Authorization": f"Bearer {self.server.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            with error:
                raw = error.read()
                return error.code, json.loads(raw) if raw else None

    def events(self) -> EventStream:
        return EventStream(f"http://127.0.0.1:{self.server.port}/api/events?token={self.server.token}")

    def saved_meeting(self, name: str, lines: list[tuple[str, int, float, str]]) -> Path:
        folder = self.root / name
        folder.mkdir(parents=True)
        transcript = Transcript(folder, datetime(2026, 10, 16, 20, 20))
        for track, speaker, start, text in lines:
            transcript.add(track, Segment(start, start + 2.0, speaker), text)
        return folder


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[MeetingRig]:
    built = MeetingRig(tmp_path)
    yield built
    built.recorder.stop()
    built.server.stop()


def test_before_any_meeting_there_is_no_live_meeting(rig: MeetingRig) -> None:
    assert rig.call("GET", "/api/meetings/live") == (200, {"meeting": None})


def test_the_overview_lists_saved_meetings_the_folder_and_whether_the_models_are_ready(rig: MeetingRig) -> None:
    rig.saved_meeting("2026-10-16 Ops Onboarding", [("them", 0, 0.0, "All right, let's jump in."), ("me", 0, 5.0, "Theek hai.")])

    status, overview = rig.call("GET", "/api/meetings")

    assert status == 200
    assert overview["recording"] is False and overview["models_ready"] is True
    assert overview["folder"] == str(rig.root)
    assert [m["id"] for m in overview["meetings"]] == ["2026-10-16 Ops Onboarding"]
    assert [m["name"] for m in overview["models"]] == ["nemotron-3-diarization", "qwen3-asr-1.7b"]
    assert all(m["download"] is None for m in overview["models"])


def test_starting_without_the_models_says_what_to_download_and_the_download_button_fetches_only_those(rig: MeetingRig) -> None:
    rig.installed["nemotron-3-diarization"] = False

    status, body = rig.call("POST", "/api/meetings/start")
    assert status == 400 and body["error"] == "Download the meeting models first: Speaker labels."

    assert rig.call("POST", "/api/meetings/models/download")[0] == 202
    assert rig.downloads == ["nemotron-3-diarization"]


def test_a_meeting_started_from_the_window_streams_its_lines_and_stops_into_a_saved_meeting(rig: MeetingRig) -> None:
    stream = rig.events()

    assert rig.call("POST", "/api/meetings/start")[0] == 202
    kinds: list[str] = []
    entry: dict[str, Any] = {}
    while not entry:
        event = stream.until("meeting")
        kinds.append(event["type"])
        entry = event if event["type"] == "entry" else {}
    assert "started" in kinds
    assert entry["id"] == "2026-10-29 14-05" and entry["entry"]["key"] == "them:0"

    status, body = rig.call("GET", "/api/meetings/live")
    live = body["meeting"]
    assert status == 200 and live["recording"] is True and live["id"] == "2026-10-29 14-05"

    assert rig.call("POST", "/api/meetings/stop")[0] == 202
    _status, overview = rig.call("GET", "/api/meetings")
    assert overview["recording"] is False
    assert [m["id"] for m in overview["meetings"]] == ["2026-10-29 14-05"]
    stream.close()


def test_renaming_a_speaker_in_a_saved_meeting_rewrites_its_files_and_tells_every_window(rig: MeetingRig) -> None:
    folder = rig.saved_meeting("2026-10-21 Harbor scoping call", [("them", 0, 0.0, "Hi."), ("them", 1, 3.0, "Hello.")])
    stream = rig.events()

    assert rig.call("POST", "/api/meetings/2026-10-21 Harbor scoping call/rename", {"key": "them:1", "name": "  نادیہ رحمان  "})[0] == 204

    names = stream.until("meeting")
    assert names == {"type": "names", "names": {"them:0": "Speaker 1", "them:1": "نادیہ رحمان"}, "id": "2026-10-21 Harbor scoping call"}
    assert "**نادیہ رحمان**" in (folder / "transcript.md").read_text(encoding="utf-8")
    status, transcript = rig.call("GET", "/api/meetings/2026-10-21 Harbor scoping call")
    assert status == 200 and transcript["names"]["them:1"] == "نادیہ رحمان"
    stream.close()


def test_a_name_longer_than_sixty_characters_is_cut_to_sixty(rig: MeetingRig) -> None:
    rig.saved_meeting("2026-10-21 Harbor scoping call", [("them", 0, 0.0, "Hi.")])

    rig.call("POST", "/api/meetings/2026-10-21 Harbor scoping call/rename", {"key": "them:0", "name": "Z" * 500})

    _status, transcript = rig.call("GET", "/api/meetings/2026-10-21 Harbor scoping call")
    assert transcript["names"]["them:0"] == "Z" * 60


@pytest.mark.parametrize("body", [
    {"key": "../../x", "name": "Omar"}, {"key": "them:x", "name": "Omar"}, {"name": "Omar"}, {"key": "them:0"}, ["them:0"],
])
def test_a_rename_that_names_no_real_speaker_is_refused(rig: MeetingRig, body: object) -> None:
    rig.saved_meeting("2026-10-21 Harbor scoping call", [("them", 0, 0.0, "Hi.")])

    assert rig.call("POST", "/api/meetings/2026-10-21 Harbor scoping call/rename", body)[0] == 400


@pytest.mark.parametrize("meeting_id", ["..", "missing", "vocabulary.txt"])
def test_only_meetings_in_the_meetings_folder_can_be_read_or_opened(rig: MeetingRig, meeting_id: str) -> None:
    rig.root.mkdir(parents=True)
    (rig.root / "vocabulary.txt").write_text("Northwind\n", encoding="utf-8")

    assert rig.call("GET", f"/api/meetings/{meeting_id}")[0] in (404, 400)
    assert rig.call("POST", f"/api/meetings/{meeting_id}/open")[0] in (404, 400)
    assert rig.opened == []


def test_a_saved_transcript_that_cannot_be_read_is_explained(rig: MeetingRig) -> None:
    folder = rig.root / "Dana call"
    folder.mkdir(parents=True)
    (folder / "transcript.json").write_text("{ half written", encoding="utf-8")

    status, body = rig.call("GET", "/api/meetings/Dana call")

    assert status == 400 and body["error"].startswith("That meeting's transcript could not be read")


def test_editing_names_and_terms_creates_the_shared_file_and_opens_it(rig: MeetingRig) -> None:
    assert rig.call("POST", "/api/meetings/vocabulary/open")[0] == 204
    assert rig.opened == [rig.root / "vocabulary.txt"]
    assert (rig.root / "vocabulary.txt").read_text(encoding="utf-8").startswith("# Names and terms")


def test_open_folder_opens_a_saved_meetings_folder(rig: MeetingRig) -> None:
    folder = rig.saved_meeting("2026-10-16 Ops Onboarding", [("them", 0, 0.0, "Hi.")])

    assert rig.call("POST", "/api/meetings/2026-10-16 Ops Onboarding/open")[0] == 204
    assert rig.call("POST", "/api/meetings/folder/open")[0] == 204
    assert rig.opened == [folder, rig.root]


def _with_fake_file_reader(rig: MeetingRig) -> None:
    rig.recorder._check_file = lambda path: None
    rig.recorder._decode_file = lambda path, progress, cancelled: np.concatenate([voice(0, 2.5), silence(1)])


def test_a_dropped_file_is_transcribed_into_a_saved_meeting_named_after_it(rig: MeetingRig, tmp_path: Path) -> None:
    _with_fake_file_reader(rig)
    video = tmp_path / "Weekly sync – ہفتہ وار.mp4"
    video.write_bytes(b"fake")

    stream = rig.events()
    assert rig.call("POST", "/api/meetings/files", {"path": str(video), "language": "ur"})[0] == 202
    events: list[dict[str, Any]] = []
    while not events or events[-1]["type"] != "saved":
        events.append(stream.until("meeting"))
    stream.close()

    started = next(event for event in events if event["type"] == "started")
    assert started["source"] == "Weekly sync – ہفتہ وار.mp4"
    assert any(event["type"] == "entry" for event in events)
    meetings = rig.call("GET", "/api/meetings")[1]["meetings"]
    assert [m["id"] for m in meetings] == ["2026-10-29 14-05 Weekly sync – ہفتہ وار"]
    assert rig.call("GET", "/api/meetings/live")[1] == {"meeting": None}


@pytest.mark.parametrize(("body", "message"), [
    ({"language": "en"}, "Say which file to transcribe and in which language."),
    ({"path": "notes.mp4", "language": "en"}, "notes.mp4 isn't there any more."),
    ({"path": r"C:\nowhere\gone.mp4", "language": "en"}, "gone.mp4 isn't there any more."),
])
def test_a_file_request_that_doesnt_name_a_file_on_disk_is_refused(rig: MeetingRig, body: dict, message: str) -> None:
    status, error = rig.call("POST", "/api/meetings/files", body)

    assert status == 400 and message in json.dumps(error, ensure_ascii=False)


def test_a_file_in_a_language_skryb_doesnt_offer_is_refused(rig: MeetingRig, tmp_path: Path) -> None:
    _with_fake_file_reader(rig)
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"fake")

    status, error = rig.call("POST", "/api/meetings/files", {"path": str(video), "language": "fr"})

    assert status == 400 and "Pick English or Urdu for the file." in json.dumps(error)


def _ids(rig: MeetingRig) -> list[str]:
    return [m["id"] for m in rig.call("GET", "/api/meetings")[1]["meetings"]]


def test_renaming_a_saved_meeting_renames_its_folder_and_answers_with_its_new_id(rig: MeetingRig) -> None:
    rig.saved_meeting("2026-10-16 20-20 Ops Onboarding", [("them", 0, 0.0, "Welcome aboard.")])

    status, body = rig.call("POST", "/api/meetings/2026-10-16 20-20 Ops Onboarding/name", {"name": "Onboarding – ٹیم"})

    assert status == 200 and body == {"id": "2026-10-16 20-20 Onboarding – ٹیم"}
    assert _ids(rig) == ["2026-10-16 20-20 Onboarding – ٹیم"]
    status, transcript = rig.call("GET", "/api/meetings/2026-10-16 20-20 Onboarding – ٹیم")
    assert status == 200 and transcript["entries"][0]["text"] == "Welcome aboard."


def test_deleting_a_saved_meeting_sends_its_folder_to_the_recycle_bin(rig: MeetingRig) -> None:
    rig.saved_meeting("2026-10-16 20-20 Ops Onboarding", [("them", 0, 0.0, "Welcome aboard.")])
    rig.saved_meeting("2026-10-21 15-05 Harbor scoping call", [("them", 0, 0.0, "Hi.")])

    assert rig.call("DELETE", "/api/meetings/2026-10-16 20-20 Ops Onboarding")[0] == 204

    assert _ids(rig) == ["2026-10-21 15-05 Harbor scoping call"]
    assert (rig.recycle_bin / "2026-10-16 20-20 Ops Onboarding" / "transcript.json").is_file()


def test_a_delete_windows_refuses_keeps_the_meeting_and_says_why(rig: MeetingRig) -> None:
    rig.saved_meeting("2026-10-16 20-20 Ops Onboarding", [("them", 0, 0.0, "Welcome aboard.")])
    rig.recycle_refusal = OSError("Windows couldn't move 2026-10-16 20-20 Ops Onboarding to the Recycle Bin (code 0x20).")

    status, body = rig.call("DELETE", "/api/meetings/2026-10-16 20-20 Ops Onboarding")

    assert status == 400 and body["error"].startswith("Couldn't delete it: Windows couldn't move")
    assert _ids(rig) == ["2026-10-16 20-20 Ops Onboarding"]


@pytest.mark.parametrize("meeting_id", ["..", "missing", "vocabulary.txt"])
def test_only_meetings_in_the_meetings_folder_can_be_renamed_or_deleted(rig: MeetingRig, meeting_id: str) -> None:
    rig.root.mkdir(parents=True)
    (rig.root / "vocabulary.txt").write_text("Northwind\n", encoding="utf-8")

    assert rig.call("POST", f"/api/meetings/{meeting_id}/name", {"name": "x"})[0] in (404, 400)
    assert rig.call("DELETE", f"/api/meetings/{meeting_id}")[0] in (404, 400)
    assert (rig.root / "vocabulary.txt").is_file() and not rig.recycle_bin.exists()


@pytest.mark.parametrize("body", [{}, {"name": 7}, ["Retro"], None])
def test_a_rename_without_a_name_is_refused(rig: MeetingRig, body: object) -> None:
    rig.saved_meeting("2026-10-16 20-20 Ops Onboarding", [("them", 0, 0.0, "Hi.")])

    assert rig.call("POST", "/api/meetings/2026-10-16 20-20 Ops Onboarding/name", body)[0] == 400
    assert _ids(rig) == ["2026-10-16 20-20 Ops Onboarding"]


def test_a_meeting_being_recorded_cant_be_renamed_or_deleted(rig: MeetingRig) -> None:
    stream = rig.events()
    rig.call("POST", "/api/meetings/start")
    while stream.until("meeting")["type"] != "entry":
        pass

    renamed = rig.call("POST", "/api/meetings/2026-10-29 14-05/name", {"name": "Retro"})
    deleted = rig.call("DELETE", "/api/meetings/2026-10-29 14-05")

    assert renamed[0] == 400 and renamed[1]["error"] == "This meeting is still being recorded or transcribed. Stop it first."
    assert deleted[0] == 400
    assert (rig.root / "2026-10-29 14-05").is_dir()
    stream.close()


def test_the_meeting_just_recorded_can_be_renamed_and_its_speakers_named_in_the_new_folder(rig: MeetingRig) -> None:
    stream = rig.events()
    rig.call("POST", "/api/meetings/start")
    while stream.until("meeting")["type"] != "entry":
        pass
    rig.call("POST", "/api/meetings/stop")
    stream.close()

    _status, body = rig.call("POST", "/api/meetings/2026-10-29 14-05/name", {"name": "Retro"})
    assert rig.call("POST", f"/api/meetings/{body['id']}/rename", {"key": "them:0", "name": "Sana"})[0] == 204

    assert body["id"] == "2026-10-29 14-05 Retro"
    assert not (rig.root / "2026-10-29 14-05").exists()
    assert "**Sana**" in (rig.root / "2026-10-29 14-05 Retro" / "transcript.md").read_text(encoding="utf-8")
