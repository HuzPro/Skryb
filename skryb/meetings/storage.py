"""Where meetings are kept: one folder per meeting under the meetings folder.

Each meeting folder holds ``them.flac`` (what the PC played), ``me.flac``
(the microphone), ``transcript.json`` and ``transcript.md``. One
``vocabulary.txt`` beside the meeting folders serves every meeting.
"""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from skryb.core.config import ConfigurationStore
from skryb.meetings.transcript import TRANSCRIPT_JSON, VOCABULARY, VOCABULARY_TEMPLATE, Transcript

_SECOND_DRIVE = Path("D:/")
# English; Urdu, written in Roman Urdu; or each sentence in whichever of the two it is.
LANGUAGES = ("en", "ur", "auto")


_NAME_CHARS = 80
_DATE_AND_TIME = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}(?: [0-9]{2}-[0-9]{2})?")


@dataclass(frozen=True)
class MeetingSettings:
    folder: Path
    record_microphone: bool
    language: str


def meeting_settings_from(config: ConfigurationStore) -> MeetingSettings:
    """The saved meeting settings; a blank folder is the default meetings folder."""
    folder = str(config.get("meetings.folder", "") or "").strip()
    language = config.get("meetings.language", "en")
    return MeetingSettings(
        folder=Path(folder) if folder else default_meetings_folder(),
        record_microphone=config.get("meetings.record_microphone", True) is not False,
        language=language if isinstance(language, str) and language in LANGUAGES else "en",
    )


def default_meetings_folder(
    drive_exists: Callable[[Path], bool] = Path.exists, home: Path | None = None,
) -> Path:
    """``D:\\Meetings`` on a PC with a D: drive, else ``Skryb Meetings`` in Documents."""
    if drive_exists(_SECOND_DRIVE):
        return _SECOND_DRIVE / "Meetings"
    return (home or Path.home()) / "Documents" / "Skryb Meetings"


def new_meeting_folder(root: Path, started: datetime, name: str = "") -> Path:
    """``YYYY-MM-DD HH-MM``, then *name* when given, with `` (2)`` and up when a meeting already has that name."""
    return _unused(root / " ".join(part for part in (started.strftime("%Y-%m-%d %H-%M"), _folder_safe(name)) if part))


def rename_meeting(folder: Path, name: str) -> Path:
    """Call the meeting in *folder* *name*: its date and time stay, a blank name leaves only them. Returns the new folder."""
    stamp = _DATE_AND_TIME.match(folder.name)
    target = _unused(folder.with_name(" ".join(part for part in (stamp and stamp.group(0), _folder_safe(name)) if part) or "Meeting"), folder)
    # Compared by name: Windows paths are equal in any case, and a change of case is still a rename.
    if target.name != folder.name:
        folder.rename(target)
    return target


def _unused(folder: Path, owner: Path | None = None) -> Path:
    """*folder*, or with `` (2)`` and up when another meeting has it; *owner* may keep its own name, in any case."""
    candidate, n = folder, 2
    while candidate.exists() and not (owner is not None and os.path.samefile(candidate, owner)):
        candidate = folder.with_name(f"{folder.name} ({n})")
        n += 1
    return candidate


def _folder_safe(name: str) -> str:
    """*name* as part of a folder name: no characters Windows refuses, one line, at most 80 characters."""
    cleaned = "".join(" " if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in name)
    return " ".join(cleaned.split())[:_NAME_CHARS].rstrip(" .")


def vocabulary_file(root: Path) -> Path:
    """The shared names-and-terms list, created with a commented example the first time."""
    path = root / VOCABULARY
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(VOCABULARY_TEMPLATE, encoding="utf-8")
    return path


@dataclass(frozen=True)
class MeetingSummary:
    id: str
    started: str
    seconds: float
    speakers: int
    lines: int
    words: int
    has_audio: bool

    def json(self) -> dict[str, object]:
        return {
            "id": self.id, "started": self.started, "seconds": self.seconds, "speakers": self.speakers,
            "lines": self.lines, "words": self.words, "has_audio": self.has_audio,
        }


def list_meetings(root: Path) -> list[MeetingSummary]:
    """Every meeting folder with a readable transcript, newest first."""
    try:
        folders = [entry for entry in os.scandir(root) if entry.is_dir() and (Path(entry.path) / TRANSCRIPT_JSON).is_file()]
    except OSError:
        return []
    summaries = [summary for summary in map(_summary, folders) if summary is not None]
    return sorted(summaries, key=lambda summary: (summary.started, summary.id), reverse=True)


def meeting_folder(root: Path, meeting_id: str) -> Path | None:
    """The folder of meeting *meeting_id*, which must be a meeting directly under *root*."""
    if not meeting_id or meeting_id in (".", "..") or any(sep in meeting_id for sep in ("/", "\\", ":")):
        return None
    folder = root / meeting_id
    return folder if (folder / TRANSCRIPT_JSON).is_file() else None


def _summary(entry: os.DirEntry[str]) -> MeetingSummary | None:
    folder = Path(entry.path)
    try:
        transcript = Transcript.load(folder)
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    entries = transcript.entries
    return MeetingSummary(
        id=entry.name,
        started=transcript.started.isoformat(timespec="seconds"),
        seconds=max((e.end for e in entries), default=0.0),
        speakers=len({e.key for e in entries}),
        lines=len(entries),
        words=sum(len(e.text.split()) for e in entries),
        has_audio=(folder / "them.flac").is_file() or (folder / "me.flac").is_file(),
    )
