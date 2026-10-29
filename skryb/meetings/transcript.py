"""The growing transcript of one meeting: its lines, speaker names and saved files.

``transcript.json`` and ``transcript.md`` are rewritten whole on every line,
in the layout the standalone meeting transcriber wrote, so meetings it saved
open here unchanged.
"""

import json
import logging
import os
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from skryb.meetings.segmenter import Segment

Event = dict[str, object]
Publish = Callable[[Event], None]

TRANSCRIPT_JSON = "transcript.json"
TRANSCRIPT_MD = "transcript.md"
VOCABULARY = "vocabulary.txt"
VOCABULARY_TEMPLATE = """\
# Names and terms the transcript should spell right, one per line, e.g.
# Launch Board
# BRIGHTLINE
# Speaker names you type in the meeting window are added automatically.
"""
# On shorter clips the list makes the model invent words ("Oh." became "Boston.").
_MIN_CONTEXT_SECONDS = 2.0
_ECHO_SHARE = 0.6
_TRACKS = ("me", "them")
_logger = logging.getLogger(__name__)


def asr_context(vocabulary: Path, names: dict[str, str], segment: Segment) -> str | None:
    """The names-and-terms hint for one clip, re-read each time so edits apply mid-meeting."""
    if segment.end - segment.start < _MIN_CONTEXT_SECONDS:
        return None
    try:
        lines = vocabulary.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        lines = []
    terms = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    terms += [name for name in names.values() if name not in terms]
    return f"Names and terms that may come up: {', '.join(terms)}." if terms else None


@dataclass
class Entry:
    id: int
    track: str  # "me" (microphone) or "them" (speaker output)
    speaker: int
    start: float
    end: float
    text: str

    @property
    def key(self) -> str:
        return f"{self.track}:{self.speaker}"


def default_label(key: str) -> str:
    track, _, speaker = key.partition(":")
    return "Me" if track == "me" else f"Speaker {int(speaker) + 1}"


def is_speaker_key(key: str) -> bool:
    track, _, speaker = key.partition(":")
    return track in _TRACKS and speaker.isascii() and speaker.isdigit()


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def is_echo(mic: Entry, speakers: Entry) -> bool:
    """The mic heard the same words the speakers played at the same time."""
    if min(mic.end, speakers.end) <= max(mic.start, speakers.start):
        return False
    a, b = _words(mic.text), _words(speakers.text)
    if not a or not b:
        return False
    matched = sum(block.size for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return matched / min(len(a), len(b)) >= _ECHO_SHARE


def _ignore(_event: Event) -> None:
    return None


class Transcript:
    """One meeting's lines, names and files; every change is saved before it is published."""

    def __init__(self, folder: Path, started: datetime, publish: Publish = _ignore) -> None:
        self.folder = folder
        self.started = started
        self.entries: list[Entry] = []
        self.names: dict[str, str] = {}
        self._publish = publish
        self._lock = threading.Lock()
        self._next_id = 0

    @classmethod
    def load(cls, folder: Path, publish: Publish = _ignore) -> "Transcript":
        """A saved meeting, read back; lines that cannot be read are skipped."""
        data = json.loads((folder / TRANSCRIPT_JSON).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{folder / TRANSCRIPT_JSON} is not a transcript.")
        transcript = cls(folder, _started(data.get("started"), folder), publish)
        transcript.entries = sorted(
            (entry for entry in map(_entry, data.get("entries") or []) if entry is not None),
            key=lambda entry: entry.start,
        )
        names = data.get("names")
        if isinstance(names, dict):
            transcript.names = {
                str(key): str(name) for key, name in names.items()
                if is_speaker_key(str(key)) and str(name).strip() and str(name) != default_label(str(key))
            }
        transcript._next_id = max((entry.id for entry in transcript.entries), default=-1) + 1
        return transcript

    def label(self, key: str) -> str:
        return self.names.get(key) or default_label(key)

    def add(self, track: str, segment: Segment, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            entry = Entry(self._next_id, track, segment.speaker, round(segment.start, 2), round(segment.end, 2), text.strip())
            self._next_id += 1
            others = [e for e in self.entries if e.track != track]
            # ponytail: on speakers without echo cancellation the mic re-hears remote voices; drop those
            # mic lines by text match. Add WebRTC AEC if overlapping talk gets lost.
            if track == "me" and any(is_echo(entry, e) for e in others):
                _logger.info("Dropped a microphone line that echoed the speakers.")
                return
            echoes = [e for e in others if track == "them" and is_echo(e, entry)]
            for echo in echoes:
                self.entries.remove(echo)
            self.entries.append(entry)
            self.entries.sort(key=lambda e: e.start)
            self._save()
        for echo in echoes:
            self._publish({"type": "remove", "entry_id": echo.id})
        self._publish({"type": "entry", "entry": self.entry_view(entry)})

    def rename(self, key: str, name: str) -> None:
        with self._lock:
            name = name.strip()
            if name and name != default_label(key):
                self.names[key] = name
            else:
                self.names.pop(key, None)
            self._save()
        self._publish({"type": "names", "names": self.name_view()})

    def entry_view(self, entry: Entry) -> Event:
        return {**asdict(entry), "key": entry.key}

    def name_view(self) -> dict[str, str]:
        keys = {e.key for e in self.entries} | set(self.names)
        return {key: self.label(key) for key in sorted(keys)}

    def view(self) -> Event:
        with self._lock:
            return {
                "folder": str(self.folder),
                "started": self.started.isoformat(timespec="seconds"),
                "names": self.name_view(),
                "entries": [self.entry_view(e) for e in self.entries],
            }

    def markdown(self) -> str:
        lines = [f"# Meeting {self.started:%Y-%m-%d %H:%M}", ""]
        previous = None
        for e in self.entries:
            if e.key != previous:
                lines += ["", f"**{self.label(e.key)}** [{_clock(e.start)}]"]
                previous = e.key
            lines.append(e.text)
        return "\n".join(lines).strip() + "\n"

    def save(self) -> None:
        with self._lock:
            self._save()

    def _save(self) -> None:
        data = {
            "started": self.started.isoformat(timespec="seconds"),
            "names": self.name_view(),
            "entries": [asdict(e) for e in self.entries],
        }
        write_atomic(self.folder / TRANSCRIPT_JSON, json.dumps(data, indent=1, ensure_ascii=False))
        write_atomic(self.folder / TRANSCRIPT_MD, self.markdown())


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02}:{s // 60 % 60:02}:{s % 60:02}"


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _started(value: object, folder: Path) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.fromtimestamp(folder.stat().st_mtime)


def _entry(raw: object) -> Entry | None:
    if not isinstance(raw, dict):
        return None
    try:
        entry = Entry(
            int(raw["id"]), str(raw["track"]), int(raw["speaker"]),
            float(raw["start"]), float(raw["end"]), str(raw["text"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return entry if entry.track in _TRACKS else None
