"""Past dictations and what can be done to one, for the UI shell.

Nothing here caches: the logs are the record, and a dictation finished a
moment ago is in them before its completion event is published.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from skryb.core.timestamps import utc_now
from skryb.history.entries import (
    HistoryEntry,
    filter_entries,
    find_entry,
    load_history,
)
from skryb.history.management import (
    delete_all_in_background,
    delete_entries,
    export_audio,
    export_csv,
)
from skryb.recording.audio_segment import load_wav

RetryOutcome = str | Exception
Retranscribe = Callable[[Path, Callable[[RetryOutcome], None]], None]


def play_wav(path: Path) -> None:
    """Start playing *path* on the default output device, without waiting."""
    audio = load_wav(path)
    import sounddevice as sd

    try:
        sd.play(audio.data, audio.sample_rate)
    except sd.PortAudioError as exc:  # no output device, or one that will not open
        raise OSError(str(exc)) from exc


def stop_playback() -> None:
    import sounddevice as sd

    sd.stop()


def parse_entry_id(text: str) -> tuple[str, int] | None:
    session, _dot, line = text.rpartition(".")
    if not session or not line.isdigit():
        return None
    return session, int(line)


class HistoryService:
    def __init__(
        self,
        diagnostics_dir: Path,
        retranscribe: Retranscribe,
        *,
        play: Callable[[Path], None] = play_wav,
        stop: Callable[[], None] = stop_playback,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._dir = diagnostics_dir
        self._retranscribe = retranscribe
        self._play = play
        self._stop = stop
        self._now = now

    def entries(self, query: str = "", limit: int | None = None) -> list[HistoryEntry]:
        """The newest *limit* (all by default) whose text holds *query*, ignoring case."""
        return filter_entries(load_history(self._dir), query)[:limit]

    def find(self, entry_id: str) -> HistoryEntry | None:
        key = parse_entry_id(entry_id)
        return None if key is None else find_entry(self._dir, key)

    def delete(self, entry: HistoryEntry) -> None:
        self._stop()
        delete_entries(self._dir, [entry], self._now())

    def delete_all(self, on_done: Callable[[bool], None]) -> None:
        """Delete everything on a daemon thread; *on_done* is told from it."""
        self._stop()
        delete_all_in_background(self._dir, on_done)

    def export(self, entry: HistoryEntry, destination: Path) -> None:
        export_audio(self._dir, entry, destination)

    def export_csv(self, query: str, destination: Path) -> int:
        """Every entry *query* keeps, as one spreadsheet row each."""
        return export_csv(self.entries(query), destination)

    def play(self, entry: HistoryEntry) -> None:
        if entry.audio_path is None or not entry.audio_path.is_file():
            raise FileNotFoundError("No recording was saved for this entry.")
        self._play(entry.audio_path)

    def stop(self) -> None:
        self._stop()

    def retry(self, entry: HistoryEntry, on_done: Callable[[RetryOutcome], None]) -> None:
        """Transcribe the recording again on a worker; the text is not saved to history."""
        if entry.audio_path is None or not entry.audio_path.is_file():
            raise FileNotFoundError("No recording was saved for this entry.")
        self._retranscribe(entry.audio_path, on_done)
