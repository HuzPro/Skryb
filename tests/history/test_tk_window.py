import json
import tkinter
import wave
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    RecordingCancelledEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.history import tk_window
from voicetotext.history.tk_window import HistoryWindow

_SESSION = "20260924T063505Z-7aff44ec"
_START = datetime(2026, 9, 24, 6, 35, 5, tzinfo=UTC)


class FakeRetranscribe:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Callable]] = []

    def __call__(self, wav_path: Path, on_done: Callable) -> None:
        self.calls.append((wav_path, on_done))


def _run_now(action: Callable[[], None]) -> None:
    action()


def _write_wav(path: Path, seconds: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * int(16000 * seconds))


def _log(tmp_path: Path, events: list[dict]) -> Path:
    session = tmp_path / _SESSION
    session.mkdir(parents=True, exist_ok=True)
    with (session / "session.jsonl").open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    return session


def _dictation(index: int, text: str, **overrides) -> dict:
    event = {
        "event": "transcription",
        "index": index,
        "requested_at": (_START + timedelta(minutes=index)).isoformat(),
        "audio_seconds": 3.5,
        "audio_path": f"audio/{index:04d}.wav",
        "text": text,
        "transcript_path": f"transcripts/{index:04d}.txt",
        "is_empty": not text.strip(),
        "error": None,
    }
    event.update(overrides)
    return event


@pytest.fixture()
def no_real_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``show`` would map the Toplevel on the host desktop; keep it withdrawn.
    monkeypatch.setattr(tkinter.Toplevel, "deiconify", lambda self: None)
    monkeypatch.setattr(tkinter.Toplevel, "focus_force", lambda self: None)


@pytest.fixture()
def retranscribe() -> FakeRetranscribe:
    return FakeRetranscribe()


@pytest.fixture()
def open_history(tk_root, tmp_path, retranscribe, no_real_window):
    windows: list[HistoryWindow] = []

    def open_(saves_history: Callable[[], bool] = lambda: False) -> HistoryWindow:
        history = HistoryWindow(
            tk_root, tmp_path, retranscribe, _run_now,
            saves_history=saves_history, run_in_background=_run_now,
        )
        history.show()
        windows.append(history)
        return history

    yield open_
    for history in windows:
        history.close()


def _rows(history: HistoryWindow) -> list[tuple]:
    tree = history._tree
    return [tuple(tree.item(iid, "values")) for iid in tree.get_children()]


def _select(history: HistoryWindow, row: int) -> None:
    history._tree.selection_set(history._tree.get_children()[row])
    history._render_selection()


def _detail(history: HistoryWindow) -> str:
    return history._detail.get("1.0", "end-1c")


def _enabled(button) -> bool:
    return not button.instate(["disabled"])


def test_lists_dictations_newest_first_with_length_and_status(
    tmp_path, open_history,
) -> None:
    _log(tmp_path, [
        _dictation(1, "First thing this morning."),
        _dictation(2, "", error="CUDA failed with error out of memory",
                   audio_path=""),
        {"event": "cancelled", "index": 3,
         "at": (_START + timedelta(minutes=3)).isoformat(), "audio_seconds": 61.0,
         "audio_path": "audio/0003.wav"},
    ])

    rows = _rows(open_history())

    assert [(text, length, status) for _time, text, length, status in rows] == [
        ("(cancelled - not transcribed)", "1:01", "Cancelled"),
        ("(CUDA failed with error out of memory)", "0:03", "Failed"),
        ("First thing this morning.", "0:03", ""),
    ]


def test_an_open_window_lists_each_new_dictation_without_pressing_refresh(
    tmp_path, tk_root, retranscribe, no_real_window,
) -> None:
    bus = SimpleEventBus()
    loads: list[str] = []

    def load_now(work: Callable[[], None]) -> None:
        loads.append("load")
        work()

    history = HistoryWindow(
        tk_root, tmp_path, retranscribe, _run_now,
        saves_history=lambda: True, run_in_background=load_now, event_bus=bus,
    )
    history.show()
    try:
        _log(tmp_path, [_dictation(1, "Book the dentist for Thursday.")])
        bus.publish(TranscriptionCompletedEvent(text="Book the dentist for Thursday."))
        _log(tmp_path, [_dictation(2, "", error="CUDA out of memory", audio_path="")])
        bus.publish(TranscriptionFailedEvent(error="CUDA out of memory"))
        _log(tmp_path, [
            {"event": "cancelled", "index": 3,
             "at": (_START + timedelta(minutes=3)).isoformat(), "audio_seconds": 4.0,
             "audio_path": "audio/0003.wav"},
        ])
        bus.publish(RecordingCancelledEvent(recoverable=True))

        assert [row[3] for row in _rows(history)] == ["Cancelled", "Failed", ""]
        assert history._hint.cget("text") == ""
    finally:
        history.close()

    bus.publish(TranscriptionCompletedEvent(text="after closing"))
    assert len(loads) == 4


def test_opening_again_reuses_the_open_window_and_rereads_the_logs(
    tmp_path, open_history,
) -> None:
    _log(tmp_path, [_dictation(1, "one")])
    history = open_history()
    window = history._window
    _log(tmp_path, [_dictation(2, "two")])

    history.show()

    assert history._window is window
    assert len(_rows(history)) == 2


def test_selecting_a_dictation_shows_its_full_text(tmp_path, open_history) -> None:
    text = "Line one of a long note.\nLine two — with ünïcödé and 你好 🎙️."
    _log(tmp_path, [_dictation(1, text)])
    history = open_history()

    _select(history, 0)

    assert _detail(history) == text


def test_search_narrows_the_list_and_clearing_it_restores_everything(
    tmp_path, open_history,
) -> None:
    _log(tmp_path, [
        _dictation(1, "Book the dentist"),
        _dictation(2, "Email the landlord"),
        _dictation(3, "book club on friday"),
    ])
    history = open_history()

    history._query.set("BOOK")
    assert [row[1] for row in _rows(history)] == [
        "book club on friday", "Book the dentist",
    ]
    history._query.set("")
    assert len(_rows(history)) == 3


def test_selection_survives_a_refresh_and_a_search(tmp_path, open_history) -> None:
    _log(tmp_path, [_dictation(1, "keep me selected"), _dictation(2, "other")])
    history = open_history()
    _select(history, 1)

    _log(tmp_path, [_dictation(3, "newer")])
    history.refresh()
    history._query.set("keep")

    assert _detail(history) == "keep me selected"


def test_copy_puts_the_shown_text_on_the_clipboard(
    tmp_path, open_history, monkeypatch,
) -> None:
    _log(tmp_path, [_dictation(1, "Copy this, please.")])
    history = open_history()
    copied: list[str] = []
    monkeypatch.setattr(history._window, "clipboard_clear", lambda: copied.clear())
    monkeypatch.setattr(history._window, "clipboard_append", copied.append)
    _select(history, 0)

    history._copy_button.invoke()

    assert copied == ["Copy this, please."]


def test_retry_shows_transcribing_then_the_new_text(
    tmp_path, open_history, retranscribe,
) -> None:
    session = _log(tmp_path, [_dictation(1, "old wrong text")])
    _write_wav(session / "audio" / "0001.wav")
    history = open_history()
    _select(history, 0)

    history._retry_button.invoke()
    assert _detail(history) == "Transcribing…"
    assert not _enabled(history._retry_button)
    assert not _enabled(history._copy_button)

    wav_path, on_done = retranscribe.calls[0]
    on_done("new right text")

    assert wav_path == session / "audio" / "0001.wav"
    assert _detail(history) == "new right text"
    assert _enabled(history._copy_button)
    assert _enabled(history._retry_button)


def test_a_failed_retry_keeps_the_old_text_and_shows_the_error(
    tmp_path, open_history, retranscribe,
) -> None:
    session = _log(tmp_path, [_dictation(1, "old text")])
    _write_wav(session / "audio" / "0001.wav")
    history = open_history()
    _select(history, 0)

    history._retry_button.invoke()
    retranscribe.calls[0][1](RuntimeError("model is still loading"))

    assert _detail(history) == "old text"
    assert history._status.cget("text") == "Transcription failed: model is still loading"


def test_a_retry_that_finishes_after_closing_is_ignored(
    tmp_path, open_history, retranscribe,
) -> None:
    session = _log(tmp_path, [_dictation(1, "old text")])
    _write_wav(session / "audio" / "0001.wav")
    history = open_history()
    _select(history, 0)
    history._retry_button.invoke()

    history.close()
    retranscribe.calls[0][1]("arrived late")

    assert history._window is None


def test_a_cancelled_recording_offers_transcribe(tmp_path, open_history) -> None:
    session = _log(tmp_path, [
        {"event": "cancelled", "index": 1, "at": _START.isoformat(),
         "audio_seconds": 2.0, "audio_path": "audio/0001.wav"},
    ])
    _write_wav(session / "audio" / "0001.wav")
    history = open_history()

    _select(history, 0)

    assert history._retry_button.cget("text") == "Transcribe"
    assert _enabled(history._retry_button)
    assert not _enabled(history._copy_button)


def test_retry_and_play_are_disabled_when_the_recording_is_missing(
    tmp_path, open_history,
) -> None:
    _log(tmp_path, [
        _dictation(1, "audio file was deleted"),
        _dictation(2, "", audio_path="", transcript_path="", is_empty=False),
    ])
    history = open_history()

    for row in (0, 1):
        _select(history, row)
        assert not _enabled(history._retry_button)
        assert not _enabled(history._play_button)


def test_play_turns_into_stop_and_back(
    tmp_path, open_history, monkeypatch,
) -> None:
    session = _log(tmp_path, [_dictation(1, "listen to me")])
    _write_wav(session / "audio" / "0001.wav")
    played: list[Path] = []
    stopped: list[bool] = []
    monkeypatch.setattr(tk_window, "_play_wav", lambda path: played.append(path) or 2.0)
    monkeypatch.setattr(tk_window, "_stop_playback", lambda: stopped.append(True))
    history = open_history()
    _select(history, 0)

    history._play_button.invoke()
    assert history._play_button.cget("text") == "Stop"
    history._play_button.invoke()

    assert played == [session / "audio" / "0001.wav"]
    assert stopped == [True]
    assert history._play_button.cget("text") == "Play"


def test_a_recording_that_cannot_be_played_shows_an_error(
    tmp_path, open_history,
) -> None:
    session = _log(tmp_path, [_dictation(1, "corrupt audio")])
    (session / "audio").mkdir()
    (session / "audio" / "0001.wav").write_bytes(b"RIFF not really a wav")
    history = open_history()
    _select(history, 0)

    history._play_button.invoke()

    assert history._status.cget("text").startswith("Could not play the recording")
    assert history._play_button.cget("text") == "Play"


def test_an_empty_logs_folder_explains_how_to_get_history(
    tmp_path, open_history,
) -> None:
    history = open_history()

    assert _rows(history) == []
    assert history._hint.cget("text").startswith("No dictations yet.")


def test_an_empty_history_stops_asking_to_turn_saving_on_once_it_is_on(
    tmp_path, open_history,
) -> None:
    saving = [False]
    history = open_history(saves_history=lambda: saving[0])
    assert "Save history" in history._hint.cget("text")

    saving[0] = True
    history.refresh()

    assert _rows(history) == []
    assert history._hint.cget("text") == "No dictations yet."


def test_lists_thousands_of_dictations(tmp_path, open_history) -> None:
    _log(tmp_path, [_dictation(i, f"dictation number {i}") for i in range(1, 4001)])

    history = open_history()

    rows = _rows(history)
    assert len(rows) == 4000
    assert rows[0][1] == "dictation number 4000"


def _answer_confirmations(monkeypatch, answer: bool) -> list[str]:
    asked: list[str] = []

    def ask(title: str, _message: str, **_kwargs) -> bool:
        asked.append(title)
        return answer

    monkeypatch.setattr(tk_window.messagebox, "askyesno", ask)
    return asked


def test_delete_asks_first_then_removes_only_the_selected_dictation(
    tmp_path, open_history, monkeypatch,
) -> None:
    session = _log(tmp_path, [
        _dictation(1, "Send it over."),
        _dictation(2, "Send it over."),
        _dictation(3, "Send it over."),
    ])
    for index in (1, 2, 3):
        _write_wav(session / "audio" / f"{index:04d}.wav")
    history = open_history()
    _select(history, 1)
    asked = _answer_confirmations(monkeypatch, answer=False)

    history._delete_button.invoke()
    assert len(_rows(history)) == 3

    _answer_confirmations(monkeypatch, answer=True)
    history._delete_button.invoke()

    assert asked == ["Delete dictation"]
    assert [row[1] for row in _rows(history)] == ["Send it over."] * 2
    assert not (session / "audio" / "0002.wav").exists()
    assert (session / "audio" / "0001.wav").exists()
    assert (session / "audio" / "0003.wav").exists()


def test_delete_all_clears_the_list_after_confirming(
    tmp_path, open_history, monkeypatch,
) -> None:
    _log(tmp_path, [_dictation(i, f"note {i} — ✓") for i in range(1, 41)])
    history = open_history()
    _answer_confirmations(monkeypatch, answer=True)

    history._delete_all_button.invoke()

    assert _rows(history) == []
    assert not _enabled(history._delete_all_button)
    assert not _enabled(history._delete_button)


def test_save_audio_as_copies_the_recording_under_a_dated_name(
    tmp_path, open_history, monkeypatch,
) -> None:
    session = _log(tmp_path, [_dictation(1, "keep this recording")])
    _write_wav(session / "audio" / "0001.wav", seconds=1.0)
    destination = tmp_path / "chosen.wav"
    offered: dict = {}

    def ask_where(**options) -> str:
        offered.update(options)
        return str(destination)

    monkeypatch.setattr(tk_window.filedialog, "asksaveasfilename", ask_where)
    history = open_history()
    _select(history, 0)

    history._export_button.invoke()

    assert offered["initialfile"].startswith("dictation-")
    assert offered["initialfile"].endswith(".wav")
    assert destination.read_bytes() == (session / "audio" / "0001.wav").read_bytes()
    assert history._status.cget("text") == "Saved chosen.wav."
    assert [row[1] for row in _rows(history)] == ["keep this recording"]


def test_save_audio_as_is_unavailable_without_a_recording(
    tmp_path, open_history,
) -> None:
    _log(tmp_path, [_dictation(1, "no audio kept", audio_path="")])
    history = open_history()

    _select(history, 0)

    assert not _enabled(history._export_button)
    assert _enabled(history._delete_button)
