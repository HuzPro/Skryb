import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from skryb.meetings.capture import RATE
from skryb.meetings.segmenter import Segment
from skryb.meetings.session import _Audio
from skryb.meetings.transcript import Entry, Event, Transcript, VOCABULARY_TEMPLATE, asr_context, is_echo, is_speaker_key

LONG = Segment(10.0, 14.0, 0)


def make(tmp_path: Path, events: list[Event] | None = None) -> Transcript:
    return Transcript(tmp_path, datetime(2026, 10, 16, 20, 20), (events if events is not None else []).append)


def test_mic_lines_that_repeat_what_the_speakers_played_are_dropped_as_echo(tmp_path: Path) -> None:
    t = make(tmp_path)
    t.add("them", Segment(10.0, 14.0, 0), "One of the perks of Lisbon is the tailor can come to you.")
    t.add("me", Segment(10.1, 13.8, 0), "perks of Lisbon is the tailor can come to")
    assert [e.track for e in t.entries] == ["them"]


def test_echo_is_dropped_whichever_track_finishes_first(tmp_path: Path) -> None:
    events: list[Event] = []
    t = make(tmp_path, events)
    t.add("me", Segment(10.1, 13.8, 0), "the tailor can come to you")
    t.add("them", Segment(10.0, 14.0, 1), "One of the perks of Lisbon is the tailor can come to you.")
    assert [e.track for e in t.entries] == ["them"]
    assert {"type": "remove", "entry_id": 0} in events


def test_the_same_words_at_different_times_are_not_echo() -> None:
    mic = Entry(0, "me", 0, 30.0, 31.0, "Yeah, exactly.")
    speakers = Entry(1, "them", 0, 10.0, 11.0, "Yeah, exactly.")
    assert not is_echo(mic, speakers)


def test_overlapping_talk_with_different_words_is_kept(tmp_path: Path) -> None:
    t = make(tmp_path)
    t.add("them", Segment(5.0, 9.0, 0), "So the estimator connects to LedgerLine first.")
    t.add("me", Segment(7.0, 8.0, 0), "Right, makes sense.")
    assert len(t.entries) == 2


def test_lines_with_no_words_are_never_echo() -> None:
    assert not is_echo(Entry(0, "me", 0, 1.0, 2.0, "..."), Entry(1, "them", 0, 1.0, 2.0, "Okay."))


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_transcriptions_are_ignored(tmp_path: Path, text: str) -> None:
    t = make(tmp_path)
    t.add("them", Segment(1.0, 2.0, 0), text)
    assert t.entries == []
    assert not (tmp_path / "transcript.md").exists()


def test_lines_arriving_out_of_order_are_kept_in_time_order(tmp_path: Path) -> None:
    t = make(tmp_path)
    t.add("me", Segment(20.0, 22.0, 0), "Second.")
    t.add("them", Segment(3.0, 5.0, 0), "First.")
    t.add("them", Segment(40.0, 41.0, 1), "Third.")
    assert [e.text for e in t.entries] == ["First.", "Second.", "Third."]
    saved = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert [e["text"] for e in saved["entries"]] == ["First.", "Second.", "Third."]


def test_renaming_a_speaker_relabels_every_line_and_the_saved_files(tmp_path: Path) -> None:
    events: list[Event] = []
    t = make(tmp_path, events)
    t.add("them", Segment(0.0, 2.0, 0), "All right, let's jump in.")
    t.add("them", Segment(3.0, 5.0, 1), "Sounds good.")
    t.add("them", Segment(6.0, 8.0, 0), "Launch Board first.")
    t.rename("them:0", "Leo Brandt")
    t.rename("them:1", "نادیہ رحمان")
    markdown = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert markdown.count("**Leo Brandt**") == 2
    assert "**نادیہ رحمان**" in markdown
    assert events[-1] == {"type": "names", "names": {"them:0": "Leo Brandt", "them:1": "نادیہ رحمان"}}


def test_clearing_a_name_goes_back_to_the_numbered_label(tmp_path: Path) -> None:
    t = make(tmp_path)
    t.add("them", Segment(0.0, 2.0, 2), "Hello.")
    t.rename("them:2", "Omar")
    t.rename("them:2", "   ")
    assert t.label("them:2") == "Speaker 3"
    assert t.label("me:0") == "Me"


def test_consecutive_lines_from_one_speaker_share_a_heading(tmp_path: Path) -> None:
    t = make(tmp_path)
    t.add("me", Segment(61.0, 62.0, 0), "I think that's just it.")
    t.add("me", Segment(63.0, 64.0, 0), "Yeah.")
    t.add("them", Segment(3725.0, 3726.0, 0), "That's amazing.")
    assert t.markdown() == (
        "# Meeting 2026-10-16 20:20\n\n\n**Me** [00:01:01]\nI think that's just it.\nYeah.\n\n"
        "**Speaker 1** [01:02:05]\nThat's amazing.\n"
    )


def test_an_hour_long_meeting_of_two_thousand_lines_saves_as_it_goes(tmp_path: Path) -> None:
    t = make(tmp_path)
    t.entries = [
        Entry(i, "them" if i % 3 else "me", i % 4, i * 1.8, i * 1.8 + 1.5, f"Line {i}: the estimator ✓ connects.")
        for i in range(1990)
    ]
    t._next_id = 1990
    started = time.monotonic()
    for i in range(1990, 2000):
        t.add("them" if i % 3 else "me", Segment(i * 1.8, i * 1.8 + 1.5, i % 4), f"Line {i}: the estimator ✓ connects.")
    assert (time.monotonic() - started) / 10 < 0.25  # a line arrives every ~2 s; saving must stay well under that
    saved = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert len(saved["entries"]) == 2000
    assert not list(tmp_path.glob("*.tmp"))


def test_a_saved_meeting_reads_back_with_its_names_and_lines(tmp_path: Path) -> None:
    t = make(tmp_path)
    t.add("them", Segment(0.0, 2.0, 0), "Hello there.")
    t.add("me", Segment(3.0, 4.0, 0), "Haan theek hai.")
    t.rename("them:0", "Marta Lindgren")

    again = Transcript.load(tmp_path)

    assert again.started == datetime(2026, 10, 16, 20, 20)
    assert again.names == {"them:0": "Marta Lindgren"}
    assert [(e.key, e.text) for e in again.entries] == [("them:0", "Hello there."), ("me:0", "Haan theek hai.")]
    again.add("them", Segment(5.0, 6.0, 1), "New line.")
    assert again.entries[-1].id == 2


def test_a_transcript_written_by_hand_skips_the_lines_it_cannot_read(tmp_path: Path) -> None:
    (tmp_path / "transcript.json").write_text(json.dumps({
        "started": "not a date",
        "names": {"them:0": "Speaker 1", "them:1": "Ivan Petrov", "../x": "intruder", "them:2": ""},
        "entries": [
            {"id": 0, "track": "them", "speaker": 0, "start": 1.0, "end": 2.0, "text": "Kept."},
            {"id": 1, "track": "them", "speaker": "x", "start": 2.0, "end": 3.0, "text": "Bad speaker."},
            {"id": 2, "track": "radio", "speaker": 0, "start": 3.0, "end": 4.0, "text": "Unknown track."},
            "not a line",
            {"id": 3, "track": "me", "speaker": 0, "start": 0.5, "end": 0.9, "text": "Earlier, kept."},
        ],
    }), encoding="utf-8")

    transcript = Transcript.load(tmp_path)

    assert [e.text for e in transcript.entries] == ["Earlier, kept.", "Kept."]
    assert transcript.names == {"them:1": "Ivan Petrov"}


def test_a_file_that_is_not_a_transcript_is_refused(tmp_path: Path) -> None:
    (tmp_path / "transcript.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        Transcript.load(tmp_path)


@pytest.mark.parametrize("key, valid", [
    ("them:0", True), ("me:0", True), ("them:12", True), ("them:", False), ("them:-1", False),
    ("../me:0", False), ("them:१", False), ("radio:0", False), ("", False),
])
def test_only_real_speaker_keys_are_accepted(key: str, valid: bool) -> None:
    assert is_speaker_key(key) is valid


def test_audio_slices_by_time_across_many_appends() -> None:
    audio = _Audio()
    tone = np.arange(RATE * 600, dtype=np.float32)  # ten minutes, each sample its own index
    for i in range(0, len(tone), 1600):
        audio.append(tone[i:i + 1600])
    piece = audio.between(123.5, 125.0)
    assert piece[0] == 123.5 * RATE and len(piece) == 1.5 * RATE
    assert len(audio.between(599.0, 700.0)) == RATE


# -- the vocabulary hint ---------------------------------------------------------


def test_terms_from_the_file_become_the_hint(tmp_path: Path) -> None:
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("Launch Board\n\n  BRIGHTLINE  \n# a comment\nQuoteDesk\n", encoding="utf-8")
    assert asr_context(vocabulary, {}, LONG) == "Names and terms that may come up: Launch Board, BRIGHTLINE, QuoteDesk."


def test_speaker_names_typed_in_are_added_once(tmp_path: Path) -> None:
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("Leo Brandt\n", encoding="utf-8")
    names = {"them:0": "Leo Brandt", "them:2": "نادیہ رحمان"}
    assert asr_context(vocabulary, names, LONG) == "Names and terms that may come up: Leo Brandt, نادیہ رحمان."


def test_short_clips_get_no_hint_because_it_makes_the_model_invent_words(tmp_path: Path) -> None:
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("Launch Board\n", encoding="utf-8")
    assert asr_context(vocabulary, {}, Segment(10.0, 11.9, 0)) is None
    assert asr_context(vocabulary, {}, Segment(10.0, 12.0, 0)) is not None


def test_no_file_and_no_names_means_no_hint(tmp_path: Path) -> None:
    assert asr_context(tmp_path / "missing.txt", {}, LONG) is None


def test_a_vocabulary_saved_in_another_encoding_is_skipped_rather_than_crashing(tmp_path: Path) -> None:
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_bytes("Zürich\n".encode("utf-16"))
    assert asr_context(vocabulary, {"them:0": "Omar"}, LONG) == "Names and terms that may come up: Omar."


def test_the_starter_file_is_all_comments_so_it_adds_nothing(tmp_path: Path) -> None:
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text(VOCABULARY_TEMPLATE, encoding="utf-8")
    assert asr_context(vocabulary, {}, LONG) is None


def test_a_vocabulary_of_hundreds_of_terms_is_passed_whole(tmp_path: Path) -> None:
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("\n".join(f"Term{i}" for i in range(500)), encoding="utf-8")
    hint = asr_context(vocabulary, {}, LONG)
    assert hint is not None
    assert hint.count(", ") == 499 and hint.endswith("Term499.")
