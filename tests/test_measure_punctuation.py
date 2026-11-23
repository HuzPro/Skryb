"""The punctuation measurement reads logs nobody can reproduce and audio nobody may share.

So every log, clip and sentence here is invented: the logs carry the faults
real ones do (half-written lines, duplicates, blank fields, other scripts),
the clips are synthetic speech and room noise at the levels the session logs
record, and the engine is a stand-in that splits a sentence at a long pause,
which is the habit being measured.
"""

import json
import re
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.measure_punctuation import (
    DEFAULT_MODEL,
    _Dictation,
    judged_places,
    locate,
    main,
    place_in_clip,
    punctuation_spots,
    sample_clips,
    sentences,
    shorten_pauses,
)
from skryb.recording.audio_segment import load_wav

pytestmark = pytest.mark.integration

_RATE = 16000
# The room as the session logs record it on a desk microphone: about -56 dBFS.
_ROOM = 0.0016
_FADE = 80
_OTHER_MODEL = "parakeet-ultra"
_ENGINE = "native_qwen3_asr"


# --- Synthetic audio ---------------------------------------------------------


def _room(seconds: float, rng: np.random.Generator, level: float = _ROOM, rate: int = _RATE) -> np.ndarray:
    return rng.normal(0.0, level, round(seconds * rate)).astype(np.float32)


def _speech(
    seconds: float, rng: np.random.Generator, level: float = 0.05, rate: int = _RATE, room: float = _ROOM,
) -> np.ndarray:
    """A voiced sound with a syllable rhythm, over the room's noise, never dropping to the room's level."""
    t = np.arange(round(seconds * rate)) / rate
    voice = sum(np.sin(2 * np.pi * 150 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 7))
    syllables = 0.35 + 0.65 * np.abs(np.sin(np.pi * 4 * t))
    signal = voice * syllables
    signal = level * signal / np.sqrt(np.mean(signal**2))
    return (signal + rng.normal(0.0, room, t.size)).astype(np.float32)


def _whisper(seconds: float, rng: np.random.Generator, level: float = 4 * _ROOM) -> np.ndarray:
    """Breath noise with a syllable rhythm, about 12 dB over the room: quieter than talking, louder than the room."""
    t = np.arange(round(seconds * _RATE)) / _RATE
    breath = rng.normal(0.0, 1.0, t.size) * (0.3 + 0.7 * np.abs(np.sin(np.pi * 4 * t)))
    breath = level * breath / np.sqrt(np.mean(breath**2))
    return (breath + rng.normal(0.0, _ROOM, t.size)).astype(np.float32)


def _position_of(haystack: np.ndarray, needle: np.ndarray) -> int | None:
    for start in np.flatnonzero(haystack == needle[0]):
        if np.array_equal(haystack[start:start + needle.size], needle):
            return int(start)
    return None


def _largest_step(samples: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(samples.astype(np.float64)))))


# --- Shortening pauses ---------------------------------------------------------


class TestShorteningPauses:
    def test_a_pause_longer_than_the_cap_is_cut_from_its_middle(self) -> None:
        rng = np.random.default_rng(1)
        clip = np.concatenate([_speech(1.0, rng), _room(1.5, rng), _speech(1.0, rng)])
        pause_start, pause_end = 16000, 40000

        shortened = shorten_pauses(clip, _RATE, 300)

        removed = clip.size - shortened.size
        assert removed == (pause_end - pause_start) - 4800
        cut_from = int(np.flatnonzero(shortened[: clip.size] != clip[: shortened.size])[0])
        assert cut_from - pause_start == 2400
        assert pause_end - (cut_from + removed) == 2400
        assert np.array_equal(shortened[:cut_from], clip[:cut_from])
        assert np.array_equal(shortened[cut_from + _FADE:], clip[cut_from + removed + _FADE:])

    def test_what_is_left_of_the_pause_is_the_rooms_own_noise_and_never_inserted_zeros(self) -> None:
        rng = np.random.default_rng(2)
        clip = np.concatenate([_speech(1.0, rng), _room(2.0, rng), _speech(1.0, rng)])

        shortened = shorten_pauses(clip, _RATE, 300)

        left = shortened[16000:16000 + 4800]
        assert np.count_nonzero(shortened == 0.0) == np.count_nonzero(clip == 0.0) == 0
        assert np.sqrt(np.mean(left[:2000] ** 2)) == pytest.approx(_ROOM, rel=0.15)
        assert np.sqrt(np.mean(left[-2000:] ** 2)) == pytest.approx(_ROOM, rel=0.15)
        assert shortened.dtype == np.float32

    def test_each_join_is_crossfaded_so_there_is_no_click(self) -> None:
        rng = np.random.default_rng(3)
        # A mains hum is the room noise a hard cut clicks on: the two ends meet out of phase.
        t = np.arange(round(1.52 * _RATE)) / _RATE
        hum = (0.0015 * np.sin(2 * np.pi * 130 * t)).astype(np.float32)
        clip = np.concatenate([_speech(1.0, rng, room=0.0), hum, _speech(1.0, rng, room=0.0)])

        shortened = shorten_pauses(clip, _RATE, 300)

        assert shortened.size < clip.size
        join = int(np.flatnonzero(shortened[: clip.size] != clip[: shortened.size])[0])
        assert _largest_step(shortened[join - 100:join + _FADE + 100]) < 3 * _largest_step(hum)

    @pytest.mark.parametrize(
        ("pause_seconds", "removed"),
        [(0.2, 0), (0.3, 0), (0.32, 320), (1.0, 11200), (12.0, 187200)],
    )
    def test_only_a_pause_longer_than_the_cap_is_shortened(self, pause_seconds: float, removed: int) -> None:
        rng = np.random.default_rng(4)
        clip = np.concatenate([_speech(1.0, rng), _room(pause_seconds, rng), _speech(1.0, rng)])

        assert clip.size - shorten_pauses(clip, _RATE, 300).size == removed

    def test_leading_and_trailing_silence_are_left_as_they_are(self) -> None:
        rng = np.random.default_rng(5)
        clip = np.concatenate([_room(2.0, rng), _speech(1.0, rng), _room(2.5, rng)])

        assert shorten_pauses(clip, _RATE, 300) is clip

    def test_a_clip_with_no_long_pause_comes_back_unchanged(self) -> None:
        rng = np.random.default_rng(6)
        parts = [_speech(0.6, rng)]
        for _ in range(10):
            parts += [_room(0.2, rng), _speech(0.6, rng)]
        clip = np.concatenate(parts)
        original = clip.copy()

        shortened = shorten_pauses(clip, _RATE, 300)

        assert shortened is clip
        assert np.array_equal(shortened, original)

    def test_a_whispered_phrase_between_two_long_pauses_is_kept_whole(self) -> None:
        rng = np.random.default_rng(7)
        whisper = _whisper(1.5, rng)
        clip = np.concatenate([_speech(1.0, rng), _room(0.8, rng), whisper, _room(0.8, rng), _speech(1.0, rng)])

        shortened = shorten_pauses(clip, _RATE, 300)

        assert clip.size - shortened.size == 2 * (round(0.8 * _RATE) - 4800)
        assert _position_of(shortened, whisper) is not None

    def test_quiet_speech_at_the_edge_of_a_pause_is_never_cut_whatever_the_cap(self) -> None:
        rng = np.random.default_rng(8)
        # A word trailing off so faintly that no level can tell it from the room.
        trailing_off = _whisper(0.1, rng, level=0.8 * _ROOM)
        clip = np.concatenate([_speech(1.0, rng), trailing_off, _room(1.0, rng), _speech(1.0, rng)])

        shortened = shorten_pauses(clip, _RATE, 20)

        assert shortened.size < clip.size
        assert np.array_equal(shortened[:16000 + 1600], clip[:16000 + 1600])

    def test_in_a_noisy_room_nothing_is_taken_for_a_pause(self) -> None:
        rng = np.random.default_rng(9)
        noisy = 0.02
        clip = np.concatenate([
            _speech(1.0, rng, level=0.08, room=noisy), _room(1.5, rng, level=noisy), _speech(1.0, rng, level=0.08, room=noisy),
        ])

        assert shorten_pauses(clip, _RATE, 300) is clip

    @pytest.mark.parametrize("length", [0, 1, 319, 320, 3 * _RATE])
    def test_digital_silence_and_clips_too_short_to_read_come_back_unchanged(self, length: int) -> None:
        clip = np.zeros(length, dtype=np.float32)

        assert shorten_pauses(clip, _RATE, 300) is clip

    def test_pauses_are_found_at_any_sample_rate(self) -> None:
        rng = np.random.default_rng(10)
        rate = 8000
        clip = np.concatenate([_speech(1.0, rng, rate=rate), _room(1.0, rng, rate=rate), _speech(1.0, rng, rate=rate)])

        assert clip.size - shorten_pauses(clip, rate, 300).size == 8000 - 2400

    def test_a_long_dictation_with_a_thousand_pauses_is_shortened_at_every_one(self) -> None:
        rng = np.random.default_rng(11)
        rate = 8000
        # The same phrase over and over, as a dictation that repeats itself does.
        phrase = _speech(0.3, rng, rate=rate)
        parts = [phrase]
        for _ in range(1000):
            parts += [_room(0.6, rng, rate=rate), phrase]
        clip = np.concatenate(parts)

        shortened = shorten_pauses(clip, rate, 300)

        assert clip.size - shortened.size == 1000 * (4800 - 2400)
        assert np.array_equal(shortened[-phrase.size:], phrase)


# --- Log fixtures ----------------------------------------------------------------


def _transcription(index: int, text: object = "", seconds: object = 5.0, **fields: object) -> dict[str, object]:
    event: dict[str, object] = {
        "event": "transcription",
        "index": index,
        "requested_at": "2026-11-20T10:00:00Z",
        "finished_at": "2026-11-20T10:00:01Z",
        "model": DEFAULT_MODEL,
        "engine": _ENGINE,
        "device": "auto",
        "audio_seconds": seconds,
        "sample_rate": _RATE,
        "audio_path": f"audio/{index:04d}.wav",
        "text": text,
        "error": None,
    }
    event.update(fields)
    return event


def _write_session(logs: Path, name: str, lines: list[object]) -> Path:
    session = logs / name
    session.mkdir(parents=True, exist_ok=True)
    rendered = [line if isinstance(line, str) else json.dumps(line, ensure_ascii=False) for line in lines]
    (session / "session.jsonl").write_text("".join(line + "\n" for line in rendered), encoding="utf-8")
    return session


def _write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(_RATE)
        output.writeframes(pcm.tobytes())


def _run_checks(tmp_path: Path, *extra: str) -> dict:
    out = tmp_path / "eval"
    main(["checks", "--logs", str(tmp_path / "logs"), "--out", str(out), *extra])
    model = extra[extra.index("--model") + 1] if "--model" in extra else DEFAULT_MODEL
    return json.loads((out / f"checks-{model}.json").read_text(encoding="utf-8"))


def _engine_text(report: dict, bucket: str = "all") -> dict[str, int]:
    return report["fields"]["text"][bucket]


# --- checks: reading the logs ----------------------------------------------------


class TestReadingTheLogs:
    def test_only_dictations_of_the_chosen_model_are_counted(self, tmp_path: Path) -> None:
        _write_session(tmp_path / "logs", "s1", [
            _transcription(1, "The lantern is lit."),
            _transcription(2, "The orchard is quiet."),
            _transcription(3, "The kettle is on.", model=_OTHER_MODEL),
            _transcription(4, "The kettle is off.", model=_OTHER_MODEL),
            _transcription(5, "The kettle is cold.", model=_OTHER_MODEL),
        ])

        qwen = _run_checks(tmp_path)
        parakeet = _run_checks(tmp_path, "--model", _OTHER_MODEL)

        assert _engine_text(qwen)["dictations"] == 2
        assert _engine_text(parakeet)["dictations"] == 3
        assert qwen["models_in_logs"] == {_OTHER_MODEL: 3, DEFAULT_MODEL: 2}

    def test_corrupt_and_half_written_lines_are_skipped_and_the_rest_counted(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        session = _write_session(logs, "s1", [
            {"event": "app_started"},
            _transcription(1, "The lantern is lit."),
            "{not json at all",
            "[1, 2, 3]",
            '"a bare string"',
            "",
            "   ",
            _transcription(2, "The orchard is quiet."),
        ])
        # The app was writing this line when the folder was read: it stops inside the "؟".
        cut_short = _transcription(3)
        del cut_short["text"]
        cut_short["text"] = "هل وصل الطرد؟"
        with (session / "session.jsonl").open("ab") as stream:
            stream.write(json.dumps(cut_short, ensure_ascii=False).encode("utf-8")[:-3])
        (logs / "s2-no-log").mkdir()
        (logs / "stray.txt").write_text("not a session", encoding="utf-8")

        report = _run_checks(tmp_path)

        assert _engine_text(report)["dictations"] == 2

    def test_a_line_logged_twice_counts_once_but_the_same_words_dictated_twice_count_twice(
        self, tmp_path: Path,
    ) -> None:
        line = _transcription(1, "Light the lantern and. Then wait.")
        _write_session(tmp_path / "logs", "s1", [line, line, _transcription(2, "Light the lantern and. Then wait.")])

        report = _run_checks(tmp_path)

        assert _engine_text(report)["dictations"] == 2
        assert _engine_text(report)["never_final_app"] == 2

    def test_a_dictation_with_blank_or_missing_text_is_not_counted(self, tmp_path: Path) -> None:
        missing = _transcription(3)
        del missing["text"]
        _write_session(tmp_path / "logs", "s1", [
            _transcription(1, ""),
            _transcription(2, "   \n "),
            missing,
            _transcription(4, 42),
            _transcription(5, None),
            _transcription(6, "The lantern is lit.", typed_text="The lantern is lit. "),
            _transcription(7, "The orchard is quiet."),
            _transcription(8, "The kettle is on.", typed_text=""),
        ])

        report = _run_checks(tmp_path)

        assert _engine_text(report)["dictations"] == 3
        assert report["fields"]["typed_text"]["all"]["dictations"] == 1

    def test_take_lengths_split_at_15_30_and_60_seconds(self, tmp_path: Path) -> None:
        lengths: list[object] = [0, 14.99, 15, 29.99, 30.0, 59.99, 60, 600.5, None, -1, "12", True]
        _write_session(tmp_path / "logs", "s1", [
            _transcription(index, "The lantern is lit.", seconds=seconds)
            for index, seconds in enumerate(lengths, start=1)
        ])

        report = _run_checks(tmp_path)

        buckets = {name: tally["dictations"] for name, tally in report["fields"]["text"].items()}
        assert buckets == {
            "all": 12, "under 15 s": 2, "15-30 s": 2, "30-60 s": 2, "60 s and over": 2, "length unknown": 4,
        }


# --- checks: what is counted -----------------------------------------------------


class TestWhatChecksCount:
    def test_a_full_stop_after_and_counts_on_both_lists_and_after_to_only_on_the_wider_list(
        self, tmp_path: Path,
    ) -> None:
        _write_session(tmp_path / "logs", "s1", [
            _transcription(1, "Bring the lantern and. Walk to. The orchard is far."),
            _transcription(2, "Is it the? Yes."),
            # A word alone in its sentence may be the whole thought, as the app's rule allows.
            _transcription(3, "And. So."),
        ])

        tally = _engine_text(_run_checks(tmp_path))

        assert tally["never_final_app"] == 2
        assert tally["never_final_wide"] == 3

    def test_sentences_opening_with_and_but_so_or_are_counted_with_the_first_sentence_kept_apart(
        self, tmp_path: Path,
    ) -> None:
        _write_session(tmp_path / "logs", "s1", [
            _transcription(1, "And the lantern went out. But the orchard stayed bright. So we waited."),
            _transcription(2, "We waited. Or did we leave? Andes peaks rose. Sofas stayed."),
        ])

        tally = _engine_text(_run_checks(tmp_path))

        assert (tally["sentences"], tally["starts"]) == (7, 4)
        assert (tally["inner_sentences"], tally["inner_starts"]) == (5, 3)

    def test_fragments_under_three_words_are_counted_only_in_dictations_of_two_or_more_sentences(
        self, tmp_path: Path,
    ) -> None:
        _write_session(tmp_path / "logs", "s1", [
            _transcription(1, "Yes."),
            _transcription(2, "Okay then."),
            _transcription(3, "The lantern is lit. Okay. Then we walk to the orchard. Right then."),
        ])

        tally = _engine_text(_run_checks(tmp_path))

        assert tally["fragments"] == 2
        assert tally["sentences_in_multi"] == 4

    def test_a_dictation_of_five_words_or_more_without_a_closing_mark_is_counted(self, tmp_path: Path) -> None:
        _write_session(tmp_path / "logs", "s1", [
            _transcription(1, "the lantern is lit"),
            _transcription(2, "the lantern is lit now"),
            _transcription(3, "The lantern is lit now,"),
            _transcription(4, "The lantern is lit now."),
            _transcription(5, "The lantern is lit. And the orchard is"),
        ])

        tally = _engine_text(_run_checks(tmp_path))

        assert (tally["five_word_dictations"], tally["no_final_mark"]) == (4, 3)

    def test_question_marks_at_the_end_and_before_it_are_counted_in_every_script(self, tmp_path: Path) -> None:
        _write_session(tmp_path / "logs", "s1", [
            _transcription(1, "Is the lantern lit? I think so."),
            _transcription(2, "Did the orchard flood?"),
            _transcription(3, "هل وصل الطرد؟ نعم وصل."),
            _transcription(4, "میں کل آؤں گا۔ ٹھیک ہے؟"),
            _transcription(5, "今天下雨。你带伞了吗？"),
            _transcription(6, "Is it? Is it really? Yes!"),
        ])

        tally = _engine_text(_run_checks(tmp_path))

        assert tally["ends_question"] == 3
        assert tally["question_marks"] == 7
        assert tally["inner_questions"] == 4

    def test_commas_and_full_stops_before_and_but_so_or_are_counted_apart(self, tmp_path: Path) -> None:
        _write_session(tmp_path / "logs", "s1", [
            _transcription(
                1,
                "We packed the lantern, and the kettle. But the map stayed home, so we turned back. "
                "Andes peaks rose, Oregon too. Or not.",
            ),
        ])

        tally = _engine_text(_run_checks(tmp_path))

        assert (tally["comma_before"], tally["stop_before"]) == (2, 2)

    def test_decimals_and_version_numbers_are_not_sentence_ends(self) -> None:
        found = sentences("Version 0.17.1 shipped with 3.5 times the speed. Nice")

        assert [sentence.text for sentence in found] == ["Version 0.17.1 shipped with 3.5 times the speed.", "Nice"]
        assert [sentence.mark for sentence in found] == [".", ""]

    def test_the_engine_text_and_the_typed_text_are_counted_apart(self, tmp_path: Path) -> None:
        _write_session(tmp_path / "logs", "s1", [
            _transcription(1, "Bring the lantern and. Then wait.", typed_text="Bring the lantern and then wait. "),
        ])

        report = _run_checks(tmp_path)

        assert report["fields"]["text"]["all"]["never_final_app"] == 1
        assert report["fields"]["typed_text"]["all"]["never_final_app"] == 0
        assert report["fields"]["typed_text"]["all"]["sentences"] == 1

    def test_three_thousand_dictations_are_counted_exactly(self, tmp_path: Path) -> None:
        texts = [
            "Bring the lantern and. Then walk to the orchard.",
            "Is the kettle on? But the stove is cold.",
            "the orchard gate stays open",
        ]
        for session in range(100):
            _write_session(tmp_path / "logs", f"s{session:03d}", [
                _transcription(index, texts[index % 3], seconds=float((session + index) % 70))
                for index in range(30)
            ])

        report = _run_checks(tmp_path)

        tally = _engine_text(report)
        assert tally["dictations"] == 3000
        assert tally["sentences"] == 5000
        assert tally["never_final_app"] == 1000
        assert tally["inner_starts"] == 1000
        assert tally["inner_questions"] == 1000
        assert tally["no_final_mark"] == 1000
        assert sum(report["fields"]["text"][name]["dictations"] for name in report["fields"]["text"] if name != "all") == 3000


# --- checks: where the results go -------------------------------------------------


_INVENTED_WORDS = ("lantern", "orchard", "kettle", "الطرد")


def _invented_logs(tmp_path: Path) -> None:
    _write_session(tmp_path / "logs", "s1", [
        _transcription(1, "Bring the lantern and. Then walk to the orchard.", typed_text="Bring the lantern."),
        _transcription(2, "Is the kettle on? But it is.", seconds=40.0),
        _transcription(3, "هل وصل الطرد؟ نعم."),
    ])


class TestWhereChecksWrite:
    def test_the_console_and_the_report_show_counts_and_never_a_dictated_word(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _invented_logs(tmp_path)

        report = _run_checks(tmp_path)

        shown = capsys.readouterr()
        written = json.dumps(report, ensure_ascii=False)
        for word in _INVENTED_WORDS:
            assert word not in shown.out + shown.err
            assert word not in written
        assert "dictations" in shown.out

    def test_example_sentences_are_written_only_when_asked_and_only_to_their_own_file(self, tmp_path: Path) -> None:
        _invented_logs(tmp_path)
        out = tmp_path / "eval"

        _run_checks(tmp_path)
        assert sorted(path.name for path in out.iterdir()) == [f"checks-{DEFAULT_MODEL}.json"]

        _run_checks(tmp_path, "--examples", "1")
        lines = (out / f"checks-{DEFAULT_MODEL}-examples.jsonl").read_text(encoding="utf-8").splitlines()
        examples = [json.loads(line) for line in lines]
        assert examples
        assert len({(example["field"], example["check"]) for example in examples}) == len(examples)
        assert {"field", "check", "id", "sentence"} == set(examples[0])

    def test_results_are_refused_inside_the_logs_folder(self, tmp_path: Path) -> None:
        _invented_logs(tmp_path)

        with pytest.raises(SystemExit, match="inside the logs folder"):
            main(["checks", "--logs", str(tmp_path / "logs"), "--out", str(tmp_path / "logs" / "eval")])
        assert not (tmp_path / "logs" / "eval").exists()

    def test_results_are_refused_inside_a_git_checkout(self, tmp_path: Path) -> None:
        _invented_logs(tmp_path)
        (tmp_path / "project" / ".git").mkdir(parents=True)

        with pytest.raises(SystemExit, match="inside the git checkout"):
            main(["checks", "--logs", str(tmp_path / "logs"), "--out", str(tmp_path / "project" / "deep" / "eval")])
        assert not (tmp_path / "project" / "deep").exists()

    def test_results_are_refused_inside_this_repository(self, tmp_path: Path) -> None:
        _invented_logs(tmp_path)
        inside = Path(__file__).resolve().parents[1] / "punctuation-eval-should-not-exist"

        with pytest.raises(SystemExit, match="inside the git checkout"):
            main(["checks", "--logs", str(tmp_path / "logs"), "--out", str(inside)])
        assert not inside.exists()

    def test_a_missing_logs_folder_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="No such logs folder"):
            main(["checks", "--logs", str(tmp_path / "nowhere"), "--out", str(tmp_path / "eval")])


# --- replay ------------------------------------------------------------------------


_SPLIT = "Light the lantern. And walk to the orchard."
_JOINED = "Light the lantern, and walk to the orchard."


def _longest_inner_pause(samples: np.ndarray, sample_rate: int) -> float:
    frame = sample_rate // 50
    whole = samples.size // frame
    levels = np.sqrt(np.mean(samples[: whole * frame].reshape(whole, frame).astype(np.float64) ** 2, axis=1))
    quiet = np.concatenate(([0], (levels < 4 * _ROOM).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(quiet))
    inner = [(end - start) / 50 for start, end in zip(edges[::2], edges[1::2]) if start > 0 and end < whole]
    return max(inner, default=0.0)


class _PauseHearingEngine:
    """Ends a sentence at any pause over half a second, as the speech model is suspected of doing."""

    def __init__(self, drops_a_word_when_joined: bool = False) -> None:
        self.heard: list[tuple[np.dtype, int, int]] = []
        self.clips: list[np.ndarray] = []
        self.built_for: list[tuple[str, str]] = []
        self.closed = 0
        self._drops_a_word = drops_a_word_when_joined

    def build(self, model: str, engine: str) -> tuple:
        self.built_for.append((model, engine))
        return self.decode, self.close

    def decode(self, samples: np.ndarray, sample_rate: int) -> str:
        self.heard.append((samples.dtype, sample_rate, samples.size))
        self.clips.append(samples.copy())
        if _longest_inner_pause(samples, sample_rate) > 0.5:
            return _SPLIT
        return _JOINED.replace("the orchard", "orchard") if self._drops_a_word else _JOINED

    def close(self) -> None:
        self.closed += 1


class _WobblingEngine(_PauseHearingEngine):
    """Writes a comma it did not write before whenever it hears the very same audio a second time."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[bytes] = set()

    def decode(self, samples: np.ndarray, sample_rate: int) -> str:
        text = super().decode(samples, sample_rate)
        fingerprint = samples.tobytes()
        if fingerprint in self._seen:
            return text.replace("Light the", "Light, the")
        self._seen.add(fingerprint)
        return text


_SESSION = "20261120T100000Z-0a1b2c3d"


def _replay_logs(tmp_path: Path) -> Path:
    """Two clips to replay, plus one of each kind the replay must step over."""
    rng = np.random.default_rng(20)
    logs = tmp_path / "logs"
    session = _write_session(logs, "20261120T100000Z-0a1b2c3d", [
        _transcription(1, _SPLIT, seconds=3.5),
        _transcription(2, "Light the lantern, and walk to the orchards.", seconds=2.2),
        _transcription(3, "The kettle is on.", seconds=2.0),
        _transcription(4, "The kettle is off.", seconds=2.0),
        _transcription(5, "The kettle is cold.", model=_OTHER_MODEL, engine="nano_parakeet"),
        _transcription(6, "The kettle is warm.", engine="qwen3_asr"),
        _transcription(7, "Nothing saved.", audio_path=""),
    ])
    _write_wav(session / "audio" / "0001.wav", np.concatenate([_speech(1.0, rng), _room(1.5, rng), _speech(1.0, rng)]))
    _write_wav(session / "audio" / "0002.wav", np.concatenate([_speech(1.0, rng), _room(0.2, rng), _speech(1.0, rng)]))
    # 0003 was deleted with its session's audio; 0004 is a file cut short on disk.
    (session / "audio" / "0004.wav").write_bytes(b"RIFF\x00\x00")
    _write_wav(session / "audio" / "0005.wav", _speech(1.0, rng))
    _write_wav(session / "audio" / "0006.wav", _speech(1.0, rng))
    return logs


def _run_replay(
    tmp_path: Path, engine: _PauseHearingEngine, *caps: int, limit: int | None = None, extra: tuple[str, ...] = (),
) -> dict:
    out = tmp_path / "eval"
    arguments = ["replay", "--logs", str(tmp_path / "logs"), "--out", str(out), *extra]
    for cap in caps or (300, 800):
        arguments += ["--cap", str(cap)]
    if limit is not None:
        arguments += ["--limit", str(limit)]
    main(arguments, build_decoder=engine.build)
    return json.loads((out / f"replay-{DEFAULT_MODEL}-summary.json").read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestReplay:
    def test_each_clip_is_decoded_raw_and_once_per_cap_and_the_marks_compared(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)

        summary = _run_replay(tmp_path, _PauseHearingEngine())

        assert (summary["clips"], summary["failed"], summary["audio_gone"]) == (2, 1, 1)
        raw, short, long = (summary["variants"][name] for name in ("raw", "300 ms", "800 ms"))
        assert (raw["full_stops"], short["full_stops"], long["full_stops"]) == (3, 2, 3)
        assert (short["clips_fewer_full_stops"], short["clips_more_full_stops"]) == (1, 0)
        assert (long["clips_fewer_full_stops"], long["clips_decoded"]) == (0, 1)
        assert short["seconds_removed"] == pytest.approx(1.2, abs=0.05)
        assert short["clips_words_changed"] == 0

    def test_the_engine_is_built_once_for_the_model_and_the_engine_most_of_its_clips_used(
        self, tmp_path: Path,
    ) -> None:
        _replay_logs(tmp_path)
        engine = _PauseHearingEngine()

        _run_replay(tmp_path, engine)

        assert engine.built_for == [(DEFAULT_MODEL, _ENGINE)]
        assert engine.closed == 1

    def test_the_decoder_hears_the_saved_clip_as_float32_at_its_own_rate(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)
        engine = _PauseHearingEngine()

        _run_replay(tmp_path, engine)

        assert {(dtype, rate) for dtype, rate, _ in engine.heard} == {(np.dtype(np.float32), _RATE)}

    def test_a_clip_with_no_long_pause_is_decoded_once_not_once_per_cap(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)
        engine = _PauseHearingEngine()

        _run_replay(tmp_path, engine)

        # The long-pause clip: raw, 300 ms and 800 ms. The other: raw only.
        assert len(engine.heard) == 4

    def test_every_place_raw_and_trimmed_differ_is_written_for_listening(self, tmp_path: Path) -> None:
        logs = _replay_logs(tmp_path)

        _run_replay(tmp_path, _PauseHearingEngine())

        differences = _jsonl(tmp_path / "eval" / f"replay-{DEFAULT_MODEL}-differences.jsonl")
        assert len(differences) == 1
        difference = differences[0]
        assert difference["id"] == "20261120T100000Z-0a1b2c3d/0001"
        assert (difference["variant"], difference["cap_ms"]) == ("300 ms", 300)
        assert Path(difference["audio_path"]) == logs / "20261120T100000Z-0a1b2c3d" / "audio" / "0001.wav"
        assert (difference["raw_text"], difference["trimmed_text"]) == (_SPLIT, _JOINED)
        assert (difference["spot"]["raw_mark"], difference["spot"]["trimmed_mark"]) == (".", ",")

    def test_the_raw_replay_is_checked_against_the_logged_text(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)

        summary = _run_replay(tmp_path, _PauseHearingEngine())

        assert summary["logged_text"] == 2
        assert summary["reproduced_exactly"] == 1
        assert summary["reproduced_word_for_word"] == 1

    def test_a_trim_that_changes_the_words_is_counted(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)

        summary = _run_replay(tmp_path, _PauseHearingEngine(drops_a_word_when_joined=True))

        assert summary["variants"]["300 ms"]["clips_words_changed"] == 1
        assert summary["variants"]["800 ms"]["clips_words_changed"] == 0

    def test_a_second_run_decodes_nothing_already_finished_and_loads_no_engine(self, tmp_path: Path) -> None:
        logs = _replay_logs(tmp_path)
        first = _run_replay(tmp_path, _PauseHearingEngine())
        # A failed clip is tried again on every run, so take the broken file away.
        (logs / "20261120T100000Z-0a1b2c3d" / "audio" / "0004.wav").unlink()
        again = _PauseHearingEngine()

        second = _run_replay(tmp_path, again)

        assert again.built_for == []
        assert again.heard == []
        assert second["variants"] == first["variants"]

    def test_a_run_cut_off_mid_line_redoes_only_the_clip_it_was_on(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)
        _run_replay(tmp_path, _PauseHearingEngine())
        results = tmp_path / "eval" / f"replay-{DEFAULT_MODEL}.jsonl"
        lines = results.read_text(encoding="utf-8").splitlines()
        results.write_text(lines[0] + "\n" + lines[1][: len(lines[1]) // 2], encoding="utf-8")
        again = _PauseHearingEngine()

        summary = _run_replay(tmp_path, again)

        assert len(again.heard) in (1, 3)
        assert summary["clips"] == 2
        # The redone record starts on a line of its own instead of being glued to the broken one.
        lines = results.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert {json.loads(lines[0])["id"], json.loads(lines[2])["id"]} == {
            "20261120T100000Z-0a1b2c3d/0001", "20261120T100000Z-0a1b2c3d/0002",
        }

    def test_asking_for_a_new_cap_decodes_only_that_cap(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)
        _run_replay(tmp_path, _PauseHearingEngine(), 300)
        again = _PauseHearingEngine()

        summary = _run_replay(tmp_path, again, 300, 500)

        # Only the long-pause clip has anything for 500 ms to cut; its raw text is kept from the first run.
        assert len(again.heard) == 1
        assert set(summary["variants"]) == {"raw", "300 ms", "500 ms"}

    def test_limit_takes_a_seeded_sample_of_the_clips(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)
        engine = _PauseHearingEngine()

        summary = _run_replay(tmp_path, engine, limit=1)

        assert summary["clips"] + summary["failed"] == 1

    def test_the_console_and_the_per_dictation_file_show_counts_and_never_a_dictated_word(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _replay_logs(tmp_path)

        _run_replay(tmp_path, _PauseHearingEngine())

        shown = capsys.readouterr()
        per_dictation = (tmp_path / "eval" / f"replay-{DEFAULT_MODEL}-per-dictation.jsonl").read_text(encoding="utf-8")
        summary = (tmp_path / "eval" / f"replay-{DEFAULT_MODEL}-summary.json").read_text(encoding="utf-8")
        for word in _INVENTED_WORDS:
            assert word not in shown.out + shown.err
            assert word not in per_dictation
            assert word not in summary

    def test_a_cap_of_zero_or_less_is_refused(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)

        with pytest.raises(SystemExit):
            _run_replay(tmp_path, _PauseHearingEngine(), 0)


class TestReplayControls:
    def test_with_the_control_each_clip_is_decoded_again_as_it_is_and_without_its_first_20_ms(
        self, tmp_path: Path,
    ) -> None:
        logs = _replay_logs(tmp_path)
        engine = _PauseHearingEngine()

        _run_replay(tmp_path, engine, extra=("--control",))

        # Long-pause clip: raw, 300 ms, 800 ms, repeat, shift. The other: raw, repeat, shift.
        assert len(engine.heard) == 8
        recorded = load_wav(logs / _SESSION / "audio" / "0001.wav").data
        assert sum(np.array_equal(clip, recorded) for clip in engine.clips) == 2
        assert sum(np.array_equal(clip, recorded[320:]) for clip in engine.clips) == 1

    def test_the_controls_are_counted_and_compared_with_raw_like_the_caps(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)

        summary = _run_replay(tmp_path, _PauseHearingEngine(), extra=("--control",))

        repeat, shift = summary["variants"]["repeat"], summary["variants"]["shift"]
        assert set(repeat) == set(shift) == set(summary["variants"]["300 ms"])
        assert (repeat["clips_decoded"], shift["clips_decoded"]) == (2, 2)
        assert (repeat["seconds_removed"], shift["seconds_removed"]) == (0.0, 0.0)
        assert (repeat["spots"], shift["spots"], repeat["clips_words_changed"]) == (0, 0, 0)
        entries = _jsonl(tmp_path / "eval" / f"replay-{DEFAULT_MODEL}-per-dictation.jsonl")
        assert all({"raw", "300 ms", "800 ms", "repeat", "shift"} <= set(entry) for entry in entries)
        assert {entry["shift"]["removed_seconds"] for entry in entries} == {0.02}

    def test_an_engine_that_wobbles_run_to_run_shows_up_in_the_repeat_control(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)

        summary = _run_replay(tmp_path, _WobblingEngine(), extra=("--control",))

        assert summary["variants"]["repeat"]["spots"] == 2
        assert summary["variants"]["shift"]["spots"] == 0
        differences = _jsonl(tmp_path / "eval" / f"replay-{DEFAULT_MODEL}-differences.jsonl")
        repeats = [difference for difference in differences if difference["variant"] == "repeat"]
        assert len(repeats) == 2
        assert {difference["cap_ms"] for difference in repeats} == {None}
        assert {(d["spot"]["raw_mark"], d["spot"]["trimmed_mark"]) for d in repeats} == {("", ",")}

    def test_a_clip_replayed_without_the_control_gets_only_the_controls_decoded_when_asked_again(
        self, tmp_path: Path,
    ) -> None:
        _replay_logs(tmp_path)
        _run_replay(tmp_path, _PauseHearingEngine())
        again = _PauseHearingEngine()

        summary = _run_replay(tmp_path, again, extra=("--control",))

        assert len(again.heard) == 2 * 2
        assert summary["clips"] == 2
        assert summary["variants"]["raw"]["full_stops"] == 3

    def test_a_result_file_written_before_the_controls_existed_is_still_read(self, tmp_path: Path) -> None:
        logs = _replay_logs(tmp_path)
        out = tmp_path / "eval"
        out.mkdir()

        def record(index: int, raw: str, trimmed: dict[str, tuple[str, float]]) -> dict[str, object]:
            return {
                "id": f"{_SESSION}/{index:04d}",
                "audio_path": str(logs / _SESSION / "audio" / f"{index:04d}.wav"),
                "audio_seconds": 3.5,
                "logged_text": raw,
                "raw": raw,
                "trimmed": {cap: {"text": text, "removed_seconds": cut} for cap, (text, cut) in trimmed.items()},
            }

        (out / f"replay-{DEFAULT_MODEL}.jsonl").write_text(
            json.dumps(record(1, _SPLIT, {"300": (_JOINED, 1.2), "500": (_JOINED, 1.0), "800": (_SPLIT, 0.7)}))
            + "\n"
            + json.dumps(record(2, _JOINED, {"300": (_JOINED, 0.0), "500": (_JOINED, 0.0), "800": (_JOINED, 0.0)}))
            + "\n",
            encoding="utf-8",
        )
        again = _PauseHearingEngine()

        summary = _run_replay(tmp_path, again, 300, 500, 800, extra=("--control",))

        assert len(again.heard) == 2 * 2
        assert list(summary["variants"]) == ["raw", "300 ms", "500 ms", "800 ms", "repeat", "shift"]
        assert summary["variants"]["500 ms"]["clips_fewer_full_stops"] == 1

    def test_the_summary_counts_and_but_so_or_after_the_first_sentence_and_the_marks_before_them(
        self, tmp_path: Path,
    ) -> None:
        _replay_logs(tmp_path)

        summary = _run_replay(tmp_path, _PauseHearingEngine(), extra=("--control",))

        picked = ("inner_starts", "stop_before", "comma_before")
        variants = summary["variants"]
        assert [variants["raw"][key] for key in picked] == [1, 1, 1]
        assert [variants["300 ms"][key] for key in picked] == [0, 0, 2]
        assert [variants["800 ms"][key] for key in picked] == [1, 1, 1]
        assert [variants["shift"][key] for key in picked] == [1, 1, 1]


_LENGTHS: tuple[float | None, ...] = (0.5, 14.99, 15.0, 29.99, 30.0, 45.0, None)


def _clips_logged_at(tmp_path: Path, lengths: tuple[float | None, ...]) -> None:
    """One short clip per entry, each logged as lasting that long; the replay goes by the log."""
    rng = np.random.default_rng(30)
    session = _write_session(tmp_path / "logs", _SESSION, [
        _transcription(index, "The kettle hums.", seconds=seconds) for index, seconds in enumerate(lengths, start=1)
    ])
    for index in range(1, len(lengths) + 1):
        _write_wav(session / "audio" / f"{index:04d}.wav", _speech(0.3, rng))


def _replayed_lengths(tmp_path: Path) -> set[float | None]:
    entries = _jsonl(tmp_path / "eval" / f"replay-{DEFAULT_MODEL}-per-dictation.jsonl")
    return {_LENGTHS[int(entry["id"].rsplit("/", 1)[1]) - 1] for entry in entries}


class TestReplayLengthRange:
    @pytest.mark.parametrize(
        ("bounds", "kept"),
        [
            (("--min-seconds", "15", "--max-seconds", "30"), {15.0, 29.99}),
            (("--min-seconds", "15"), {15.0, 29.99, 30.0, 45.0}),
            (("--max-seconds", "15"), {0.5, 14.99}),
            ((), set(_LENGTHS)),
        ],
    )
    def test_only_clips_logged_within_the_length_range_are_replayed(
        self, tmp_path: Path, bounds: tuple[str, ...], kept: set[float | None],
    ) -> None:
        _clips_logged_at(tmp_path, _LENGTHS)

        _run_replay(tmp_path, _PauseHearingEngine(), extra=bounds)

        assert _replayed_lengths(tmp_path) == kept

    @pytest.mark.parametrize("seed", range(5))
    def test_the_length_range_is_applied_before_the_limit(self, tmp_path: Path, seed: int) -> None:
        _clips_logged_at(tmp_path, _LENGTHS)

        _run_replay(tmp_path, _PauseHearingEngine(), limit=1, extra=("--min-seconds", "30", "--seed", str(seed)))

        assert len(_replayed_lengths(tmp_path)) == 1
        assert _replayed_lengths(tmp_path) <= {30.0, 45.0}

    @pytest.mark.parametrize(
        "bounds",
        [("--min-seconds", "30", "--max-seconds", "15"), ("--min-seconds", "15", "--max-seconds", "15"), ("--min-seconds", "-1")],
    )
    def test_a_length_range_that_is_empty_or_negative_is_refused(self, tmp_path: Path, bounds: tuple[str, ...]) -> None:
        _clips_logged_at(tmp_path, _LENGTHS)

        with pytest.raises(SystemExit):
            _run_replay(tmp_path, _PauseHearingEngine(), extra=bounds)

    def test_progress_is_printed_every_25_clips_with_the_wall_time_at_the_end_and_no_dictated_text(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _clips_logged_at(tmp_path, (20.0,) * 60)

        _run_replay(tmp_path, _PauseHearingEngine())

        shown = capsys.readouterr().out
        progress = re.findall(r"^replayed (\d+) of (\d+) clips, \d+ s$", shown, flags=re.MULTILINE)
        assert progress == [("25", "60"), ("50", "60")]
        assert re.search(r"^Took \d+ s for 60 clips decoded this run, [\d.]+ s a clip", shown, flags=re.MULTILINE)
        assert "kettle" not in shown


class TestSamplingClips:
    @staticmethod
    def _dictations(count: int) -> list[_Dictation]:
        return [
            _Dictation(f"s/{index:04d}", DEFAULT_MODEL, _ENGINE, Path(f"{index}.wav"), 1.0, "", "")
            for index in range(count)
        ]

    def test_the_same_seed_picks_the_same_clips_and_another_seed_picks_others(self) -> None:
        dictations = self._dictations(1000)

        first, again, other = (sample_clips(dictations, 20, seed) for seed in (0, 0, 1))

        assert first == again
        assert first != other
        assert len(set(first)) == 20

    @pytest.mark.parametrize(("limit", "picked"), [(None, 1000), (0, 0), (5000, 1000)])
    def test_a_limit_never_picks_a_clip_twice_or_more_than_there_are(self, limit: int | None, picked: int) -> None:
        sample = sample_clips(self._dictations(1000), limit, 0)

        assert len(sample) == len(set(sample)) == picked


class TestPunctuationSpots:
    def test_a_mark_that_moved_is_found_across_a_changed_word(self) -> None:
        spots = punctuation_spots("Bring the tent. And the stove", "Bring a tent, and the stove")

        assert [(spot["raw_mark"], spot["trimmed_mark"]) for spot in spots] == [(".", ",")]
        assert spots[0]["raw"] == "Bring the tent. And the stove"

    def test_a_change_in_case_alone_is_not_a_spot(self) -> None:
        assert punctuation_spots("Light the lantern.", "light the Lantern.") == []

    def test_marks_in_other_scripts_are_compared(self) -> None:
        spots = punctuation_spots("هل وصل الطرد؟ نعم", "هل وصل الطرد. نعم")

        assert [(spot["raw_mark"], spot["trimmed_mark"]) for spot in spots] == [("؟", ".")]


# --- Judging -------------------------------------------------------------------

_JUDGE_RATE = 100


def _clock(seconds: float) -> np.ndarray:
    """Stand-in audio in which every sample holds its own time, so a slice of it knows where it came from."""
    return (np.arange(round(seconds * _JUDGE_RATE)) / _JUDGE_RATE).astype(np.float32)


class _TimedSpeech:
    """Hears each word that was finished inside the stretch of audio it is given."""

    def __init__(self, words: list[str], ends: list[float], mishears_every: int = 0) -> None:
        self._words, self._ends, self._mishears_every = words, ends, mishears_every
        self.decodes = 0
        self.longest = 0.0

    def decode(self, samples: np.ndarray, sample_rate: int) -> str:
        self.decodes += 1
        self.longest = max(self.longest, samples.size / sample_rate)
        first, last = float(samples[0]), float(samples[-1]) + 1 / sample_rate
        heard = [word for word, end in zip(self._words, self._ends) if first < end <= last + 1e-4]
        if self._mishears_every:
            heard = [word if index % self._mishears_every else "hmm" for index, word in enumerate(heard, start=1)]
        return " ".join(heard)


def _rambling_dictation(count: int, seed: int) -> tuple[list[str], list[float]]:
    """Words said at an uneven pace, with a long think every forty or so, and the same phrases over and over."""
    rng = np.random.default_rng(seed)
    stock = ["lantern", "orchard", "kettle", "the", "and", "then", "naïve", "الطرد", "walk", "to", "it's", "re-check"]
    words = [stock[int(rng.integers(len(stock)))] for _ in range(count)]
    gaps = rng.uniform(0.2, 0.5, count)
    gaps[rng.integers(count, size=count // 40)] += rng.uniform(2.0, 9.0, count // 40)
    return words, np.cumsum(gaps).tolist()


class TestFindingAPlaceInItsClip:
    @pytest.mark.parametrize("at", [0, 1, 37, 412, 1000, 1998, 1999])
    def test_a_word_is_found_just_after_it_was_said_in_a_dictation_of_two_thousand_words(self, at: int) -> None:
        words, ends = _rambling_dictation(2000, seed=31)
        engine = _TimedSpeech(words, ends)

        found = locate(words, at, _clock(ends[-1] + 1.0), _JUDGE_RATE, engine.decode)

        assert found is not None
        assert -0.011 <= found - ends[at] <= 0.4
        assert engine.decodes <= 16
        # Never the whole lead-up: decoding minutes of audio per try is what made long dictations slow.
        assert engine.longest <= 30.0

    def test_a_word_before_a_long_pause_is_placed_at_the_start_of_the_pause_not_inside_it(self) -> None:
        words = ["pack", "the", "kettle", "and", "walk"]
        ends = [0.4, 0.7, 1.2, 9.5, 9.9]

        found = locate(words, 2, _clock(11.0), _JUDGE_RATE, _TimedSpeech(words, ends).decode)

        assert found is not None
        assert 1.2 <= found <= 1.6

    def test_a_place_is_still_found_when_the_engine_mishears_one_word_in_seven(self) -> None:
        words, ends = _rambling_dictation(600, seed=32)
        engine = _TimedSpeech(words, ends, mishears_every=7)
        samples = _clock(ends[-1] + 1.0)

        found = [locate(words, at, samples, _JUDGE_RATE, engine.decode) for at in (50, 201, 333, 590)]

        # A misheard word is only known to have gone by once the next one is heard.
        assert all(spot is not None and -0.011 <= spot - ends[at] <= 1.0 for spot, at in zip(found, (50, 201, 333, 590)))

    def test_a_word_the_engine_never_hears_is_reported_as_not_found(self) -> None:
        words = ["pack", "the", "kettle"]
        silent = _TimedSpeech(["hmm", "hmm"], [0.5, 1.0])

        assert locate(words, 1, _clock(3.0), _JUDGE_RATE, silent.decode) is None

    def test_a_stray_match_far_past_the_words_really_heard_does_not_count_as_hearing_them(self) -> None:
        # The stretch ends on a cut-off word the engine wrote as "the", which is also said much later.
        words = ["pack", "the", "kettle", "and", "walk", "down", "to", "orchard", "gate", "before", "dusk", "on", "the", "mill"]
        engine = _TimedSpeech(["pack", "the", "kettle", "and", "the"], [0.4, 0.7, 1.2, 1.5, 1.8])

        found = locate(words, 6, _clock(8.0), _JUDGE_RATE, engine.decode)

        assert found is None


_ERRAND = ("pack", "the", "lantern", "and", "the", "kettle", "then", "walk", "down", "to", "the", "orchard", "gate")


class TestCuttingAClipRoundAPlace:
    @pytest.mark.parametrize("at", [0, 3, 37, 412, 1000, 1996, 1999])
    def test_the_clip_holds_the_place_with_the_words_before_and_after_it(self, at: int) -> None:
        words, ends = _rambling_dictation(2000, seed=33)
        engine = _TimedSpeech(words, ends)

        cut = place_in_clip(words, at, _clock(ends[-1] + 1.0), _JUDGE_RATE, engine.decode)

        assert cut is not None
        begin, finish = cut[0] / _JUDGE_RATE, cut[1] / _JUDGE_RATE
        assert begin <= (ends[at - 4] if at >= 5 else 0.0)
        assert finish >= ends[min(at + 3, 1999)]
        assert finish - begin <= 20.0

    def test_a_long_pause_right_after_the_place_is_played_through_to_the_words_that_follow(self) -> None:
        words = list(_ERRAND)
        ends = [0.4, 0.7, 1.2, 1.5, 1.8, 2.3, 9.6, 10.0, 10.4, 10.7, 11.0, 11.5, 12.0]

        cut = place_in_clip(words, 5, _clock(13.0), _JUDGE_RATE, _TimedSpeech(words, ends).decode)

        assert cut is not None
        assert cut[0] / _JUDGE_RATE <= 0.4
        assert cut[1] / _JUDGE_RATE >= 10.4

    def test_a_place_the_engine_cannot_hear_in_the_clip_cut_for_it_is_dropped(self) -> None:
        words = list(_ERRAND)
        ends = [0.4, 0.7, 1.2, 1.5, 1.8, 2.3, 2.7, 3.0, 3.4, 3.7, 4.0, 4.5, 5.0]
        # In a long stretch the engine gets every word; in a short one it loses "kettle".
        whole = _TimedSpeech(words, ends)

        def decode(samples: np.ndarray, sample_rate: int) -> str:
            heard = whole.decode(samples, sample_rate)
            return heard if samples.size / sample_rate > 12 else heard.replace("kettle", "cattle")

        assert place_in_clip(words, 5, _clock(40.0), _JUDGE_RATE, whole.decode) is not None
        assert place_in_clip(words, 5, _clock(40.0), _JUDGE_RATE, decode) is None

    def test_a_think_of_over_twenty_seconds_at_the_place_is_not_cut_into_a_clip(self) -> None:
        words = list(_ERRAND)
        ends = [0.4, 0.7, 1.2, 1.5, 1.8, 2.3, 31.6, 32.0, 32.4, 32.7, 33.0, 33.5, 34.0]

        assert place_in_clip(words, 5, _clock(35.0), _JUDGE_RATE, _TimedSpeech(words, ends).decode) is None


_RAMBLE = (
    "Pack the lantern and the kettle. And walk down to the orchard before the gate is shut. "
    "Then light the stove, pour the tea. So we can sit by the naïve old mill and wait for الطرد to come."
)
_RAMBLE_JOINED = (
    "Pack the lantern and the kettle, and walk down to the orchard before the gate is shut. "
    "Then light the stove. Pour the tea so we can sit by the naïve old mill and wait for الطرد to come."
)
_WORDS_A_SECOND = 2


class _SteadyReader:
    """Hears the ramble at two words a second in whatever stretch of a saved clip it is given, marks and all."""

    def __init__(self, tmp_path: Path) -> None:
        self.built_for: list[tuple[str, str]] = []
        self.closed = 0
        self._clips = [load_wav(path).data for path in sorted((tmp_path / "logs" / _SESSION / "audio").glob("*.wav"))]

    def build(self, model: str, engine: str) -> tuple:
        self.built_for.append((model, engine))
        return self.decode, self.close

    def decode(self, samples: np.ndarray, sample_rate: int) -> str:
        first = next(at for at in (_position_of(clip, samples) for clip in self._clips) if at is not None)
        heard = range(-(-first * _WORDS_A_SECOND // sample_rate), (first + samples.size) * _WORDS_A_SECOND // sample_rate)
        return " ".join(_RAMBLE.split()[index] for index in heard)

    def close(self) -> None:
        self.closed += 1


def _replayed(tmp_path: Path, dictations: int) -> Path:
    """An eval folder as a finished replay leaves it: *dictations* clips of the same ramble."""
    rng = np.random.default_rng(40)
    out = tmp_path / "eval"
    out.mkdir()
    seconds = len(_RAMBLE.split()) / _WORDS_A_SECOND
    records = []
    for index in range(1, dictations + 1):
        clip = tmp_path / "logs" / _SESSION / "audio" / f"{index:04d}.wav"
        _write_wav(clip, _speech(seconds, rng))
        records.append({
            "id": f"{_SESSION}/{index:04d}",
            "audio_path": str(clip),
            "audio_seconds": seconds,
            "logged_text": _RAMBLE,
            "raw": _RAMBLE,
            "trimmed": {
                "500": {"text": _RAMBLE_JOINED, "removed_seconds": 1.2},
                "800": {"text": _RAMBLE, "removed_seconds": 0.4},
            },
        })
    (out / f"replay-{DEFAULT_MODEL}.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8",
    )
    (out / f"replay-{DEFAULT_MODEL}-summary.json").write_text(json.dumps({"engine": _ENGINE}), encoding="utf-8")
    return out


def _run_judge(tmp_path: Path, reader: _SteadyReader, *extra: str) -> dict:
    out = tmp_path / "eval"
    main(["judge", "--logs", str(tmp_path / "logs"), "--out", str(out), *extra], build_decoder=reader.build)
    script = (out / "judge" / "items.js").read_text(encoding="utf-8")
    assert script.startswith("window.JUDGE = ") and script.endswith(";\n")
    return json.loads(script[len("window.JUDGE = "):-2])


class TestJudging:
    def test_at_most_two_places_are_taken_from_one_dictation_and_the_same_seed_takes_the_same_ones(
        self, tmp_path: Path,
    ) -> None:
        records = _jsonl(_replayed(tmp_path, 40) / f"replay-{DEFAULT_MODEL}.jsonl")

        places = judged_places(records, "500", seed=3)

        # The ramble differs from its joined version at three places, in each of forty dictations.
        assert len(places) == 80
        assert max(Counter(record["id"] for record, _ in places).values()) == 2
        assert len({spot["word"] for _, spot in places}) == 3
        assert [(r["id"], s["word"]) for r, s in places] == [(r["id"], s["word"]) for r, s in judged_places(records, "500", 3)]
        assert [(r["id"], s["word"]) for r, s in places] != [(r["id"], s["word"]) for r, s in judged_places(records, "500", 4)]

    def test_a_variant_that_matches_raw_or_was_never_replayed_has_no_places(self, tmp_path: Path) -> None:
        records = _jsonl(_replayed(tmp_path, 3) / f"replay-{DEFAULT_MODEL}.jsonl")

        assert judged_places(records, "800", seed=0) == []
        assert judged_places(records, "300", seed=0) == []

    def test_the_page_gets_one_short_clip_and_both_versions_for_each_place(self, tmp_path: Path) -> None:
        _replayed(tmp_path, 30)
        reader = _SteadyReader(tmp_path)

        page = _run_judge(tmp_path, reader, "--count", "40")

        folder = tmp_path / "eval" / "judge"
        assert (folder / "index.html").read_text(encoding="utf-8") == (
            Path(__file__).parents[1] / "scripts" / "punctuation_judge.html"
        ).read_text(encoding="utf-8")
        assert page["variant"] == "pauses shortened to 500 ms"
        assert len(page["items"]) == 40
        assert sorted(path.name for path in (folder / "clips").iterdir()) == [f"{n:03d}.wav" for n in range(1, 41)]
        assert (reader.built_for, reader.closed) == ([(DEFAULT_MODEL, _ENGINE)], 1)
        for item in page["items"]:
            with wave.open(str(folder / item["clip"])) as clip:
                assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, _RATE)
                assert 3.0 < clip.getnframes() / _RATE <= 7.0
                loudest = np.abs(np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2")).max()
            assert 0.85 * 32767 < loudest <= 0.9 * 32767 + 1

    def test_each_version_is_shown_with_the_differing_mark_inside_the_highlighted_part(self, tmp_path: Path) -> None:
        _replayed(tmp_path, 30)

        page = _run_judge(tmp_path, _SteadyReader(tmp_path), "--count", "40")

        for item in page["items"]:
            raw, other = item[item["raw"]], item["b" if item["raw"] == "a" else "a"]
            assert " ".join(part for part in (raw["before"], raw["spot"] + raw["after"]) if part) in _RAMBLE
            assert " ".join(part for part in (other["before"], other["spot"] + other["after"]) if part) in _RAMBLE_JOINED
            assert raw["spot"].casefold().replace(item["raw_mark"], "") == other["spot"].casefold().replace(
                item["trimmed_mark"], "",
            )
            assert item["raw_mark"] != item["trimmed_mark"]
            assert (item["raw_mark"] or " ") in raw["spot"] and (item["trimmed_mark"] or " ") in other["spot"]

    def test_which_side_holds_the_version_as_recorded_is_mixed_and_repeats_with_the_seed(self, tmp_path: Path) -> None:
        _replayed(tmp_path, 30)

        sides = [item["raw"] for item in _run_judge(tmp_path, _SteadyReader(tmp_path), "--count", "40")["items"]]
        again = [item["raw"] for item in _run_judge(tmp_path, _SteadyReader(tmp_path), "--count", "40")["items"]]

        assert 10 <= sides.count("a") <= 30
        assert sides == again

    def test_the_console_shows_counts_and_never_a_dictated_word(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _replayed(tmp_path, 5)

        _run_judge(tmp_path, _SteadyReader(tmp_path), "--count", "6")

        printed = capsys.readouterr().out
        assert "6 places to judge, 0 skipped" in printed
        assert not any(word in printed for word in ("lantern", "kettle", "orchard", "الطرد"))

    def test_judging_before_any_replay_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "logs").mkdir()

        with pytest.raises(SystemExit, match="Run replay first"):
            main(["judge", "--logs", str(tmp_path / "logs"), "--out", str(tmp_path / "eval")])

    def test_a_variant_with_nothing_to_judge_is_refused_without_loading_the_engine(self, tmp_path: Path) -> None:
        _replayed(tmp_path, 3)
        reader = _SteadyReader(tmp_path)

        with pytest.raises(SystemExit, match="no place where 800 ms differs"):
            _run_judge(tmp_path, reader, "--variant", "800")
        assert reader.built_for == []
