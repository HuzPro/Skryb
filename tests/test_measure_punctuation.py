"""The punctuation measurement reads logs nobody can reproduce and audio nobody may share.

So every log, clip and sentence here is invented: the logs carry the faults
real ones do (half-written lines, duplicates, blank fields, other scripts),
the clips are synthetic speech and room noise at the levels the session logs
record, and the engine is a stand-in that splits a sentence at a long pause,
which is the habit being measured.
"""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.measure_punctuation import (
    DEFAULT_MODEL,
    _Dictation,
    main,
    punctuation_spots,
    sample_clips,
    sentences,
    shorten_pauses,
)

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


_SPLIT = "Light the lantern and. Then walk to the orchard."
_JOINED = "Light the lantern and then walk to the orchard."


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
        self.built_for: list[tuple[str, str]] = []
        self.closed = 0
        self._drops_a_word = drops_a_word_when_joined

    def build(self, model: str, engine: str) -> tuple:
        self.built_for.append((model, engine))
        return self.decode, self.close

    def decode(self, samples: np.ndarray, sample_rate: int) -> str:
        self.heard.append((samples.dtype, sample_rate, samples.size))
        if _longest_inner_pause(samples, sample_rate) > 0.5:
            return _SPLIT
        return _JOINED.replace("the orchard", "orchard") if self._drops_a_word else _JOINED

    def close(self) -> None:
        self.closed += 1


def _replay_logs(tmp_path: Path) -> Path:
    """Two clips to replay, plus one of each kind the replay must step over."""
    rng = np.random.default_rng(20)
    logs = tmp_path / "logs"
    session = _write_session(logs, "20261120T100000Z-0a1b2c3d", [
        _transcription(1, _SPLIT, seconds=3.5),
        _transcription(2, "Light the lantern and then walk to the orchards.", seconds=2.2),
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


def _run_replay(tmp_path: Path, engine: _PauseHearingEngine, *caps: int, limit: int | None = None) -> dict:
    out = tmp_path / "eval"
    arguments = ["replay", "--logs", str(tmp_path / "logs"), "--out", str(out)]
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
        assert (long["clips_fewer_full_stops"], long["clips_shortened"]) == (0, 1)
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
        assert difference["cap_ms"] == 300
        assert Path(difference["audio_path"]) == logs / "20261120T100000Z-0a1b2c3d" / "audio" / "0001.wav"
        assert (difference["raw_text"], difference["trimmed_text"]) == (_SPLIT, _JOINED)
        assert (difference["spot"]["raw_mark"], difference["spot"]["trimmed_mark"]) == (".", "")

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

    def test_asking_for_a_new_cap_redoes_the_clips_that_lack_it(self, tmp_path: Path) -> None:
        _replay_logs(tmp_path)
        _run_replay(tmp_path, _PauseHearingEngine(), 300)
        again = _PauseHearingEngine()

        summary = _run_replay(tmp_path, again, 300, 500)

        assert len(again.heard) == 1 + 3
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
