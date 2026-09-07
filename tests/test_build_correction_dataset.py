"""The corpus builder joins two logs that share no identifier.

Every join here is a timing guess, so the tests pin the guess against a
synthetic pair of logs rather than trusting a run against real dictation
nobody can reproduce.
"""

import json
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.build_correction_dataset import _Options, _edit_kind, build

pytestmark = pytest.mark.integration

_TOLERANCE = timedelta(seconds=30)
_SAMPLE_RATE = 16000
_FRAMES = 800
_START = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)


def _at(second: int) -> str:
    moment = _START + timedelta(seconds=second)
    return moment.isoformat().replace("+00:00", "Z")


def _transcription(index: int, second: int, text: str) -> dict[str, object]:
    return {
        "event": "transcription",
        "index": index,
        "requested_at": _at(second - 1),
        "finished_at": _at(second),
        "wall_seconds": 1.0,
        "engine_seconds": 0.8,
        "model": "crisperwhisper-2.0-turbo",
        "engine": "crisperwhisper",
        "device": "cuda",
        "audio_seconds": 1.5,
        "sample_rate": _SAMPLE_RATE,
        "audio_rms": 0.12,
        "audio_peak": 0.44,
        "audio_noise_floor": 0.01,
        "language_probability": 0.99,
        "audio_path": f"audio/{index:04d}.wav",
        "text": text,
        "error": None,
    }


def _delivered(
    seq: int, second: int, text: str, pipeline_mode: str = "batch",
) -> dict[str, object]:
    return {
        "event": "delivered",
        "seq": seq,
        "at": _at(second),
        "text": text,
        "pipeline_mode": pipeline_mode,
        "output_mode": "paste",
    }


def _span(seq: int, pasted: str, final: str) -> dict[str, object]:
    return {"seq": seq, "pasted_text": pasted, "final_text": final}


def _sent(
    block: int,
    spans: list[dict[str, object]],
    reason: str = "enter",
    reliable: bool = True,
    flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "event": "sent",
        "block": block,
        "at": _at(30),
        "reason": reason,
        "process": "notepad.exe",
        "title": "Untitled - Notepad",
        "text": "".join(str(span["final_text"]) for span in spans),
        "reliable": reliable,
        "flags": list(flags or []),
        "key_count": 4,
        "spans": spans,
    }


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(_SAMPLE_RATE)
        output.writeframes(b"\x00\x00" * _FRAMES)


def _session(
    logs_dir: Path,
    name: str,
    transcriptions: list[dict[str, object]],
    edits: list[dict[str, object]],
) -> None:
    session = logs_dir / name
    _write_jsonl(session / "session.jsonl", transcriptions)
    if edits:
        _write_jsonl(session / "edits.jsonl", edits)
    for transcription in transcriptions:
        _write_wav(session / str(transcription["audio_path"]))


def _build(tmp_path: Path) -> Path:
    output_dir = tmp_path / "corpus"
    build(_Options(tmp_path / "logs", output_dir, _TOLERANCE))
    return output_dir


def _records(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class TestJoiningDeliveriesToTranscriptions:
    def test_should_build_one_row_from_a_clean_single_paste_block(
        self,
        tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "20260906T120000Z-abcd1234",
            [_transcription(1, 10, "hello world")],
            [
                _delivered(1, 9, "Hello world."),
                _sent(1, [_span(1, "Hello world.", "Hello world.")]),
            ],
        )

        output_dir = _build(tmp_path)

        rows = _records(output_dir / "dataset.jsonl")
        assert len(rows) == 1
        assert rows[0]["raw_text"] == "hello world"
        assert rows[0]["pasted_text"] == "Hello world."
        assert rows[0]["final_text"] == "Hello world."
        assert rows[0]["wer_strict"] == 0.0
        assert rows[0]["edit_kind"] == "none"
        assert rows[0]["paste_count"] == 1
        assert rows[0]["process"] == "notepad.exe"
        assert rows[0]["end_reason"] == "enter"

    def test_should_attribute_each_paste_in_a_block_to_its_own_transcription(
        self,
        tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "two-pastes",
            [_transcription(1, 10, "first"), _transcription(2, 20, "second")],
            [
                _delivered(1, 9, "First."),
                _delivered(2, 19, "Second."),
                _sent(
                    1,
                    [
                        _span(1, "First.", "First."),
                        _span(2, "Second.", "Second!"),
                    ],
                ),
            ],
        )

        output_dir = _build(tmp_path)

        rows = _records(output_dir / "dataset.jsonl")
        assert [row["transcription_index"] for row in rows] == [1, 2]
        assert [row["raw_text"] for row in rows] == ["first", "second"]
        assert [row["final_text"] for row in rows] == ["First.", "Second!"]
        assert [row["paste_count"] for row in rows] == [2, 2]
        assert [row["block"] for row in rows] == [1, 1]

    def test_should_drop_a_delivery_whose_transcription_is_outside_tolerance(
        self,
        tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "late-transcription",
            [_transcription(1, 300, "far too late")],
            [
                _delivered(1, 9, "Orphan."),
                _sent(1, [_span(1, "Orphan.", "Orphan.")]),
            ],
        )

        output_dir = _build(tmp_path)

        assert _records(output_dir / "dataset.jsonl") == []

    def test_should_skip_a_session_without_edits_and_keep_going(
        self,
        tmp_path: Path,
    ) -> None:
        logs_dir = tmp_path / "logs"
        _session(logs_dir, "a-no-edit-capture", [_transcription(1, 10, "quiet")], [])
        _session(
            logs_dir,
            "b-with-edit-capture",
            [_transcription(1, 10, "hello world")],
            [
                _delivered(1, 9, "Hello world."),
                _sent(1, [_span(1, "Hello world.", "Hello world.")]),
            ],
        )

        output_dir = _build(tmp_path)

        rows = _records(output_dir / "dataset.jsonl")
        assert [row["session"] for row in rows] == ["b-with-edit-capture"]


class TestCorpusAudio:
    def test_should_exclude_missing_audio_from_the_manifest(self, tmp_path: Path) -> None:
        session_name = "missing-audio"
        _session(
            tmp_path / "logs", session_name,
            [_transcription(1, 10, "hello world")],
            [_delivered(1, 11, "hello world"),
             _sent(1, [_span(1, "hello world", "hello world")])],
        )
        (tmp_path / "logs" / session_name / "audio" / "0001.wav").unlink()

        output_dir = _build(tmp_path)

        assert len(_records(output_dir / "dataset.jsonl")) == 1
        assert _records(output_dir / "manifest.jsonl") == []

    def test_should_copy_the_wav_into_the_corpus(self, tmp_path: Path) -> None:
        _session(
            tmp_path / "logs",
            "with-audio",
            [_transcription(3, 10, "hello world")],
            [
                _delivered(1, 9, "Hello world."),
                _sent(1, [_span(1, "Hello world.", "Hello world.")]),
            ],
        )

        output_dir = _build(tmp_path)

        rows = _records(output_dir / "dataset.jsonl")
        assert rows[0]["audio_path"] == "audio/with-audio/0003.wav"
        copied = output_dir / "audio" / "with-audio" / "0003.wav"
        assert copied.is_file()
        with wave.open(str(copied), "rb") as audio:
            assert audio.getnframes() == _FRAMES


class TestTrainingManifest:
    def test_should_keep_only_reliable_rows_the_dictionary_left_alone(
        self,
        tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "mixed",
            [
                _transcription(1, 10, "the quick brown fox"),
                _transcription(2, 20, "the quick brown fox"),
                _transcription(3, 30, "the quick brown ox"),
            ],
            [
                _delivered(1, 9, "the quick brown fox"),
                _delivered(2, 19, "the quick brown fox"),
                _delivered(3, 29, "the quick brown fox"),
                _sent(1, [_span(1, "the quick brown fox", "the quick brown box")]),
                _sent(
                    2,
                    [_span(2, "the quick brown fox", "the quick brown box")],
                    reliable=False,
                ),
                _sent(3, [_span(3, "the quick brown fox", "the quick brown box")]),
            ],
        )

        output_dir = _build(tmp_path)

        rows = _records(output_dir / "dataset.jsonl")
        assert [row["edit_kind"] for row in rows] == ["lexical"] * 3
        assert [row["pipeline_delta"] for row in rows] == [0.0, 0.0, 0.25]
        assert _records(output_dir / "manifest.jsonl") == [
            {
                "audio_filepath": "audio/mixed/0001.wav",
                "duration": 1.5,
                "text": "the quick brown box",
            }
        ]


class TestEditKind:
    def test_should_report_none_for_an_untouched_paste(self) -> None:
        assert _edit_kind("Hello world.", "Hello world.") == "none"

    def test_should_report_case_only_for_a_capitalisation_fix(self) -> None:
        assert _edit_kind("hello world", "Hello world") == "case_only"

    def test_should_report_lexical_for_a_substituted_word(self) -> None:
        assert _edit_kind("the quick brown fox", "the quick brown box") == "lexical"


class TestAcousticManifestScope:
    """The manifest is the acoustic training target, so its filter matters."""

    def test_should_keep_a_dictation_the_user_did_not_touch(
        self, tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "clean",
            [_transcription(1, 10, "the quick brown fox")],
            [
                _delivered(1, 9, "the quick brown fox"),
                _sent(1, [_span(1, "the quick brown fox", "the quick brown fox")]),
            ],
        )

        output_dir = _build(tmp_path)

        rows = _records(output_dir / "dataset.jsonl")
        assert rows[0]["edit_kind"] == "none"
        assert [entry["text"] for entry in _records(output_dir / "manifest.jsonl")] == [
            "the quick brown fox"
        ]

    def test_should_leave_streaming_fragments_out_of_the_manifest(
        self, tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "live",
            [_transcription(1, 10, "the quick brown fox")],
            [
                _delivered(1, 9, "the quick brown fox", pipeline_mode="streaming"),
                _sent(1, [_span(1, "the quick brown fox", "the quick brown fox")]),
            ],
        )

        output_dir = _build(tmp_path)

        assert len(_records(output_dir / "dataset.jsonl")) == 1
        assert _records(output_dir / "manifest.jsonl") == []


class TestDeliveryOrdering:
    """A batch delivery is reported after its transcription event, not before."""

    def test_should_match_a_delivery_reported_after_the_transcription(
        self, tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "afterwards",
            [_transcription(1, 10, "the quick brown fox")],
            [
                _delivered(1, 11, "the quick brown fox"),
                _sent(1, [_span(1, "the quick brown fox", "the quick brown fox")]),
            ],
        )

        output_dir = _build(tmp_path)

        rows = _records(output_dir / "dataset.jsonl")
        assert [row["transcription_index"] for row in rows] == [1]

    def test_should_match_a_delivery_reported_before_the_transcription(
        self, tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "beforehand",
            [_transcription(1, 10, "the quick brown fox")],
            [
                _delivered(1, 9, "the quick brown fox"),
                _sent(1, [_span(1, "the quick brown fox", "the quick brown fox")]),
            ],
        )

        output_dir = _build(tmp_path)

        rows = _records(output_dir / "dataset.jsonl")
        assert [row["transcription_index"] for row in rows] == [1]

    def test_should_pick_the_nearer_of_two_transcriptions(
        self, tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "twice",
            [
                _transcription(1, 10, "first clip"),
                _transcription(2, 60, "second clip"),
            ],
            [
                _delivered(1, 11, "first clip"),
                _delivered(2, 61, "second clip"),
                _sent(1, [_span(1, "first clip", "first clip")]),
                _sent(2, [_span(2, "second clip", "second clip")]),
            ],
        )

        output_dir = _build(tmp_path)

        rows = _records(output_dir / "dataset.jsonl")
        assert [row["transcription_index"] for row in rows] == [1, 2]


class TestManifestExcludesFlaggedRows:
    def test_should_exclude_a_row_whose_caret_left_the_pasted_text(
        self, tmp_path: Path,
    ) -> None:
        _session(
            tmp_path / "logs",
            "flagged",
            [_transcription(1, 10, "the quick brown fox")],
            [
                _delivered(1, 11, "the quick brown fox"),
                _sent(
                    1,
                    [_span(1, "the quick brown fox", "the quick brown fox")],
                    flags=["caret_before_span"],
                ),
            ],
        )

        output_dir = _build(tmp_path)

        assert len(_records(output_dir / "dataset.jsonl")) == 1
        assert _records(output_dir / "manifest.jsonl") == []
