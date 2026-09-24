import json
import math
from pathlib import Path

import pytest

from voicetotext.diagnostics.timing_estimate import (
    TranscriptionTimeEstimate,
    engine_key,
)

_GPU = engine_key("crisperwhisper-2.0-turbo", "crisperwhisper", "cuda")
_CPU = engine_key("crisperwhisper-2.0-turbo", "crisperwhisper", "cpu")


def _write_session(folder: Path, events: list[object], extra_lines: str = "") -> None:
    folder.mkdir(parents=True)
    lines = [json.dumps(event) for event in events]
    (folder / "session.jsonl").write_text(
        "\n".join(lines) + "\n" + extra_lines, encoding="utf-8",
    )


def _transcription(audio: float, wall: float, device: str = "cuda", **extra) -> dict:
    return {
        "event": "transcription",
        "model": "crisperwhisper-2.0-turbo",
        "engine": "crisperwhisper",
        "device": device,
        "audio_seconds": audio,
        "wall_seconds": wall,
        "error": None,
        **extra,
    }


class TestEstimate:
    def test_has_no_estimate_until_it_has_seen_a_few_transcriptions(self):
        estimate = TranscriptionTimeEstimate()
        estimate.record(_GPU, 2.0, 0.4)
        estimate.record(_GPU, 4.0, 0.5)

        assert estimate.estimate(_GPU, 3.0) is None

    def test_follows_a_fixed_start_up_cost_plus_time_per_second_of_audio(self):
        estimate = TranscriptionTimeEstimate()
        for audio in (1.0, 2.0, 5.0, 10.0, 30.0):
            estimate.record(_GPU, audio, 0.3 + 0.02 * audio)

        assert estimate.estimate(_GPU, 20.0) == pytest.approx(0.7)

    def test_keeps_each_device_separate(self):
        estimate = TranscriptionTimeEstimate()
        for audio in (1.0, 5.0, 10.0):
            estimate.record(_GPU, audio, 0.4)
            estimate.record(_CPU, audio, 4.0)

        assert estimate.estimate(_GPU, 5.0) == pytest.approx(0.4)
        assert estimate.estimate(_CPU, 5.0) == pytest.approx(4.0)

    def test_uses_the_typical_time_when_every_clip_was_the_same_length(self):
        estimate = TranscriptionTimeEstimate()
        for wall in (0.4, 0.4, 0.5, 0.4, 9.0):
            estimate.record(_GPU, 3.0, wall)

        assert estimate.estimate(_GPU, 3.0) == pytest.approx(0.4)

    @pytest.mark.parametrize(
        ("audio", "wall"),
        [(0.0, 0.4), (-1.0, 0.4), (2.0, 0.0), (math.nan, 0.4), (2.0, math.inf)],
    )
    def test_ignores_timings_that_cannot_be_real(self, audio, wall):
        estimate = TranscriptionTimeEstimate()
        for _ in range(5):
            estimate.record(_GPU, audio, wall)

        assert estimate.estimate(_GPU, 2.0) is None

    def test_never_promises_less_than_an_instant(self):
        estimate = TranscriptionTimeEstimate()
        for audio, wall in ((1.0, 0.01), (2.0, 0.01), (3.0, 0.01)):
            estimate.record(_GPU, audio, wall)

        assert estimate.estimate(_GPU, 0.1) >= 0.05

    def test_learns_from_only_the_most_recent_two_hundred_of_thousands(self):
        estimate = TranscriptionTimeEstimate()
        for index in range(5000):
            slow_old_gpu = index < 4800
            estimate.record(_GPU, 1.0 + index % 30, 5.0 if slow_old_gpu else 0.5)

        assert estimate.estimate(_GPU, 10.0) == pytest.approx(0.5)


class TestSeedFrom:
    def test_learns_past_timings_from_earlier_sessions(self, tmp_path: Path):
        _write_session(
            tmp_path / "20260901T000000Z-00000001",
            [_transcription(audio, 0.3 + 0.02 * audio) for audio in (1.0, 5.0, 30.0)],
        )
        estimate = TranscriptionTimeEstimate()

        estimate.seed_from(tmp_path, skip=tmp_path / "current")

        assert estimate.estimate(_GPU, 20.0) == pytest.approx(0.7)

    def test_skips_failures_broken_lines_and_other_events(self, tmp_path: Path):
        _write_session(
            tmp_path / "20260901T000000Z-00000001",
            [
                {"event": "app_started"},
                _transcription(2.0, 99.0, error="CUDA out of memory"),
                _transcription(2.0, "slow"),
                _transcription(True, 0.4),
                "not an object",
                *[_transcription(2.0, 0.4, text="café ✓") for _ in range(3)],
            ],
            extra_lines='{"event": "transcription", "audio_sec',
        )
        estimate = TranscriptionTimeEstimate()

        estimate.seed_from(tmp_path, skip=tmp_path / "current")

        assert estimate.estimate(_GPU, 2.0) == pytest.approx(0.4)

    def test_does_not_learn_from_the_session_that_is_running(self, tmp_path: Path):
        current = tmp_path / "20260902T000000Z-00000002"
        _write_session(current, [_transcription(2.0, 0.4) for _ in range(3)])
        estimate = TranscriptionTimeEstimate()

        estimate.seed_from(tmp_path, skip=current)

        assert estimate.estimate(_GPU, 2.0) is None

    def test_copes_with_a_missing_log_folder(self, tmp_path: Path):
        estimate = TranscriptionTimeEstimate()

        estimate.seed_from(tmp_path / "missing", skip=tmp_path)

        assert estimate.estimate(_GPU, 2.0) is None

    def test_prefers_the_newest_sessions_when_there_are_hundreds(self, tmp_path: Path):
        for day in range(1, 301):
            wall = 0.4 if day > 250 else 6.0
            _write_session(
                tmp_path / f"2026{day:04d}T000000Z-{day:08d}",
                [_transcription(float(1 + day % 7), wall)],
            )
        estimate = TranscriptionTimeEstimate()

        estimate.seed_from(tmp_path, skip=tmp_path / "current")

        assert estimate.estimate(_GPU, 3.0) < 6.0
