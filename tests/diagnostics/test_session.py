import json
import struct
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.fakes.config import FakeConfig
from voicetotext.core.paths import AppPaths
from voicetotext.diagnostics.session import SessionDiagnostics
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.result import TranscriptionResult

pytestmark = pytest.mark.integration
_HARDWARE = HardwareProfile(
    kind="amd_rocm",
    device_name="Test GPU",
    gfx_arch="gfx-test",
)


def _session_paths(base: Path, name: str = "current") -> AppPaths:
    paths = AppPaths(base=base)
    paths.ensure_dirs()
    session_dir = paths.diagnostics_dir / name
    session_dir.mkdir()
    paths.activate_session(session_dir)
    return paths


def _events(paths: AppPaths) -> list[dict[str, object]]:
    lines = (
        (paths.session_dir / "session.jsonl")
        .read_text(
            encoding="utf-8",
        )
        .splitlines()
    )
    return [json.loads(line) for line in lines]


def _audio() -> AudioSegment:
    return AudioSegment(
        data=np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        sample_rate=5,
    )


class TestSessionEvents:
    def test_should_record_app_engine_transcription_and_stop_events(
        self,
        tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(paths, FakeConfig(), _HARDWARE)
        diagnostics.engine_loaded(1.25, "cuda")
        requested_at = datetime.now(UTC) - timedelta(seconds=2)
        diagnostics.transcription_finished(
            _audio(),
            TranscriptionResult(
                text="partial result",
                language="en",
                language_probability=0.9,
                transcription_seconds=0.75,
            ),
            requested_at,
        )
        diagnostics.dispose()

        events = _events(paths)
        assert [event["event"] for event in events] == [
            "app_started",
            "engine_loaded",
            "transcription",
            "app_stopped",
        ]
        started = events[0]
        assert started["model"] == "parakeet-tdt-0.6b-v3"
        assert started["engine"] == "nano_parakeet"
        assert started["device"] == "cuda"
        assert started["hardware_kind"] == "amd_rocm"
        assert isinstance(started["app_version"], str)
        assert isinstance(started["os"], str)
        assert isinstance(started["python_version"], str)

        transcription = events[2]
        assert transcription["index"] == 1
        assert transcription["requested_at"] == requested_at.isoformat().replace(
            "+00:00",
            "Z",
        )
        assert isinstance(transcription["finished_at"], str)
        assert float(transcription["wall_seconds"]) >= 2.0
        assert transcription["engine_seconds"] == 0.75
        assert transcription["audio_seconds"] == 1.0
        assert transcription["sample_rate"] == 5
        assert transcription["audio_path"] == "audio/0001.wav"
        assert transcription["text"] == "partial result"
        assert transcription["transcript_path"] == "transcripts/0001.txt"
        assert transcription["char_count"] == 14
        assert transcription["word_count"] == 2
        assert transcription["is_empty"] is False
        assert transcription["error"] is None
        assert events[3]["transcription_count"] == 1

    def test_should_record_failed_transcription_with_all_fields(
        self,
        tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(paths, FakeConfig(), _HARDWARE)

        diagnostics.transcription_finished(
            _audio(),
            RuntimeError("decoder failed"),
            datetime.now(UTC),
        )

        event = _events(paths)[-1]
        assert event["event"] == "transcription"
        assert event["engine_seconds"] is None
        assert event["text"] == ""
        assert event["char_count"] == 0
        assert event["word_count"] == 0
        assert event["is_empty"] is True
        assert event["error"] == "decoder failed"
        assert (paths.session_dir / "transcripts" / "0001.txt").read_text(
            encoding="utf-8",
        ) == ""

    def test_should_record_engine_load_failure(
        self,
        tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(paths, FakeConfig(), _HARDWARE)

        diagnostics.engine_load_failed(3.5, RuntimeError("out of memory"), "cuda")

        event = _events(paths)[-1]
        assert event["event"] == "engine_load_failed"
        assert event["model"] == "parakeet-tdt-0.6b-v3"
        assert event["engine"] == "nano_parakeet"
        assert event["device"] == "cuda"
        assert event["seconds"] == 3.5
        assert event["error"] == "out of memory"

    def test_should_write_app_stopped_once(self, tmp_path: Path) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(paths, FakeConfig(), _HARDWARE)

        diagnostics.dispose()
        diagnostics.dispose()

        events = _events(paths)
        assert sum(event["event"] == "app_stopped" for event in events) == 1
        assert float(events[-1]["uptime_seconds"]) >= 0.0


class TestTranscriptionArtifacts:
    def test_should_write_exact_audio_as_16_bit_pcm_wav(
        self,
        tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(paths, FakeConfig(), _HARDWARE)

        diagnostics.transcription_finished(
            _audio(),
            RuntimeError("failed"),
            datetime.now(UTC),
        )

        with wave.open(str(paths.session_dir / "audio" / "0001.wav"), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 5
            assert wav.getnframes() == 5
            samples = struct.unpack("<5h", wav.readframes(5))
        assert samples == (-32768, -16384, 0, 16384, 32767)

    def test_should_write_utf8_transcript_and_summary_log(
        self,
        tmp_path: Path,
        app_caplog: pytest.LogCaptureFixture,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(paths, FakeConfig(), _HARDWARE)
        result = TranscriptionResult(
            text="café transcript",
            language="en",
            language_probability=1.0,
            transcription_seconds=0.25,
        )

        diagnostics.transcription_finished(
            _audio(),
            result,
            datetime.now(UTC),
        )

        assert (paths.session_dir / "transcripts" / "0001.txt").read_text(
            encoding="utf-8",
        ) == "café transcript"
        assert "audio_seconds=1.000" in app_caplog.text
        assert "engine_seconds=0.250" in app_caplog.text
        assert "char_count=15" in app_caplog.text
        assert "word_count=2" in app_caplog.text
        assert "café transcript" in app_caplog.text


class TestSessionRetention:
    def test_should_prune_oldest_sessions_beyond_retention(
        self,
        tmp_path: Path,
    ) -> None:
        paths = AppPaths(base=tmp_path)
        paths.ensure_dirs()
        for name in ("20260801T000000Z-old0001", "20260802T000000Z-old0002"):
            (paths.diagnostics_dir / name).mkdir()
        current = paths.diagnostics_dir / "20260803T000000Z-current1"
        current.mkdir()
        paths.activate_session(current)
        config = FakeConfig({"diagnostics.retain_sessions": 2})

        SessionDiagnostics(paths, config, _HARDWARE)

        assert sorted(path.name for path in paths.diagnostics_dir.iterdir()) == [
            "20260802T000000Z-old0002",
            "20260803T000000Z-current1",
        ]


class TestBestEffortWrites:
    def test_should_warn_without_raising_when_event_write_fails(
        self,
        tmp_path: Path,
        app_caplog: pytest.LogCaptureFixture,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(paths, FakeConfig(), _HARDWARE)
        event_path = paths.session_dir / "session.jsonl"
        event_path.unlink()
        event_path.mkdir()

        diagnostics.engine_loaded(1.0, "cuda")

        assert "Diagnostics write failed for engine_loaded" in app_caplog.text

    def test_should_skip_artifacts_and_events_when_disabled(
        self,
        tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        config = FakeConfig({"diagnostics.enabled": False})
        diagnostics = SessionDiagnostics(paths, config, _HARDWARE)

        diagnostics.transcription_finished(
            _audio(),
            RuntimeError("failed"),
            datetime.now(UTC),
        )
        diagnostics.dispose()

        assert not (paths.session_dir / "session.jsonl").exists()
        assert not (paths.session_dir / "audio").exists()
        assert not (paths.session_dir / "transcripts").exists()
