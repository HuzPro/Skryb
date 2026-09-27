import json
import struct
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.fakes.config import FakeConfig
from tests.fakes.hardware import FakeHardwareDetector
from voicetotext.core.paths import AppPaths
from voicetotext.core.startup_log import SESSION_MARKER
from voicetotext.diagnostics.session import SessionDiagnostics
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine_factory import build_engine_registry
from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.model_info import ModelInfo, get_model_info
from voicetotext.transcription.result import TranscriptionResult

pytestmark = pytest.mark.integration
_HARDWARE = FakeHardwareDetector(
    HardwareProfile(
        kind="amd_rocm", device_name="Test GPU", gfx_arch="gfx-test",
    )
)
_REGISTRY = build_engine_registry()


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
    @pytest.mark.parametrize(
        ("configured_engine", "ct2_installed", "expected_hardware"),
        [
            ("auto", True, "unprobed"),
            ("auto", False, "amd_rocm"),
            ("crisperwhisper_ct2", True, "unprobed"),
            ("crisperwhisper", True, "amd_rocm"),
        ],
    )
    def test_startup_diagnostics_only_probe_hardware_for_the_selected_engine(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        configured_engine: str,
        ct2_installed: bool,
        expected_hardware: str,
    ) -> None:
        paths = _session_paths(tmp_path / "VoiceToText – آواز")
        model = get_model_info("crisperwhisper-2.0-turbo")
        repo = paths.models_dir / f"models--{model.repo_id.replace('/', '--')}"
        snapshot = repo / "snapshots" / model.revision
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"weights")
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(model.revision, encoding="ascii")
        monkeypatch.setattr(
            "voicetotext.transcription.engine_factory._ctranslate2_installed",
            lambda: ct2_installed,
        )
        config = FakeConfig({
            "transcription.model_size": model.name,
            "transcription.engine": configured_engine,
            "transcription.device": "auto",
        })
        detector = FakeHardwareDetector()

        diagnostics = SessionDiagnostics(paths, config, _REGISTRY, detector)
        diagnostics.dispose()

        started = _events(paths)[0]
        assert started["hardware_kind"] == expected_hardware
        assert (detector.detect_calls == 0) == (expected_hardware == "unprobed")

    def test_should_record_app_started_for_unknown_model(
        self,
        tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        config = FakeConfig({
            "transcription.model_size": "some-model-i-typed-wrong",
        })

        SessionDiagnostics(paths, config, _REGISTRY, _HARDWARE)

        started = _events(paths)[0]
        assert started["event"] == "app_started"
        assert started["model"] == "some-model-i-typed-wrong"
        assert started["engine"] == "unknown"

    def test_should_record_unknown_engine_when_catalog_entry_has_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _session_paths(tmp_path)
        config = FakeConfig({"transcription.model_size": "model-without-engine"})

        def model_without_engine(_name: str) -> ModelInfo:
            return ModelInfo(
                name="model-without-engine",
                size_mb=1,
                description="Test model",
            )

        monkeypatch.setattr(
            "voicetotext.diagnostics.session.get_model_info",
            model_without_engine,
        )

        SessionDiagnostics(paths, config, _REGISTRY, _HARDWARE)

        started = _events(paths)[0]
        assert started["event"] == "app_started"
        assert started["engine"] == "unknown"

    def test_should_record_app_engine_transcription_and_stop_events(
        self,
        tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )
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
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )

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
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )

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
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )

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
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )

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
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )
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
        assert "café transcript" not in app_caplog.text


class TestCancelledRecordings:
    def test_keeps_the_audio_and_marks_it_cancelled(self, tmp_path: Path) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )

        diagnostics.recording_cancelled(_audio())

        cancelled = [event for event in _events(paths) if event["event"] == "cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0]["audio_path"] == "audio/0001.wav"
        assert cancelled[0]["audio_seconds"] == pytest.approx(1.0)
        assert (paths.session_dir / "audio" / "0001.wav").exists()

    def test_shares_numbering_with_transcriptions_so_files_never_collide(
        self, tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )
        requested = datetime.now(UTC)

        diagnostics.transcription_finished(
            _audio(), TranscriptionResult("hi", "en", 0.9, 0.1), requested,
        )
        diagnostics.recording_cancelled(_audio())

        indexes = [event["index"] for event in _events(paths) if "index" in event]
        assert indexes == [1, 2]

    def test_keeps_no_audio_without_content_consent(self, tmp_path: Path) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(paths, FakeConfig(), _REGISTRY, _HARDWARE)

        diagnostics.recording_cancelled(_audio())

        cancelled = [event for event in _events(paths) if event["event"] == "cancelled"]
        assert cancelled[0]["audio_path"] == ""
        assert not (paths.session_dir / "audio").exists()

    def test_records_a_cancel_that_captured_no_audio(self, tmp_path: Path) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )

        diagnostics.recording_cancelled(None)

        cancelled = [event for event in _events(paths) if event["event"] == "cancelled"]
        assert cancelled[0]["audio_seconds"] == 0.0
        assert cancelled[0]["audio_path"] == ""


class TestSessionRetention:
    def test_preserves_unmarked_folders_with_session_shaped_names(
        self, tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        unrelated = paths.diagnostics_dir / "20260101T000000Z-12345678"
        unrelated.mkdir()
        (unrelated / "session.log").write_text("keep me", encoding="utf-8")

        SessionDiagnostics(
            paths, FakeConfig({"diagnostics.retain_sessions": 1}),
            _REGISTRY, _HARDWARE,
        )

        assert unrelated.is_dir()

    def test_preserves_linked_sessions_and_their_targets(self, tmp_path: Path) -> None:
        paths = _session_paths(tmp_path)
        target = tmp_path / "outside"
        target.mkdir()
        (target / SESSION_MARKER).write_text("12345678" + "0" * 24, encoding="ascii")
        linked = paths.diagnostics_dir / "20260101T000000Z-12345678"
        try:
            linked.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("creating symlinks is unavailable")

        SessionDiagnostics(
            paths, FakeConfig({"diagnostics.retain_sessions": 1}),
            _REGISTRY, _HARDWARE,
        )

        assert linked.is_symlink()
        assert (target / SESSION_MARKER).exists()

    def test_preserves_unrelated_folders_when_retention_is_exceeded(
        self, tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        unrelated = paths.diagnostics_dir / "000-personal-documents"
        unrelated.mkdir()
        document = unrelated / "keep.txt"
        document.write_text("keep me", encoding="utf-8")

        SessionDiagnostics(
            paths, FakeConfig({"diagnostics.retain_sessions": 1}),
            _REGISTRY, _HARDWARE,
        )

        assert document.read_text(encoding="utf-8") == "keep me"

    def test_should_prune_oldest_sessions_beyond_retention(
        self,
        tmp_path: Path,
    ) -> None:
        paths = AppPaths(base=tmp_path)
        paths.ensure_dirs()
        for name in ("20260801T000000Z-00000001", "20260802T000000Z-00000002"):
            (paths.diagnostics_dir / name).mkdir()
            (paths.diagnostics_dir / name / SESSION_MARKER).write_text(
                name[-8:] + "0" * 24, encoding="ascii",
            )
        current = paths.diagnostics_dir / "20260803T000000Z-00000003"
        current.mkdir()
        paths.activate_session(current)
        config = FakeConfig({"diagnostics.retain_sessions": 2})

        SessionDiagnostics(paths, config, _REGISTRY, _HARDWARE)

        assert sorted(path.name for path in paths.diagnostics_dir.iterdir()) == [
            "20260802T000000Z-00000002",
            "20260803T000000Z-00000003",
        ]

    @pytest.mark.parametrize("retain", [0, -1])
    def test_should_keep_every_session_when_retention_is_zero_or_less(
        self, tmp_path: Path, retain: int,
    ) -> None:
        paths = AppPaths(base=tmp_path)
        paths.ensure_dirs()
        names = [f"202608{day:02d}T000000Z-{day:08d}" for day in range(1, 29)]
        for name in names:
            (paths.diagnostics_dir / name).mkdir()
            (paths.diagnostics_dir / name / SESSION_MARKER).write_text(
                name[-8:] + "0" * 24, encoding="ascii",
            )
        current = paths.diagnostics_dir / "20260901T000000Z-99999999"
        current.mkdir()
        paths.activate_session(current)

        SessionDiagnostics(
            paths, FakeConfig({"diagnostics.retain_sessions": retain}),
            _REGISTRY, _HARDWARE,
        )

        assert len(list(paths.diagnostics_dir.iterdir())) == len(names) + 1


class TestBestEffortWrites:
    @pytest.mark.parametrize("setting", ["diagnostics.enabled", "diagnostics.capture_content"])
    def test_stops_recording_content_without_restarting(
        self, tmp_path: Path, setting: str,
    ) -> None:
        paths = _session_paths(tmp_path)
        config = FakeConfig({"diagnostics.capture_content": True})
        diagnostics = SessionDiagnostics(paths, config, _REGISTRY, _HARDWARE)
        result = TranscriptionResult("first words", "en", 1.0, 0.1)
        diagnostics.transcription_finished(_audio(), result, datetime.now(UTC))

        config.set(setting, False)
        diagnostics.transcription_finished(
            _audio(), TranscriptionResult("private words", "en", 1.0, 0.1),
            datetime.now(UTC),
        )

        assert (paths.session_dir / "audio" / "0001.wav").exists()
        assert not (paths.session_dir / "audio" / "0002.wav").exists()
        assert not (paths.session_dir / "transcripts" / "0002.txt").exists()
        assert "private words" not in str(_events(paths))

    @pytest.mark.parametrize("enabled", [False, True])
    def test_requires_separate_content_consent_even_for_existing_configs(
        self, tmp_path: Path, app_caplog: pytest.LogCaptureFixture,
        enabled: bool,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.enabled": enabled}),
            _REGISTRY, _HARDWARE,
        )
        result = TranscriptionResult("private words", "en", 1.0, 0.1)

        diagnostics.transcription_finished(_audio(), result, datetime.now(UTC))

        assert not (paths.session_dir / "audio").exists()
        assert not (paths.session_dir / "transcripts").exists()
        assert "private words" not in app_caplog.text
        if enabled:
            assert "private words" not in str(_events(paths))

    def test_should_warn_without_raising_when_event_write_fails(
        self,
        tmp_path: Path,
        app_caplog: pytest.LogCaptureFixture,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}),
            _REGISTRY, _HARDWARE,
        )
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
        diagnostics = SessionDiagnostics(paths, config, _REGISTRY, _HARDWARE)

        diagnostics.transcription_finished(
            _audio(),
            RuntimeError("failed"),
            datetime.now(UTC),
        )
        diagnostics.dispose()

        assert not (paths.session_dir / "session.jsonl").exists()
        assert not (paths.session_dir / "audio").exists()
        assert not (paths.session_dir / "transcripts").exists()


class TestVoiceEdits:
    def test_a_voice_edit_is_logged_like_a_dictation_with_its_instruction_as_text(
        self, tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(
            paths, FakeConfig({"diagnostics.capture_content": True}), _REGISTRY, _HARDWARE,
        )
        instruction = TranscriptionResult("change Thursday to Friday", "en", 1.0, 0.1)

        diagnostics.transcription_finished(
            _audio(), instruction, datetime.now(UTC), kind="edit", output_seconds=0.4,
        )

        event = [e for e in _events(paths) if e["event"] == "transcription"][0]
        assert event["kind"] == "edit"
        assert event["text"] == "change Thursday to Friday"
        assert event["output_seconds"] == 0.4

    def test_the_transcription_estimate_leaves_out_the_time_spent_after_the_engine(
        self, tmp_path: Path,
    ) -> None:
        paths = _session_paths(tmp_path)
        diagnostics = SessionDiagnostics(paths, FakeConfig(), _REGISTRY, _HARDWARE)
        for _ in range(3):
            requested = datetime.now(UTC) - timedelta(seconds=3.0)
            diagnostics.transcription_finished(
                _audio(), TranscriptionResult("hi", "en", 1.0, 0.1), requested,
                output_seconds=2.5,
            )

        estimate = diagnostics.estimate_transcription_seconds(1.0)

        assert estimate is not None and estimate < 1.0
