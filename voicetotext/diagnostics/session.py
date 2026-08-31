import json
import logging
import platform
import shutil
import threading
import wave
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.paths import AppPaths
from voicetotext.core.startup_log import (
    process_started_at,
    process_uptime_seconds,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.model_info import DEFAULT_MODEL, get_model_info
from voicetotext.transcription.result import TranscriptionResult

_logger = logging.getLogger(__name__)


class SessionDiagnostics:
    def __init__(
        self,
        paths: AppPaths,
        config: ConfigurationStore,
        hardware: HardwareProfile,
    ) -> None:
        self._paths = paths
        self._config = config
        self._hardware = hardware
        self._enabled = config.get("diagnostics.enabled", True) is True
        retain_sessions = config.get("diagnostics.retain_sessions", 50)
        self._retain_sessions = (
            retain_sessions
            if isinstance(retain_sessions, int)
            and not isinstance(retain_sessions, bool)
            else 50
        )
        self._events_path = paths.session_dir / "session.jsonl"
        self._lock = threading.Lock()
        self._transcription_count = 0
        self._stopped = False
        self._active_model, self._active_engine = self._model_context()
        self._active_device = self._configured_device()
        self._prune_sessions()
        self._record_app_started()

    def engine_loaded(self, seconds: float, device: str) -> None:
        model, engine = self._model_context()
        self._active_model = model
        self._active_engine = engine
        self._active_device = device
        self._write_event(
            {
                "event": "engine_loaded",
                "at": self._utc_now(),
                "model": model,
                "engine": engine,
                "device": device,
                "seconds": seconds,
                "error": None,
            }
        )

    def engine_load_failed(
        self,
        seconds: float,
        error: Exception,
        device: str,
    ) -> None:
        model, engine = self._model_context()
        self._write_event(
            {
                "event": "engine_load_failed",
                "at": self._utc_now(),
                "model": model,
                "engine": engine,
                "device": device,
                "seconds": seconds,
                "error": str(error),
            }
        )

    def transcription_finished(
        self,
        audio: AudioSegment | None,
        outcome: TranscriptionResult | Exception,
        requested_at: datetime,
    ) -> None:
        index = self._next_transcription_index()
        finished_at = datetime.now(UTC)
        if isinstance(outcome, Exception):
            text = ""
            engine_seconds: float | None = None
            error: str | None = str(outcome)
        else:
            text = outcome.text
            engine_seconds = outcome.transcription_seconds
            error = None
        audio_path = Path("audio") / f"{index:04d}.wav"
        transcript_path = Path("transcripts") / f"{index:04d}.txt"
        if self._enabled and audio is not None:
            self._attempt_write(
                f"audio {index}",
                lambda: self._save_audio(self._paths.session_dir / audio_path, audio),
            )
        if self._enabled:
            self._attempt_write(
                f"transcript {index}",
                lambda: self._save_transcript(
                    self._paths.session_dir / transcript_path,
                    text,
                ),
            )
            self._write_event(
                {
                    "event": "transcription",
                    "index": index,
                    "requested_at": self._format_time(requested_at),
                    "finished_at": self._format_time(finished_at),
                    "wall_seconds": (finished_at - requested_at).total_seconds(),
                    "engine_seconds": engine_seconds,
                    "model": self._active_model,
                    "engine": self._active_engine,
                    "device": self._active_device,
                    "audio_seconds": audio.duration_seconds
                    if audio is not None
                    else 0.0,
                    "sample_rate": audio.sample_rate if audio is not None else 0,
                    "audio_path": audio_path.as_posix() if audio is not None else "",
                    "text": text,
                    "transcript_path": transcript_path.as_posix(),
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "is_empty": not text.strip(),
                    "error": error,
                }
            )
        _logger.info(
            "transcription index=%d audio_seconds=%.3f engine_seconds=%s "
            "char_count=%d word_count=%d text=%r",
            index,
            audio.duration_seconds if audio is not None else 0.0,
            f"{engine_seconds:.3f}" if engine_seconds is not None else "n/a",
            len(text),
            len(text.split()),
            text,
        )

    def dispose(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            transcription_count = self._transcription_count
        self._write_event(
            {
                "event": "app_stopped",
                "stopped_at": self._utc_now(),
                "uptime_seconds": process_uptime_seconds(),
                "transcription_count": transcription_count,
            }
        )

    def _record_app_started(self) -> None:
        self._write_event(
            {
                "event": "app_started",
                "started_at": self._format_time(process_started_at()),
                "app_version": self._app_version(),
                "model": self._active_model,
                "engine": self._active_engine,
                "device": self._active_device,
                "hardware_kind": self._hardware.kind,
                "os": platform.platform(),
                "python_version": platform.python_version(),
            }
        )

    def _write_event(self, event: dict[str, object]) -> None:
        if not self._enabled:
            return
        self._attempt_write(
            str(event["event"]),
            lambda: self._append_event(event),
        )

    def _append_event(self, event: dict[str, object]) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def _attempt_write(self, operation: str, write: Callable[[], None]) -> None:
        try:
            write()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Diagnostics write failed for %s: %s", operation, exc)

    def _prune_sessions(self) -> None:
        retained_previous = max(self._retain_sessions, 1) - 1
        try:
            previous = sorted(
                path
                for path in self._paths.diagnostics_dir.iterdir()
                if path.is_dir() and path != self._paths.session_dir
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Diagnostics session scan failed: %s", exc)
            return
        expired_count = max(len(previous) - retained_previous, 0)
        for path in previous[:expired_count]:
            self._attempt_write(
                f"prune {path.name}",
                lambda path=path: shutil.rmtree(path),
            )

    def _next_transcription_index(self) -> int:
        with self._lock:
            self._transcription_count += 1
            return self._transcription_count

    def _model_context(self) -> tuple[str, str]:
        model = str(self._config.get("transcription.model_size", DEFAULT_MODEL))
        model_info = get_model_info(model)
        configured_engine = self._config.get("transcription.engine", "auto")
        engine = (
            configured_engine
            if isinstance(configured_engine, str) and configured_engine != "auto"
            else model_info.compatible_engines[0]
        )
        return model, engine

    def _configured_device(self) -> str:
        configured_device = self._config.get("transcription.device", "auto")
        if isinstance(configured_device, str) and configured_device != "auto":
            return configured_device
        return "cpu" if self._hardware.kind == "cpu" else "cuda"

    @staticmethod
    def _save_audio(path: Path, audio: AudioSegment) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(audio.data * 32768.0, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(audio.sample_rate)
            output.writeframes(pcm.tobytes())

    @staticmethod
    def _save_transcript(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _format_time(value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @classmethod
    def _utc_now(cls) -> str:
        return cls._format_time(datetime.now(UTC))

    @staticmethod
    def _app_version() -> str:
        try:
            return version("voicetotext")
        except PackageNotFoundError:
            return "unknown"
