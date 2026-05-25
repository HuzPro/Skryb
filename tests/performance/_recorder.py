"""Accumulating JSON history of benchmark runs.

Per-identity monotonic versioning: each ``(model, engine, device,
audio)`` tuple has an independent version counter that increments every
time it gets benchmarked, regardless of whether the run succeeded or
failed. Old entries are never deleted; the file accumulates forever.

Persistence is atomic — writes go to ``<path>.tmp`` first, then
``Path.replace`` swaps them in. A KeyboardInterrupt mid-write therefore
either leaves the previous file intact or replaces it completely; the
file is never half-written.

Corrupt JSON on read is treated as a recoverable disaster: the file is
moved aside to ``<stem>.corrupted.<timestamp>.json``, a warning is
logged, and the recorder starts with an empty history. The next
``save()`` writes a fresh file. Silent overwrite is the only thing that
actually destroys data, so we never do it.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voicetotext.transcription.engine_types import HardwareProfile

from tests.performance._identity import BenchmarkIdentity
from tests.performance._runner import BenchmarkResult


_logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class StoredEntry:
    version: int
    timestamp: str
    identity: BenchmarkIdentity
    hardware: HardwareProfile
    audio_duration_seconds: float
    result: BenchmarkResult

    def to_json(self) -> dict[str, object]:
        median = self.result.median_seconds
        rtf: float | None = None
        if median is not None and median > 0:
            rtf = self.audio_duration_seconds / median
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "identity": self.identity.to_json(),
            "hardware": {
                "kind": self.hardware.kind,
                "device_name": self.hardware.device_name,
                "gfx_arch": self.hardware.gfx_arch,
            },
            "audio_duration_seconds": self.audio_duration_seconds,
            "load_seconds": self.result.load_seconds,
            "warmup_seconds": self.result.warmup_seconds,
            "runs_seconds": list(self.result.runs_seconds),
            "median_seconds": median,
            "mean_seconds": self.result.mean_seconds,
            "stdev_seconds": self.result.stdev_seconds,
            "min_seconds": self.result.min_seconds,
            "max_seconds": self.result.max_seconds,
            "real_time_factor": rtf,
            "vram_peak_mb": self.result.vram_peak_mb,
            "text_sample": self.result.text_sample,
            "status": self.result.status,
            "error_type": self.result.error_type,
            "error_message": self.result.error_message,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "StoredEntry":
        identity = BenchmarkIdentity.from_json(data["identity"])  # type: ignore[arg-type]
        hardware_data = data["hardware"]
        hardware = HardwareProfile(
            kind=hardware_data["kind"],  # type: ignore[arg-type,index]
            device_name=hardware_data["device_name"],  # type: ignore[arg-type,index]
            gfx_arch=hardware_data["gfx_arch"],  # type: ignore[arg-type,index]
        )
        runs_seconds = tuple(data.get("runs_seconds") or ())
        result = BenchmarkResult(
            status=data["status"],  # type: ignore[arg-type]
            load_seconds=data.get("load_seconds"),  # type: ignore[arg-type]
            warmup_seconds=data.get("warmup_seconds"),  # type: ignore[arg-type]
            runs_seconds=runs_seconds,  # type: ignore[arg-type]
            text_sample=data.get("text_sample"),  # type: ignore[arg-type]
            vram_peak_mb=data.get("vram_peak_mb"),  # type: ignore[arg-type]
            error_type=data.get("error_type"),  # type: ignore[arg-type]
            error_message=data.get("error_message"),  # type: ignore[arg-type]
        )
        return cls(
            version=int(data["version"]),  # type: ignore[arg-type]
            timestamp=str(data["timestamp"]),
            identity=identity,
            hardware=hardware,
            audio_duration_seconds=float(data["audio_duration_seconds"]),  # type: ignore[arg-type]
            result=result,
        )


class PerformanceRecorder:

    def __init__(self, json_path: Path) -> None:
        self._path = json_path
        self._entries: list[StoredEntry] = []
        self._loaded = False
        self._initial_count = 0

    def load(self) -> None:
        if not self._path.exists():
            self._loaded = True
            self._initial_count = 0
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._handle_corrupt(exc)
            return
        if data.get("schema_version") != _SCHEMA_VERSION:
            _logger.warning(
                "schema_version mismatch — expected %s, got %s; attempting to read anyway",
                _SCHEMA_VERSION,
                data.get("schema_version"),
            )
        for raw_entry in data.get("entries", []):
            try:
                self._entries.append(StoredEntry.from_json(raw_entry))
            except (KeyError, TypeError, ValueError) as exc:
                _logger.warning("skipping unparseable entry: %s", exc)
        self._loaded = True
        self._initial_count = len(self._entries)

    def _handle_corrupt(self, exc: json.JSONDecodeError) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = self._path.with_name(f"{self._path.stem}.corrupted.{stamp}.json")
        shutil.move(str(self._path), str(backup))
        _logger.warning(
            "benchmark_results.json was corrupted (%s); moved to %s", exc, backup,
        )
        self._loaded = True
        self._initial_count = 0

    def next_version_for(self, identity: BenchmarkIdentity) -> int:
        if not self._loaded:
            raise RuntimeError("call load() before next_version_for()")
        matching = [e.version for e in self._entries if e.identity == identity]
        return max(matching) + 1 if matching else 1

    def record(
        self,
        identity: BenchmarkIdentity,
        result: BenchmarkResult,
        hardware: HardwareProfile,
        audio_duration_seconds: float,
    ) -> StoredEntry:
        entry = StoredEntry(
            version=self.next_version_for(identity),
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            identity=identity,
            hardware=hardware,
            audio_duration_seconds=audio_duration_seconds,
            result=result,
        )
        self._entries.append(entry)
        return entry

    def save(self) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "entries": [entry.to_json() for entry in self._entries],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def entries_in_current_run(self) -> list[StoredEntry]:
        return self._entries[self._initial_count:]

    def all_entries(self) -> list[StoredEntry]:
        return list(self._entries)
