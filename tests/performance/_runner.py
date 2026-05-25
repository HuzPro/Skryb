"""Run one benchmark combination and return a structured result.

``run_benchmark`` never raises: any error during build / load / warmup
/ transcribe / dispose is captured in the returned ``BenchmarkResult``
with ``status="error"``. The caller is responsible for recording the
failure into the JSON history.

Sequence per combo:
    1. Build engine via the registry
    2. Time ``load_model()``
    3. Time ``warmup()`` (one silent-audio inference)
    4. Reset CUDA peak stats (cuda only) — happens AFTER warmup so the
       peak reflects timed-run allocations only
    5. N timed runs of ``transcribe(audio)`` via ``time.monotonic()``
       outside the adapter, so the metric is uniform across adapters
       whose internal ``transcription_seconds`` varies in what it
       includes (preprocessing vs decode-only)
    6. Capture VRAM peak (cuda only)
    7. Full teardown: dispose, del, gc, empty_cache, synchronize
"""

import gc
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine_registry import EngineRegistry, EngineSpec
from voicetotext.transcription.engine_types import EngineId

from tests.performance._combos import BenchmarkCombo


_logger = logging.getLogger(__name__)


_LANGUAGE = "en"
_DEFAULT_BEAM_SIZE_BY_ENGINE: dict[EngineId, int] = {
    "transformers": 5,
    "faster_whisper": 5,
    "granite_speech": 4,
    "nano_parakeet": 1,
}


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    status: Literal["ok", "error"]
    load_seconds: float | None = None
    warmup_seconds: float | None = None
    runs_seconds: tuple[float, ...] = ()
    text_sample: str | None = None
    vram_peak_mb: float | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def median_seconds(self) -> float | None:
        return statistics.median(self.runs_seconds) if self.runs_seconds else None

    @property
    def mean_seconds(self) -> float | None:
        return statistics.mean(self.runs_seconds) if self.runs_seconds else None

    @property
    def stdev_seconds(self) -> float | None:
        if not self.runs_seconds:
            return None
        if len(self.runs_seconds) < 2:
            return 0.0
        return statistics.stdev(self.runs_seconds)

    @property
    def min_seconds(self) -> float | None:
        return min(self.runs_seconds) if self.runs_seconds else None

    @property
    def max_seconds(self) -> float | None:
        return max(self.runs_seconds) if self.runs_seconds else None


def run_benchmark(
    *,
    combo: BenchmarkCombo,
    audio: AudioSegment,
    n_timed_runs: int,
    registry: EngineRegistry,
    models_dir: Path,
) -> BenchmarkResult:
    spec = _build_spec(combo, models_dir)
    engine = None
    try:
        engine = registry.build(combo.engine_id, spec)
        load_seconds = _timed(engine.load_model)
        warmup_seconds = _timed(engine.warmup)
        _reset_peak(combo.device)
        runs_seconds: list[float] = []
        last_text = ""
        for _ in range(n_timed_runs):
            start = time.monotonic()
            result = engine.transcribe(audio)
            runs_seconds.append(time.monotonic() - start)
            last_text = result.text
        vram_peak_mb = _capture_peak_mb(combo.device)
        return BenchmarkResult(
            status="ok",
            load_seconds=load_seconds,
            warmup_seconds=warmup_seconds,
            runs_seconds=tuple(runs_seconds),
            text_sample=last_text[:200] if last_text else "",
            vram_peak_mb=vram_peak_mb,
        )
    except Exception as exc:
        _logger.exception("benchmark failed for %s", combo.test_id)
        return BenchmarkResult(
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
        )
    finally:
        _teardown(engine, combo.device)


def _build_spec(combo: BenchmarkCombo, models_dir: Path) -> EngineSpec:
    from voicetotext.transcription.model_info import ALL_MODELS

    info = ALL_MODELS[combo.model_name]
    repo_id = info.ct2_repo_id if combo.engine_id == "faster_whisper" else info.repo_id
    return EngineSpec(
        model_name=combo.model_name,
        repo_id=repo_id,
        device=combo.device,
        language=_LANGUAGE,
        beam_size=_DEFAULT_BEAM_SIZE_BY_ENGINE[combo.engine_id],
        model_dir=models_dir,
    )


def _timed(fn) -> float:
    start = time.monotonic()
    fn()
    return time.monotonic() - start


def _reset_peak(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        _logger.exception("peak reset failed; VRAM number will be unreliable")


def _capture_peak_mb(device: str) -> float | None:
    if device != "cuda":
        return None
    try:
        import torch
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        _logger.exception("peak capture failed")
        return None


def _teardown(engine, device: str) -> None:
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            _logger.exception("dispose() raised — continuing teardown")
    del engine
    gc.collect()
    if device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            _logger.exception("post-dispose cuda cleanup raised — continuing")
