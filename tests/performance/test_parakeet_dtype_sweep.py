"""Compare nano_parakeet performance across dtype configurations.

The production adapter ([nano_parakeet_adapter.py:79-93]) hard-codes
``torch.float32`` for stability on AMD ROCm bf16 kernels. This test
empirically measures the impact of that choice on RDNA4 (gfx1200) by
loading the same model with float32, bfloat16, and float16, and timing
short + long audio inference for each.

We bypass the project's :class:`NanoParakeetAdapter` and call
``nano_parakeet.from_pretrained`` directly so the dtype can vary
without touching production code. The hf_hub_download patch
(redirecting reads to the app's models dir with local_files_only) is
replicated locally.

Outputs land in ``tests/performance/.results/dtype_comparison_results.json``.
"""

from __future__ import annotations

import gc
import json
import logging
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voicetotext.recording.audio_segment import AudioSegment

from tests.performance.conftest import PerformanceContext


_logger = logging.getLogger(__name__)


_REPO_ID = "nvidia/parakeet-tdt-0.6b-v3"
_TIMED_RUNS_SHORT = 3
_TIMED_RUNS_LONG = 1

_DTYPES_TO_TEST = ("float32", "bfloat16", "float16")

_RESULTS_PATH = Path(__file__).parent / ".results" / "dtype_comparison_results.json"


pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


@dataclass
class _AudioRunStats:
    runs_seconds: list[float]
    vram_peak_mb: float | None
    text_sample: str

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


@dataclass
class _DtypeRunRecord:
    dtype: str
    status: str  # "ok" | "error"
    load_seconds: float | None = None
    warmup_seconds: float | None = None
    short: _AudioRunStats | None = None
    long: _AudioRunStats | None = None
    error_type: str | None = None
    error_message: str | None = None


def test_should_compare_parakeet_performance_across_dtypes(
    perf_context: PerformanceContext,
    short_audio: AudioSegment,
    long_audio: AudioSegment,
) -> None:
    if perf_context.hardware.kind == "cpu":
        pytest.skip("dtype comparison only meaningful on GPU")

    records: list[_DtypeRunRecord] = []
    for dtype_name in _DTYPES_TO_TEST:
        print(f"\n=== Running parakeet sweep with dtype={dtype_name} ===", flush=True)
        record = _run_one_dtype(
            dtype_name=dtype_name,
            short_audio=short_audio,
            long_audio=long_audio,
            models_dir=perf_context.models_dir,
        )
        records.append(record)
        _summarize_to_stdout(record)
        _full_cuda_cleanup()

    _save_results(records, perf_context)
    _print_comparison_table(records, short_audio, long_audio)


def _run_one_dtype(
    *,
    dtype_name: str,
    short_audio: AudioSegment,
    long_audio: AudioSegment,
    models_dir: Path,
) -> _DtypeRunRecord:
    model = None
    try:
        import torch

        dtype = _resolve_dtype(dtype_name, torch)
        load_seconds, model = _load_parakeet(dtype, models_dir)
        warmup_seconds = _warmup(model, torch)
        _reset_peak()
        short_stats = _time_audio(
            model=model,
            audio=short_audio,
            n_runs=_TIMED_RUNS_SHORT,
        )
        _reset_peak()
        long_stats = _time_audio(
            model=model,
            audio=long_audio,
            n_runs=_TIMED_RUNS_LONG,
        )
        return _DtypeRunRecord(
            dtype=dtype_name,
            status="ok",
            load_seconds=load_seconds,
            warmup_seconds=warmup_seconds,
            short=short_stats,
            long=long_stats,
        )
    except Exception as exc:
        _logger.exception("dtype %s failed", dtype_name)
        return _DtypeRunRecord(
            dtype=dtype_name,
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
        )
    finally:
        del model
        _full_cuda_cleanup()


def _resolve_dtype(name: str, torch_module) -> object:
    mapping = {
        "float32": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
    }
    return mapping[name]


def _load_parakeet(dtype: object, models_dir: Path) -> tuple[float, object]:
    import nano_parakeet

    original_download = nano_parakeet.hf_hub_download
    nano_parakeet.hf_hub_download = _make_patched_download(
        original_download, models_dir
    )
    try:
        start = time.monotonic()
        model = nano_parakeet.from_pretrained(
            model_name=_REPO_ID,
            device="cuda",
            dtype=dtype,
        )
        load_seconds = time.monotonic() - start
        return load_seconds, model
    finally:
        nano_parakeet.hf_hub_download = original_download


def _make_patched_download(
    original: Callable[..., object], models_dir: Path,
) -> Callable[..., object]:
    cache_dir = str(models_dir)

    def patched(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("cache_dir", cache_dir)
        kwargs.setdefault("local_files_only", True)
        return original(*args, **kwargs)

    return patched


def _warmup(model: object, torch_module) -> float:
    import numpy as np

    silent_samples = np.zeros(int(16000 * 2.0), dtype=np.float32)
    start = time.monotonic()
    model.transcribe(silent_samples)
    return time.monotonic() - start


def _time_audio(
    *,
    model: object,
    audio: AudioSegment,
    n_runs: int,
) -> _AudioRunStats:
    runs: list[float] = []
    last_text = ""
    for _ in range(n_runs):
        start = time.monotonic()
        text = model.transcribe(audio.data)
        runs.append(time.monotonic() - start)
        last_text = str(text)
    return _AudioRunStats(
        runs_seconds=runs,
        vram_peak_mb=_capture_peak_mb(),
        text_sample=last_text.strip()[:200],
    )


def _reset_peak() -> None:
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        _logger.exception("peak reset failed")


def _capture_peak_mb() -> float | None:
    try:
        import torch
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        _logger.exception("peak capture failed")
        return None


def _full_cuda_cleanup() -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def _summarize_to_stdout(record: _DtypeRunRecord) -> None:
    if record.status == "error":
        print(
            f"  dtype={record.dtype} STATUS=error "
            f"{record.error_type}: {record.error_message}",
            flush=True,
        )
        return
    print(
        f"  dtype={record.dtype} STATUS=ok "
        f"load={record.load_seconds:.2f}s warmup={record.warmup_seconds:.2f}s "
        f"short_median={record.short.median_seconds*1000:.1f}ms "
        f"short_vram={record.short.vram_peak_mb:.1f}MB "
        f"long_median={record.long.median_seconds:.2f}s "
        f"long_vram={record.long.vram_peak_mb:.1f}MB",
        flush=True,
    )


def _save_results(
    records: list[_DtypeRunRecord], ctx: PerformanceContext,
) -> None:
    payload = {
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": {
            "kind": ctx.hardware.kind,
            "device_name": ctx.hardware.device_name,
            "gfx_arch": ctx.hardware.gfx_arch,
        },
        "model": _REPO_ID,
        "audio_durations_seconds": {
            "short": _audio_duration("Testing_1_2_3.mp3"),
            "long": _audio_duration("LongStory.mp3"),
        },
        "records": [_record_to_json(r) for r in records],
    }
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _RESULTS_PATH.with_suffix(_RESULTS_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(_RESULTS_PATH)
    print(f"\nSaved dtype comparison results -> {_RESULTS_PATH}", flush=True)


def _audio_duration(name: str) -> float:
    from tests.helpers.audio_file_loader import load_audio_file
    from tests.fixtures import FIXTURE_DIR

    return load_audio_file(FIXTURE_DIR / name).duration_seconds


def _record_to_json(r: _DtypeRunRecord) -> dict[str, object]:
    base: dict[str, object] = {
        "dtype": r.dtype,
        "status": r.status,
        "load_seconds": r.load_seconds,
        "warmup_seconds": r.warmup_seconds,
        "error_type": r.error_type,
        "error_message": r.error_message,
    }
    if r.short is not None:
        base["short_audio"] = _stats_to_json(r.short)
    if r.long is not None:
        base["long_audio"] = _stats_to_json(r.long)
    return base


def _stats_to_json(s: _AudioRunStats) -> dict[str, object]:
    return {
        "runs_seconds": list(s.runs_seconds),
        "median_seconds": s.median_seconds,
        "mean_seconds": s.mean_seconds,
        "stdev_seconds": s.stdev_seconds,
        "vram_peak_mb": s.vram_peak_mb,
        "text_sample": s.text_sample,
    }


def _print_comparison_table(
    records: list[_DtypeRunRecord],
    short_audio: AudioSegment,
    long_audio: AudioSegment,
) -> None:
    print("\n" + "=" * 100, flush=True)
    print("Parakeet dtype comparison", flush=True)
    print(
        f"  Short audio: {short_audio.duration_seconds:.2f}s   "
        f"Long audio: {long_audio.duration_seconds:.2f}s",
        flush=True,
    )
    print("=" * 100, flush=True)
    header = (
        f"{'dtype':<10} {'load_s':>8} {'warmup_s':>10} "
        f"{'short_ms':>10} {'short_rtf':>10} {'short_vram':>11} "
        f"{'long_s':>9} {'long_rtf':>10} {'long_vram':>11} {'status':>8}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in records:
        if r.status == "error":
            print(
                f"{r.dtype:<10} {'-':>8} {'-':>10} {'-':>10} {'-':>10} "
                f"{'-':>11} {'-':>9} {'-':>10} {'-':>11} {'error':>8}",
                flush=True,
            )
            continue
        short_med = r.short.median_seconds
        long_med = r.long.median_seconds
        short_rtf = short_audio.duration_seconds / short_med if short_med else 0.0
        long_rtf = long_audio.duration_seconds / long_med if long_med else 0.0
        print(
            f"{r.dtype:<10} {r.load_seconds:>8.2f} {r.warmup_seconds:>10.2f} "
            f"{short_med * 1000:>10.1f} {short_rtf:>10.2f}x "
            f"{r.short.vram_peak_mb:>10.1f}M "
            f"{long_med:>9.2f} {long_rtf:>10.2f}x "
            f"{r.long.vram_peak_mb:>10.1f}M {'ok':>8}",
            flush=True,
        )
    print("=" * 100, flush=True)
