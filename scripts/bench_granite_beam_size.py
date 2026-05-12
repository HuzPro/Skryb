"""Benchmark Granite Speech inference latency across ``num_beams``.

Sweeps beam counts 1..5 against a fixed-length synthetic audio
sample to quantify the accuracy/speed tradeoff exposed by Tier 3's
``transcription.beam_size`` config knob. Users can pick a value that
matches their priorities — IBM's tech report recommends 4 for
quality; this measurement says how much wall-clock you pay for it.

Run:

    python scripts/bench_granite_beam_size.py

``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`` is set automatically
(production default) so the numbers reflect the path users actually
get.
"""
import os

# Mirror :func:`voicetotext.__main__._apply_torch_env_overrides` — the
# AOTriton flag has to be set before any ``import torch`` upstack.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import statistics
import time
import warnings
from pathlib import Path

import numpy as np


_REPO_ID = "ibm-granite/granite-speech-4.1-2b"
_SAMPLE_RATE = 16000
_AUDIO_DURATION_SECONDS = 20.0
_TONE_FREQUENCY_HZ = 440.0
_TONE_AMPLITUDE = 0.1
_BENCHMARK_RUNS = 5
_BEAM_SIZES = (1, 2, 3, 4, 5)
_BASELINE_BEAM_SIZE = 4


def _build_synthetic_audio():
    from voicetotext.recording.audio_segment import AudioSegment

    samples = int(_AUDIO_DURATION_SECONDS * _SAMPLE_RATE)
    time_axis = np.arange(samples, dtype=np.float32) / _SAMPLE_RATE
    data = (
        _TONE_AMPLITUDE
        * np.sin(2.0 * np.pi * _TONE_FREQUENCY_HZ * time_axis)
    ).astype(np.float32)
    return AudioSegment(data=data, sample_rate=_SAMPLE_RATE)


def _build_adapter(num_beams: int):
    from voicetotext.transcription.granite_speech_adapter import (
        GraniteSpeechAdapter,
    )

    models_dir = Path(os.environ["APPDATA"]) / "VoiceToText" / "models"
    adapter = GraniteSpeechAdapter(
        repo_id=_REPO_ID,
        device="cuda",
        model_dir=models_dir,
        num_beams=num_beams,
    )
    adapter.load_model()
    return adapter


def _time_transcribe(adapter, audio) -> float:
    started_at = time.monotonic()
    adapter.transcribe(audio)
    return time.monotonic() - started_at


def _bench_one_beam_count(adapter, audio, num_beams: int) -> list[float]:
    # Mutating the adapter's ``_num_beams`` between sweeps keeps the
    # model resident in VRAM. ``num_beams`` is read at generate-time
    # so the change takes effect on the next transcribe; the first
    # transcribe at each new beam dimension pays a one-off kernel-
    # compile cost — the dedicated warmup absorbs it.
    adapter._num_beams = num_beams
    print(f"\n--- num_beams={num_beams} ---")
    print("  warmup...")
    adapter.warmup()
    print(f"  running {_BENCHMARK_RUNS} transcriptions...")
    timings: list[float] = []
    for index in range(_BENCHMARK_RUNS):
        elapsed = _time_transcribe(adapter, audio)
        timings.append(elapsed)
        print(f"    run {index + 1}: {elapsed:.3f}s")
    return timings


def _print_summary(results: dict[int, list[float]]) -> None:
    baseline = statistics.median(results[_BASELINE_BEAM_SIZE])
    print()
    print("=== Summary ===")
    print(
        f"{'beams':<8}{'median':<10}{'mean':<10}"
        f"{'min':<10}{'max':<10}{'vs beam=4':<12}"
    )
    for beam in _BEAM_SIZES:
        timings = results[beam]
        median = statistics.median(timings)
        speedup = baseline / median
        marker = " (baseline)" if beam == _BASELINE_BEAM_SIZE else ""
        print(
            f"{beam:<8}"
            f"{median:.3f}s    "
            f"{statistics.mean(timings):.3f}s    "
            f"{min(timings):.3f}s    "
            f"{max(timings):.3f}s    "
            f"{speedup:.2f}x{marker}"
        )


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    flag = os.environ.get(
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "<unset>",
    )
    print(f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = {flag}")

    print("Loading model...")
    load_start = time.monotonic()
    adapter = _build_adapter(num_beams=_BEAM_SIZES[0])
    print(f"Model loaded in {time.monotonic() - load_start:.2f}s")

    audio = _build_synthetic_audio()
    results: dict[int, list[float]] = {}
    for beam in _BEAM_SIZES:
        results[beam] = _bench_one_beam_count(adapter, audio, beam)

    _print_summary(results)
    adapter.dispose()


if __name__ == "__main__":
    main()
