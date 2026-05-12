"""Benchmark Granite Speech inference latency against the
``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`` flag.

The flag is supposed to flip PyTorch from the math fallback to the
AOTriton flash / mem-efficient attention kernels on AMD ROCm. We have
no objective measurement of whether that actually wins on this
hardware, so this script runs N transcriptions on a fixed-length
synthetic audio sample and prints summary statistics. Run it twice
with the env var toggled to compare:

    PowerShell:
        $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "0"
        python scripts/bench_granite_sdpa.py
        $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
        python scripts/bench_granite_sdpa.py

The model is loaded inside this process, so the env var must be set
in the shell **before** Python starts. PyTorch reads the flag during
SDPA backend selection, which has happened by the time the first
``model.generate`` returns.
"""
import os
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


def _build_synthetic_audio():
    from voicetotext.recording.audio_segment import AudioSegment

    samples = int(_AUDIO_DURATION_SECONDS * _SAMPLE_RATE)
    time_axis = np.arange(samples, dtype=np.float32) / _SAMPLE_RATE
    data = (
        _TONE_AMPLITUDE
        * np.sin(2.0 * np.pi * _TONE_FREQUENCY_HZ * time_axis)
    ).astype(np.float32)
    return AudioSegment(data=data, sample_rate=_SAMPLE_RATE)


def _build_adapter():
    from voicetotext.transcription.granite_speech_adapter import (
        GraniteSpeechAdapter,
    )

    models_dir = Path(os.environ["APPDATA"]) / "VoiceToText" / "models"
    adapter = GraniteSpeechAdapter(
        repo_id=_REPO_ID,
        device="cuda",
        model_dir=models_dir,
    )
    adapter.load_model()
    return adapter


def _print_active_flag():
    value = os.environ.get(
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "<unset>",
    )
    print(f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = {value}")


def _time_transcribe(adapter, audio) -> float:
    started_at = time.monotonic()
    adapter.transcribe(audio)
    return time.monotonic() - started_at


def _print_summary(timings: list[float]) -> None:
    print()
    print(f"Runs:   {len(timings)}")
    print(f"Median: {statistics.median(timings):.3f}s")
    print(f"Mean:   {statistics.mean(timings):.3f}s")
    print(f"Min:    {min(timings):.3f}s")
    print(f"Max:    {max(timings):.3f}s")
    stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    print(f"Stdev:  {stdev:.3f}s")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    _print_active_flag()

    print("Loading model...")
    load_start = time.monotonic()
    adapter = _build_adapter()
    print(f"Model loaded in {time.monotonic() - load_start:.2f}s")

    audio = _build_synthetic_audio()

    print("Warmup pass...")
    adapter.warmup()

    print(
        f"Running {_BENCHMARK_RUNS} transcriptions on "
        f"{_AUDIO_DURATION_SECONDS:.0f}s audio..."
    )
    timings: list[float] = []
    for index in range(_BENCHMARK_RUNS):
        elapsed = _time_transcribe(adapter, audio)
        timings.append(elapsed)
        print(f"  Run {index + 1}: {elapsed:.3f}s")

    _print_summary(timings)

    adapter.dispose()


if __name__ == "__main__":
    main()
