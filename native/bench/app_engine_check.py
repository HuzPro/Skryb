"""Load qwen3-asr-1.7b through the app's own engine factory and transcribe a few corpus clips.

Proves the app path end to end: model lookup, install check, packing, the
DLL, and that torch is never imported.

    venv/Scripts/python.exe native/bench/app_engine_check.py [--clips 5]
"""
import argparse
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--bench", type=Path, default=Path("D:/VoiceToText_bench"))
    args = parser.parse_args()
    started = time.perf_counter()

    from voicetotext.core.json_config_adapter import JsonConfigAdapter
    from voicetotext.core.paths import AppPaths
    from voicetotext.recording.audio_segment import AudioSegment
    from voicetotext.transcription.engine_factory import build_engine_registry, create_engine_for_active_model
    from voicetotext.transcription.license_policy import LicensePolicy

    paths = AppPaths()
    with tempfile.TemporaryDirectory() as tmp:
        config = JsonConfigAdapter(Path(tmp) / "config.json")
        config.set("transcription.model_size", "qwen3-asr-1.7b")
        config.set("transcription.engine", "auto")
        engine = create_engine_for_active_model(
            config, paths, build_engine_registry(), _NoProbe(), license_policy=LicensePolicy(config),
        )
    imported = time.perf_counter()
    engine.load_model()
    loaded = time.perf_counter()
    engine.warmup()
    warm = time.perf_counter()
    print(f"engine {type(engine).__name__}: imports {imported - started:.2f}s, load {loaded - imported:.2f}s, "
          f"warmup {warm - loaded:.2f}s")

    clips = [json.loads(line) for line in (args.bench / "corpus.jsonl").open(encoding="utf-8")][: args.clips]
    for clip in clips:
        with wave.open(clip["audio"]) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        result = engine.transcribe(AudioSegment(data=pcm, sample_rate=16000))
        print(f"{clip['seconds']:6.1f}s audio {result.transcription_seconds * 1000:7.1f} ms  {result.text[:100]}")
    engine.dispose()
    assert "torch" not in sys.modules, "torch was imported"
    print("torch never imported")


class _NoProbe:
    """The cached probe the app would read from config: this machine's AMD card."""

    def detect(self):
        from voicetotext.transcription.engine_types import HardwareProfile

        return HardwareProfile(kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200")

    def peek(self):
        return self.detect()


if __name__ == "__main__":
    main()
