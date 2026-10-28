"""Check the NPU engine on this computer: which device it gets, and how fast.

    python scripts\\check_npu.py [model] [recording.wav]

Uses the app's own models folder and engine, so the model must already be
downloaded (default ``underdog-bark-3.0``). The wav must be 16 kHz mono
16-bit, like the clips the diagnostics folder keeps. Without one it
transcribes 5 s of silence, which still shows the device and the speed.
"""

import sys
import time
import wave

import numpy as np

from skryb.core.paths import AppPaths
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.model_info import get_model_info
from skryb.transcription.openvino_parakeet_adapter import (
    OpenVinoParakeetAdapter,
    import_openvino,
)

_RATE = 16_000


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "underdog-bark-3.0"
    _describe_devices()
    engine = OpenVinoParakeetAdapter(AppPaths().models_dir, get_model_info(model).repo_id)
    started_at = time.monotonic()
    engine.load_model()
    print(f"{model} loaded in {time.monotonic() - started_at:.1f}s, encoder on {engine.encoder_device}")
    samples = _read_wav(sys.argv[2]) if len(sys.argv) > 2 else np.zeros(5 * _RATE, np.float32)
    for _ in range(3):
        result = engine.transcribe(AudioSegment(data=samples, sample_rate=_RATE))
        print(f"{len(samples) / _RATE:.1f}s of audio in {result.transcription_seconds:.2f}s: {result.text!r}")


def _describe_devices() -> None:
    openvino = import_openvino()
    core = openvino.Core()
    print(f"OpenVINO {openvino.__version__}, devices: {core.available_devices}")
    if "NPU" in core.available_devices:
        print(f"NPU: {core.get_property('NPU', 'FULL_DEVICE_NAME')}, "
              f"driver {core.get_property('NPU', 'NPU_DRIVER_VERSION')}")


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path) as recording:
        if (recording.getframerate(), recording.getnchannels(), recording.getsampwidth()) != (_RATE, 1, 2):
            raise SystemExit(f"{path} must be 16 kHz mono 16-bit")
        frames = recording.readframes(recording.getnframes())
    return np.frombuffer(frames, np.int16).astype(np.float32) / 32768


if __name__ == "__main__":
    main()
