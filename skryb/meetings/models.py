"""The meeting-only models: speech detection (CPU) and who is speaking (GPU).

Torch is imported here only when a meeting starts, so a launch that never
records a meeting never pays for it.
"""

import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from skryb.meetings.capture import RATE
from skryb.meetings.diarization import DiarizerStream
from skryb.meetings.segmenter import SILENCE
from skryb.transcription.model_info import ModelInfo

# NVIDIA's streaming diarizer: up to 8 speakers, numbered by first appearance.
DIARIZER = ModelInfo(
    name="nemotron-3-diarization",
    size_mb=397,
    description="Tells up to 8 speakers apart as they talk (NVIDIA Nemotron 3 Diarization)",
    repo_id="nvidia/Nemotron-3-Diarization",
    revision="f667ed73aee57d40cc39428eb768b4fd87a0a29e",
)
_SPEECH = 0.5


class Labeller(Protocol):
    """One label per frame: a speaker index, or SILENCE."""

    frame_seconds: float

    def push(self, audio: np.ndarray, last: bool = False) -> list[int]: ...


def import_torch() -> Any:
    """Torch, with the ROCm settings this GPU family needs set before its first import."""
    # MIOpen's exhaustive kernel search has hung on RDNA 4 under Windows; pick kernels by heuristic.
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    from skryb.core.rocm_env import prepare_rocm_environment

    prepare_rocm_environment()
    import torch

    return torch


def gpu_available() -> bool:
    return bool(import_torch().cuda.is_available())


def release_gpu() -> None:
    torch = import_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Vad:
    """Silero speech detection, one label per 32 ms window, on the CPU. Speech is speaker 0."""

    WINDOW = 512
    frame_seconds = WINDOW / RATE

    def __init__(self) -> None:
        self._torch = import_torch()
        threads = self._torch.get_num_threads()
        # Its import runs torch.set_num_threads(1), which would leave every other torch model
        # in the process (dictation on the CPU included) on one thread.
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._torch.set_num_threads(threads)
        self._rest = np.zeros(0, np.float32)

    def push(self, audio: np.ndarray, last: bool = False) -> list[int]:
        audio = np.concatenate([self._rest, audio])
        whole = len(audio) // self.WINDOW * self.WINDOW
        self._rest = audio[whole:]
        labels: list[int] = []
        with self._torch.no_grad():
            for window in audio[:whole].reshape(-1, self.WINDOW):
                speech = self._model(self._torch.from_numpy(window), RATE).item() > _SPEECH
                labels.append(0 if speech else SILENCE)
        return labels


class Diarizer:
    """Nemotron 3 Diarization in streaming mode on PyTorch: who is talking, one label per 10 ms.

    Speakers stay numbered the same for the whole meeting through the model's
    speaker cache. fp32: bf16 hangs MIOpen's kernel search on RDNA 4. The chunks
    and features come from ``DiarizerStream``, so the app needs no librosa.
    """

    frame_seconds = 0.01

    def __init__(self, snapshot: Path) -> None:
        torch = import_torch()
        from transformers import AutoModelForAudioFrameClassification

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForAudioFrameClassification.from_pretrained(snapshot, dtype=torch.float32)
        self._model: Any = model.to(self._device).eval()
        self._cache: Any = None
        self._stream = DiarizerStream(snapshot, self._model_step)

    def push(self, audio: np.ndarray, last: bool = False) -> list[int]:
        return self._stream.push(audio, last)

    def _model_step(self, features: np.ndarray, lookahead: int) -> np.ndarray:
        torch = self._torch
        input_features = torch.from_numpy(features)[None].to(self._device)
        mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = self._model(input_features=input_features, attention_mask=mask, num_lookahead_frames=lookahead,
                              speaker_cache=self._cache)
        self._cache = out.speaker_cache
        logits: np.ndarray = out.logits[0].float().cpu().numpy()
        return logits
