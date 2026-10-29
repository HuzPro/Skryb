"""The meeting-only models: speech detection (CPU) and who is speaking (GPU).

Torch is imported here only when a meeting starts, so a launch that never
records a meeting never pays for it.
"""

import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from skryb.meetings.capture import RATE
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
    """Nemotron 3 Diarization in streaming mode: who is talking, one label per 10 ms.

    Speakers stay numbered the same for the whole meeting through the model's
    speaker cache. fp32: bf16 hangs MIOpen's kernel search on RDNA 4.
    """

    frame_seconds = 0.01

    def __init__(self, snapshot: Path) -> None:
        torch = import_torch()
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor: Any = AutoProcessor.from_pretrained(snapshot)
        model = AutoModelForAudioFrameClassification.from_pretrained(snapshot, dtype=torch.float32)
        self._model: Any = model.to(self._device).eval()
        self._audio = np.zeros(0, np.float32)
        self._offset = 0  # samples dropped from the front of self._audio
        self._step = 0
        self._cache: Any = None

    def push(self, audio: np.ndarray, last: bool = False) -> list[int]:
        self._audio = np.concatenate([self._audio, audio])
        labels: list[int] = []
        while True:
            start, length = self._chunk_bounds()
            if start - self._offset + length > len(self._audio):
                break
            labels += self._run(self._audio[start - self._offset:start - self._offset + length], last=False)
        if last:
            start, _ = self._chunk_bounds()
            tail = self._audio[start - self._offset:]
            if len(tail) > self._processor.feature_extractor.win_length:
                labels += self._run(tail, last=True)
        return labels

    def _chunk_bounds(self) -> tuple[int, int]:
        p = self._processor
        if self._step == 0:
            return 0, int(p.num_samples_first_audio_chunk)
        return int(p.audio_chunk_start(self._step * p.num_mel_frames_per_step)), int(p.num_samples_per_audio_chunk)

    def _run(self, chunk: np.ndarray, last: bool) -> list[int]:
        inputs = self._processor(
            chunk, sampling_rate=RATE, is_streaming=True,
            is_first_audio_chunk=self._step == 0, is_last_audio_chunk=last, return_tensors="pt",
        ).to(self._device)
        if last:
            inputs["num_lookahead_frames"] = 0
        with self._torch.no_grad():
            out = self._model(**inputs, speaker_cache=self._cache)
        self._cache = out.speaker_cache
        self._step += 1
        next_start, _ = self._chunk_bounds()
        drop = max(0, next_start - self._offset)
        self._audio, self._offset = self._audio[drop:], self._offset + drop
        best = out.logits[0].sigmoid().max(dim=-1)
        labels: list[int] = self._torch.where(best.values > _SPEECH, best.indices, SILENCE).tolist()
        return labels
