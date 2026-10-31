"""Streaming Nemotron 3 Diarization: who is talking in a meeting, one label per 10 ms.

Ported to NumPy from transformers 5.20: the processor's streaming chunks and log-mel features
(``NemotronAsrStreamingFeatureExtractor``, which there needs librosa), used by both diarizers. The native one
(``NativeDiarizer``, ``native/diar``) also keeps the frame-stacking projection and the speaker cache
(``Nemotron3DiarizationSpeakerCache``) here, and runs each step's encoder, upsampler and classifier on an
RX 9060 XT (gfx1200) or an RTX 30-50 card without torch. The PyTorch one (``skryb.meetings.models.Diarizer``) runs
on any GPU.
"""

import ctypes
import json
import logging
import math
import os
import struct
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from skryb.core.exceptions import GpuEngineError, TranscriptionError
from skryb.meetings.segmenter import SILENCE
from skryb.transcription.native_engine_dll import native_diarizer_dll, rocm_runtime_dir

_logger = logging.getLogger(__name__)

_SPEECH = 0.5
_ERROR_BUFFER_BYTES = 1024
_LOG_ZERO_GUARD = 2.0 ** -24
_HIDDEN = 512


class DiarizerStream:
    """Feeds a streaming diarizer the chunks ``Nemotron3DiarizationProcessor`` cuts in streaming mode, and turns
    each chunk's speaker logits into labels: the strongest speaker per 10 ms frame, or SILENCE.

    *step* runs the model on one chunk's mel features and returns the logits of its own frames; it is given
    how many trailing encoder frames are look-ahead only (none for the last chunk).
    """

    def __init__(self, snapshot: Path, step: Callable[[np.ndarray, int], np.ndarray]) -> None:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        processor = json.loads((snapshot / "processor_config.json").read_text(encoding="utf-8"))
        self.features = StreamingFeatures(processor["feature_extractor"])
        self.factor = int(config["audio_config"]["subsampling_factor"])
        chunk, lookahead = processor["streaming_modes"][processor["streaming_mode"]]
        self._chunk, self._lookahead = int(chunk), int(lookahead)
        self._step_model = step
        self._audio = np.zeros(0, np.float32)
        self._offset = 0  # samples dropped from the front of self._audio
        self._step = 0

    @property
    def _mel_frames_per_chunk(self) -> int:
        return (self._chunk + self._lookahead) * self.factor

    @property
    def _mel_frames_per_step(self) -> int:
        return self._chunk * self.factor

    def push(self, audio: np.ndarray, last: bool = False) -> list[int]:
        self._audio = np.concatenate([self._audio, np.asarray(audio, dtype=np.float32)])
        labels: list[int] = []
        while True:
            start, length = self._chunk_bounds()
            if start - self._offset + length > len(self._audio):
                break
            labels += self._run(self._audio[start - self._offset:start - self._offset + length], last=False)
        if last:
            start, _ = self._chunk_bounds()
            tail = self._audio[start - self._offset:]
            if len(tail) > self.features.win_length:
                labels += self._run(tail, last=True)
        return labels

    def _chunk_bounds(self) -> tuple[int, int]:
        f = self.features
        if self._step == 0:
            return 0, (self._mel_frames_per_chunk - 1) * f.hop_length + f.win_length // 2
        start = self._step * self._mel_frames_per_step * f.hop_length - f.n_fft // 2
        return start, self._mel_frames_per_chunk * f.hop_length + f.win_length

    def _run(self, chunk: np.ndarray, last: bool) -> list[int]:
        features = self.features(chunk, center=self._step == 0, last=last)
        if not last and len(features) != self._mel_frames_per_chunk:
            raise TranscriptionError(f"a diarizer chunk held {len(features)} mel frames, not {self._mel_frames_per_chunk}")
        logits = self._step_model(features, 0 if last else self._lookahead)
        self._step += 1
        next_start, _ = self._chunk_bounds()
        drop = max(0, next_start - self._offset)
        self._audio, self._offset = self._audio[drop:], self._offset + drop
        probs = _sigmoid(logits)
        return np.where(probs.max(axis=-1) > _SPEECH, probs.argmax(axis=-1), SILENCE).tolist()


class NativeDiarizer:
    """Nemotron 3 Diarization with its model step on ``nd.dll``; the speaker cache in NumPy."""

    frame_seconds = 0.01

    def __init__(self, snapshot: Path, dll_path: Path | None = None) -> None:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        weights = snapshot / "model.safetensors"
        header, data_start = _safetensors_header(weights)
        self._embedder = _read_tensor(weights, header, data_start, "model.audio_tower.embedder.projection.weight")
        self._silence = _read_tensor(weights, header, data_start, "silence_embeds")
        self._cache = SpeakerCache(config["streaming_config"])
        self._stream = DiarizerStream(snapshot, self._model_step)
        self._engine = _Engine(dll_path or native_diarizer_dll() or Path("nd.dll"), weights, header, data_start,
                               int(config["audio_config"]["num_hidden_layers"]))

    def push(self, audio: np.ndarray, last: bool = False) -> list[int]:
        return self._stream.push(audio, last)

    def dispose(self) -> None:
        self._engine.free()

    def _model_step(self, features: np.ndarray, lookahead: int) -> np.ndarray:
        """``Nemotron3DiarizationForAudioFrameClassification.forward`` for one streaming chunk."""
        factor = self._stream.factor
        embeds = self._embed(features, factor)
        num_chunk_frames = len(embeds) - lookahead
        if num_chunk_frames < 1:
            raise TranscriptionError("a diarizer chunk had no frames to score")
        cached = self._cache.get_embeds()
        step_input = np.concatenate([cached, embeds]).astype(np.float32)
        logits = self._engine.forward(step_input)
        self._cache.update(step_input, logits, self._silence, num_chunk_frames)
        start = len(cached) * factor
        return logits[start:start + num_chunk_frames * factor][:len(features)]

    def _embed(self, features: np.ndarray, factor: int) -> np.ndarray:
        """Stack every ``subsampling_factor`` mel frames (the last group zero-padded) and project them."""
        padding = -len(features) % factor
        stacked = np.pad(features, ((0, padding), (0, 0))).reshape(-1, factor * features.shape[1])
        return stacked @ self._embedder.T


class StreamingFeatures:
    """``NemotronAsrStreamingFeatureExtractor``: pre-emphasis, 512-point STFT of a 400-sample symmetric Hann
    window, power, Slaney mel, natural log; never normalized. ``center`` pads 256 zeros on both sides, as for
    a stream's first chunk; later chunks are fed with that much audio already in front of them.
    """

    def __init__(self, config: dict) -> None:
        self.n_fft = int(config["n_fft"])
        self.hop_length = int(config["hop_length"])
        self.win_length = int(config["win_length"])
        self.preemphasis = float(config.get("preemphasis") or 0.0)
        offset = (self.n_fft - self.win_length) // 2
        window = np.zeros(self.n_fft, np.float32)
        window[offset:offset + self.win_length] = np.hanning(self.win_length)
        self._window = window
        self._filters = slaney_mel_filters(int(config["sampling_rate"]), self.n_fft, int(config["feature_size"]))

    def __call__(self, audio: np.ndarray, center: bool, last: bool) -> np.ndarray:
        """[frames][mel bins] for the frames this chunk fully holds."""
        x = np.asarray(audio, dtype=np.float32)
        if self.preemphasis:
            x = np.concatenate([x[:1], x[1:] - np.float32(self.preemphasis) * x[:-1]])
        length = len(x)
        if last and not center:
            extra = self.n_fft // 2 - self.hop_length
            x = np.pad(x, (0, extra))
            length += extra
        if center:
            x = np.pad(x, (self.n_fft // 2, self.n_fft // 2))
            valid = (length + self.n_fft // 2 * 2 - self.n_fft) // self.hop_length
        else:
            valid = (length - self.n_fft) // self.hop_length + 1
        frames = 1 + (len(x) - self.n_fft) // self.hop_length
        if frames < 1 or valid < 1:
            return np.zeros((0, self._filters.shape[0]), np.float32)
        index = np.arange(self.n_fft)[None, :] + self.hop_length * np.arange(frames)[:, None]
        spectrum = np.fft.rfft(x[index] * self._window, axis=-1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
        mel = power @ self._filters.T
        return np.log(mel + np.float32(_LOG_ZERO_GUARD)).astype(np.float32)[:valid]


def slaney_mel_filters(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """``librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr / 2, norm="slaney")``: [n_mels][n_fft // 2 + 1]."""
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0

    def to_mel(hz: np.ndarray) -> np.ndarray:
        hz = np.asarray(hz, dtype=np.float64)
        return np.where(hz >= min_log_hz, min_log_mel + np.log(np.maximum(hz, 1e-10) / min_log_hz) / logstep, hz / f_sp)

    def to_hz(mel: np.ndarray) -> np.ndarray:
        return np.where(mel >= min_log_mel, min_log_hz * np.exp(logstep * (mel - min_log_mel)), f_sp * mel)

    fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    mel_freqs = to_hz(np.linspace(to_mel(0.0), to_mel(sample_rate / 2.0), n_mels + 2))
    fdiff = np.diff(mel_freqs)
    ramps = np.subtract.outer(mel_freqs, fft_freqs)
    weights = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_freqs[2:n_mels + 2] - mel_freqs[:n_mels]))[:, None]
    return weights


class SpeakerCache:
    """``Nemotron3DiarizationSpeakerCache`` in streaming mode, for one stream: the Arrival-Order Speaker
    Cache and the FIFO queue of the latest encoder frames, both encoder inputs (512 wide).
    """

    def __init__(self, config: dict) -> None:
        self.fifo_length = int(config["fifo_length"])
        self.update_period = int(config["speaker_cache_update_period"])
        self.cache_length = int(config["speaker_cache_length"])
        self.num_silence_frames = int(config["speaker_cache_silence_frames_per_speaker"])
        self.threshold = float(config["prediction_score_threshold"])
        self.latest_boost = float(config["latest_frames_score_boost"])
        self.num_speakers = int(config["num_speakers"])
        self.factor = int(config["subsampling_factor"])
        budget = self.cache_length // self.num_speakers - self.num_silence_frames
        self.min_positive_scores = math.floor(budget * float(config["min_positive_scores_rate"]))
        self.num_strong_boosted = math.floor(budget * float(config["strong_boost_rate"]))
        self.num_weak_boosted = math.floor(budget * float(config["weak_boost_rate"]))
        self.embeds = np.zeros((self.cache_length, _HIDDEN), np.float32)
        self.probs = np.zeros((self.cache_length, self.num_speakers), np.float32)
        self.fifo = np.zeros((self.fifo_length, _HIDDEN), np.float32)
        self.num_cache_frames = 0
        self.num_fifo_frames = 0
        self.is_compressed = False

    def get_embeds(self) -> np.ndarray:
        return np.concatenate([self.embeds[:self.num_cache_frames], self.fifo[:self.num_fifo_frames]])

    def update(self, step_input: np.ndarray, logits: np.ndarray, silence: np.ndarray, num_chunk_frames: int) -> None:
        """Push the step's chunk frames to the FIFO, moving its oldest to the cache when it overflows."""
        num_cache, num_fifo = self.num_cache_frames, self.num_fifo_frames
        probs = _sigmoid(logits).reshape(-1, self.factor, self.num_speakers).mean(axis=1).astype(np.float32)
        chunk_start = num_cache + num_fifo
        chunk = step_input[chunk_start:chunk_start + num_chunk_frames]
        fifo = np.concatenate([self.fifo[:num_fifo], chunk])
        popped = self._num_popped_frames(len(fifo))
        if popped:
            fifo_probs = probs[num_cache:num_cache + len(fifo)]
            # An uncompressed cache still holds plain chunk frames, whose probabilities this step re-estimates;
            # a compressed one is out of order, so the probs stored with its frames are the only ones.
            stored = self.probs[:num_cache] if self.is_compressed else probs[:num_cache]
            cache_embeds = np.concatenate([self.embeds[:num_cache], fifo[:popped]])
            cache_probs = np.concatenate([stored, fifo_probs[:popped]])
            fifo = fifo[popped:]
            if len(cache_embeds) > self.cache_length:
                cache_embeds, cache_probs = self._compress(cache_embeds, cache_probs, silence)
                self.is_compressed = True
            self.num_cache_frames = len(cache_embeds)
            self.embeds[:self.num_cache_frames] = cache_embeds
            self.probs[:self.num_cache_frames] = cache_probs
        self.num_fifo_frames = len(fifo)
        self.fifo[:self.num_fifo_frames] = fifo

    def _num_popped_frames(self, num_fifo_frames: int) -> int:
        if num_fifo_frames <= self.fifo_length:
            return 0
        return min(max(self.update_period, num_fifo_frames - self.fifo_length), num_fifo_frames)

    def _frame_scores(self, probs: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            log_probs = np.log(np.maximum(probs, self.threshold))
            log_complements = np.log(np.maximum(1.0 - probs, self.threshold))
        scores = log_probs - log_complements + log_complements.sum(axis=-1, keepdims=True) - math.log(0.5)
        is_speech = probs > 0.5
        scores = np.where(is_speech, scores, -np.inf)
        is_positive = scores > 0
        has_enough_positive = is_positive.sum(axis=0, keepdims=True) >= self.min_positive_scores
        return np.where(~is_positive & is_speech & has_enough_positive, -np.inf, scores).astype(np.float32)

    @staticmethod
    def _boost(scores: np.ndarray, count: int, boost: float) -> np.ndarray:
        top = np.argpartition(-scores, count - 1, axis=0)[:count]
        boosted = scores.copy()
        np.add.at(boosted, (top, np.arange(scores.shape[1])[None, :].repeat(count, axis=0)), np.float32(boost))
        return boosted

    def _compress(self, embeds: np.ndarray, probs: np.ndarray, silence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The ``cache_length`` most telling frames, grouped by speaker in arrival order, with reserved
        silence slots per speaker filled with the learned silence embedding."""
        num_frames, num_speakers = probs.shape
        scores = self._frame_scores(probs)
        scores[self.cache_length:] += np.float32(self.latest_boost)
        scores = self._boost(scores, self.num_strong_boosted, -2.0 * math.log(0.5))
        scores = self._boost(scores, self.num_weak_boosted, -math.log(0.5))
        scores = np.concatenate([scores, np.full((self.num_silence_frames, num_speakers), np.inf, np.float32)])
        embeds = np.concatenate([embeds, silence.reshape(1, -1).astype(np.float32)])
        probs = np.concatenate([probs, np.zeros((1, num_speakers), np.float32)])
        num_scored = num_frames + self.num_silence_frames
        sentinel = num_scored * num_speakers
        flat = scores.T.reshape(-1)
        top = np.argpartition(-flat, self.cache_length - 1)[:self.cache_length]
        top = np.sort(np.where(flat[top] == -np.inf, sentinel, top))
        frames = np.where(top == sentinel, num_frames, np.minimum(top % num_scored, num_frames))
        return embeds[frames], probs[frames]


class _Engine:
    def __init__(self, dll_path: Path, weights: Path, header: dict, data_start: int, layers: int) -> None:
        if not dll_path.exists():
            raise GpuEngineError(f"The native diarizer isn't installed ({dll_path} is missing).")
        hip_bin = rocm_runtime_dir()
        if hip_bin is not None:
            os.add_dll_directory(str(hip_bin))
        lib = ctypes.CDLL(str(dll_path))
        lib.nd_load.restype = ctypes.c_void_p
        lib.nd_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64),
                                ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.nd_forward.restype = ctypes.c_int
        lib.nd_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_float), ctypes.c_char_p, ctypes.c_int]
        lib.nd_stats.restype = ctypes.c_char_p
        lib.nd_stats.argtypes = [ctypes.c_void_p]
        lib.nd_free.restype = None
        lib.nd_free.argtypes = [ctypes.c_void_p]
        names = [name for name in header if name != "__metadata__"]
        if {header[name]["dtype"] for name in names} != {"F32"}:
            raise TranscriptionError("The native diarizer needs the fp32 checkpoint")
        encoded = [name.encode() for name in names]
        offsets = [data_start + header[name]["data_offsets"][0] for name in names]
        self._error = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        started_at = time.perf_counter()
        handle = lib.nd_load(str(weights).encode(), (ctypes.c_char_p * len(encoded))(*encoded),
                             (ctypes.c_uint64 * len(offsets))(*offsets), len(names), layers,
                             self._error, _ERROR_BUFFER_BYTES)
        if not handle:
            raise GpuEngineError(f"The native diarizer failed to load: {self._error.value.decode(errors='replace')}")
        self._lib, self._handle = lib, handle
        _logger.info("[load_model native diarizer] complete in %.2fs | %s",
                     time.perf_counter() - started_at, lib.nd_stats(handle).decode())

    def forward(self, step_input: np.ndarray) -> np.ndarray:
        if self._handle is None:
            raise TranscriptionError("The native diarizer was freed")
        rows = len(step_input)
        embeds = np.ascontiguousarray(step_input, dtype=np.float32)
        logits = np.empty((rows * 8, 8), np.float32)
        status = self._lib.nd_forward(self._handle, embeds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), rows, rows,
                                      logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self._error,
                                      _ERROR_BUFFER_BYTES)
        if status != 0:
            raise TranscriptionError(f"The native diarizer failed: {self._error.value.decode(errors='replace')}")
        return logits

    def free(self) -> None:
        if self._handle is not None:
            self._lib.nd_free(self._handle)
            self._handle = None


def _safetensors_header(weights: Path) -> tuple[dict, int]:
    with weights.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def _read_tensor(weights: Path, header: dict, data_start: int, name: str) -> np.ndarray:
    info = header[name]
    if info["dtype"] != "F32":
        raise TranscriptionError(f"{name} is {info['dtype']}, not fp32")
    start, end = info["data_offsets"]
    with weights.open("rb") as handle:
        handle.seek(data_start + start)
        data = handle.read(end - start)
    return np.frombuffer(data, np.float32).reshape(info["shape"]).copy()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)

