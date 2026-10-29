"""Streaming Nemotron 3 Diarization: who is talking in a meeting, one label per 10 ms.

Ported to NumPy from transformers 5.20: the processor's streaming chunks and log-mel features
(``NemotronAsrStreamingFeatureExtractor``, which there needs librosa, a package the app doesn't ship).
``skryb.meetings.models.Diarizer`` runs the model on them.
"""

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.meetings.segmenter import SILENCE

_SPEECH = 0.5
_LOG_ZERO_GUARD = 2.0 ** -24


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










def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)

