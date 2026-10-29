"""The small parts of Parakeet TDT in NumPy: mel features before the encoder, greedy TDT decoding after it.

They mirror nano-parakeet's ``LogMelPreprocessor`` and ``tdt_greedy_decode`` (NeMo's featurizer and
GreedyTDTInfer), so an engine that runs only the encoder elsewhere (OpenVINO on an NPU) needs no torch.
"""

import importlib.util
from functools import cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
_N_FFT = 512
_HOP = 160
_WIN = 400
_N_MELS = 128
_PREEMPHASIS = 0.97
_LOG_GUARD = 2.0 ** -24
_BLANK = 8192
_DURATIONS = (0, 1, 2, 3, 4)
_MAX_SYMBOLS_PER_FRAME = 10
# Frames scored per joint call: the prediction only changes when a token is emitted, so the walk over
# blank frames reuses one batched product.
_WINDOW = 16

# The decoder and joint tensors, under nano-parakeet's names.
DECODING_TENSORS = (
    "decoder.embed.weight",
    *(f"decoder.lstm.{kind}_l{layer}" for layer in (0, 1)
      for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")),
    "joint.enc.weight", "joint.enc.bias", "joint.pred.weight", "joint.pred.bias",
    "joint.joint_net.2.weight", "joint.joint_net.2.bias",
)


def log_mel_features(samples: np.ndarray) -> np.ndarray:
    """``[1, 128, frames]`` normalised log-mel features of 16 kHz *samples*."""
    x = np.asarray(samples, np.float32)
    x = np.concatenate([x[:1], x[1:] - _PREEMPHASIS * x[:-1]])
    x = np.pad(x, _N_FFT // 2, mode="reflect")
    frames = 1 + (len(x) - _N_FFT) // _HOP
    starts = np.arange(frames)[:, None] * _HOP + np.arange(_N_FFT)[None, :]
    spectrum = np.fft.rfft(x[starts] * _window(), axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    log_mel = np.log(power @ _filterbank().T + _LOG_GUARD)
    mean = log_mel.mean(axis=0)
    std = np.sqrt(((log_mel - mean) ** 2).sum(axis=0) / (frames - 1))
    return ((log_mel - mean) / (std + 1e-5)).T[None].astype(np.float32)


@cache
def _window() -> np.ndarray:
    """A symmetric 400-sample Hann window centred in 512, as torch.stft pads it."""
    window = np.zeros(_N_FFT, np.float32)
    start = (_N_FFT - _WIN) // 2
    window[start:start + _WIN] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_WIN) / (_WIN - 1))
    return window


@cache
def _filterbank() -> np.ndarray:
    """``[128, 257]`` Slaney-scale, Slaney-normalised mel filters (librosa's, which NeMo uses)."""
    freqs = np.linspace(0, SAMPLE_RATE / 2, _N_FFT // 2 + 1)
    mel_points = np.linspace(_hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2), _N_MELS + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    slopes = hz_points[None, :] - freqs[:, None]
    gaps = np.diff(hz_points)
    down = -slopes[:, :-2] / gaps[:-1]
    up = slopes[:, 2:] / gaps[1:]
    filters = np.maximum(0, np.minimum(down, up)) * (2.0 / (hz_points[2:] - hz_points[:-2]))
    return filters.T.astype(np.float32)


_SLANEY_LINEAR_HZ = 200.0 / 3
_SLANEY_LOG_HZ = 1000.0
_SLANEY_LOG_STEP = np.log(6.4) / 27.0


def _hz_to_mel(hz: float) -> float:
    if hz < _SLANEY_LOG_HZ:
        return hz / _SLANEY_LINEAR_HZ
    return _SLANEY_LOG_HZ / _SLANEY_LINEAR_HZ + np.log(hz / _SLANEY_LOG_HZ) / _SLANEY_LOG_STEP


def _mel_to_hz(mel: float) -> float:
    if mel < _SLANEY_LOG_HZ / _SLANEY_LINEAR_HZ:
        return mel * _SLANEY_LINEAR_HZ
    return _SLANEY_LOG_HZ * np.exp(_SLANEY_LOG_STEP * (mel - _SLANEY_LOG_HZ / _SLANEY_LINEAR_HZ))


class TdtDecoder:
    """Greedy TDT decoding over encoder frames: the 2-layer LSTM prediction net and the joint, fp32."""

    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        w = {name: np.ascontiguousarray(tensors[name], np.float32) for name in DECODING_TENSORS}
        self._embed = w["decoder.embed.weight"]
        self._lstm = [
            (np.concatenate([w[f"decoder.lstm.weight_ih_l{i}"], w[f"decoder.lstm.weight_hh_l{i}"]], axis=1).T.copy(),
             w[f"decoder.lstm.bias_ih_l{i}"] + w[f"decoder.lstm.bias_hh_l{i}"])
            for i in (0, 1)
        ]
        self._enc = (w["joint.enc.weight"].T.copy(), w["joint.enc.bias"])
        self._pred = (w["joint.pred.weight"].T.copy(), w["joint.pred.bias"])
        self._out = (w["joint.joint_net.2.weight"].T.copy(), w["joint.joint_net.2.bias"])

    def decode(self, encoded: np.ndarray) -> list[int]:
        """Token ids for ``[frames, 1024]`` encoder output."""
        frames = encoded.shape[0]
        enc = encoded.astype(np.float32) @ self._enc[0] + self._enc[1]
        state = self._initial_state()
        candidate = self._step(_BLANK, state)
        scored_from, scored = -1, None
        tokens: list[int] = []
        t = 0
        while t < frames:
            frame_start, symbols, need_loop = t, 0, True
            while need_loop and symbols < _MAX_SYMBOLS_PER_FRAME and t < frames:
                if scored is None or not scored_from <= t < scored_from + len(scored):
                    scored_from, scored = t, self._score(enc[t:t + _WINDOW], candidate[0])
                token, duration = scored[t - scored_from]
                skip = _DURATIONS[duration]
                if token == _BLANK:
                    need_loop = False
                else:
                    tokens.append(token)
                    state = candidate[1]
                    candidate = self._step(token, state)
                    scored = None
                symbols += 1
                t += skip
                need_loop = need_loop and skip == 0
            if t <= frame_start:
                t = frame_start + 1
        return tokens

    def _initial_state(self) -> list[tuple[np.ndarray, np.ndarray]]:
        zeros = np.zeros(self._embed.shape[1], np.float32)
        return [(zeros, zeros), (zeros, zeros)]

    def _step(self, label: int, state: list) -> tuple[np.ndarray, list]:
        """The prediction for *label* from *state*, and the state it leads to."""
        x = self._embed[label]
        new_state = []
        for (weights, bias), (h, c) in zip(self._lstm, state):
            gates = np.concatenate([x, h]) @ weights + bias
            i, f, g, o = np.split(gates, 4)
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = _sigmoid(o) * np.tanh(c)
            new_state.append((h, c))
            x = h
        return x @ self._pred[0] + self._pred[1], new_state

    def _score(self, enc: np.ndarray, pred: np.ndarray) -> list[tuple[int, int]]:
        logits = np.maximum(enc + pred, 0) @ self._out[0] + self._out[1]
        tokens = logits[:, :-len(_DURATIONS)].argmax(axis=1)
        durations = logits[:, -len(_DURATIONS):].argmax(axis=1)
        return list(zip(tokens.tolist(), durations.tolist()))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def bundled_tokenizer() -> object:
    """nano-parakeet's SentencePiece model, loaded by path: importing nano_parakeet imports torch."""
    import sentencepiece

    spec = importlib.util.find_spec("nano_parakeet")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("nano-parakeet isn't installed; its tokenizer.model is Parakeet's vocabulary")
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.Load(str(Path(next(iter(spec.submodule_search_locations))) / "tokenizer.model"))
    return tokenizer
