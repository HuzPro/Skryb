"""A Whisper large-v3 turbo model with its encoder on an Intel NPU and its decoder on the CPU, through OpenVINO.

The encoder is nearly all of the work: a 30 s window is about 1.9 TFLOP, 0.85 s on a Core 5 320's NPU against
several seconds on its CPU. NPUW compiles one of its 32 identical layers and runs it 32 times, which brings the
first compile from about 12 minutes down to under 20 s for 16% slower passes. The decoder is four layers and
runs token by token on OpenVINO's CPU device with int8 weights, about 12 ms a token, against the
cross-attention keys and values the encoder graph already projected.

Both graphs are built from ``model.safetensors`` without torch (``openvino_whisper_graph``) the first time,
into ``snapshots/<sha>/.openvino``.
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.openvino_graph import Safetensors, widen_bf16
from skryb.transcription.openvino_parakeet_adapter import import_openvino
from skryb.transcription.openvino_whisper_graph import DECODER_FILENAME, EMBEDDING, ENCODER_FILENAME, build
from skryb.transcription.whisper_features import log_mel_spectrogram

_logger = logging.getLogger(__name__)

EXPORT_DIRECTORY_NAME = ".openvino"
CACHE_DIRECTORY_NAME = "compiled-npu"
_SAMPLE_RATE = 16_000
_NPU_ENCODER_CONFIG = {"NPU_USE_NPUW": "YES", "NPUW_FOLD": "YES"}
# OpenVINO picks the 2 performance cores of a Core 5 320; all 6 cut a decoder step from 16 to 12.5 ms.
_CPU_DECODER_CONFIG = {"INFERENCE_NUM_THREADS": os.cpu_count() or 1, "SCHEDULING_CORE_TYPE": "ANY_CORE"}


class OpenVinoWhisper:
    """Greedy decoding of one 30 s window from a prompt, the way transformers' ``generate`` does it."""

    def __init__(self, snapshot: Path) -> None:
        self._snapshot = snapshot
        self._tensor: type = object
        self._encoder: object = None
        self._decoder: object = None
        self._request: object = None
        self._embedding: np.ndarray | None = None
        self._suppressed: np.ndarray | None = None
        self._eos = 0
        self._positions = 0
        self.encoder_device: str | None = None

    def load(self) -> None:
        started_at = time.monotonic()
        openvino = import_openvino()
        files = self._snapshot / EXPORT_DIRECTORY_NAME
        if not (files / ENCODER_FILENAME).is_file() or not (files / DECODER_FILENAME).is_file():
            _logger.info("Building Whisper's OpenVINO graphs (once per download) in %s", files)
            build(openvino, self._snapshot, files)
        self._tensor = openvino.Tensor
        core = openvino.Core()
        try:
            self._encoder = core.compile_model(
                str(files / ENCODER_FILENAME), "NPU", {"CACHE_DIR": str(files / CACHE_DIRECTORY_NAME), **_NPU_ENCODER_CONFIG},
            )
            self.encoder_device = "NPU"
        except RuntimeError:
            _logger.warning("The NPU refused Whisper's encoder; it runs on the CPU", exc_info=True)
            self._encoder = core.compile_model(str(files / ENCODER_FILENAME), "CPU")
            self.encoder_device = "CPU"
        self._decoder = core.compile_model(str(files / DECODER_FILENAME), "CPU", _CPU_DECODER_CONFIG)
        self._request = self._decoder.create_infer_request()
        weights = Safetensors(self._snapshot / "model.safetensors")
        self._embedding = weights.bf16_rows(EMBEDDING)
        config = json.loads((self._snapshot / "config.json").read_text(encoding="utf-8"))
        self._eos = int(config["eos_token_id"])
        self._positions = int(config["max_target_positions"])
        self._suppressed = self._suppressed_logits(config["vocab_size"])
        _logger.info("[load whisper openvino %s] encoder on %s in %.2fs", self._snapshot.parent.parent.name,
                     self.encoder_device, time.monotonic() - started_at)

    def _suppressed_logits(self, vocab: int) -> np.ndarray:
        generation = json.loads((self._snapshot / "generation_config.json").read_text(encoding="utf-8"))
        bias = np.zeros(vocab, np.float32)
        bias[[int(t) for t in generation.get("suppress_tokens") or [] if 0 <= int(t) < vocab]] = -np.inf
        return bias

    def begin_suppress_tokens(self) -> list[int]:
        generation = json.loads((self._snapshot / "generation_config.json").read_text(encoding="utf-8"))
        return [int(t) for t in generation.get("begin_suppress_tokens") or []]

    def generate(self, samples: np.ndarray, prompt: list[int], begin_suppress: list[int] = ()) -> list[int]:
        """Token ids after *prompt* for the first 30 s of *samples*, up to the end-of-text token."""
        mel = log_mel_spectrogram(AudioSegment(np.ascontiguousarray(samples, dtype=np.float32), _SAMPLE_RATE))
        cross = self._encoder.infer_new_request({"mel": mel[None]})
        request = self._request
        request.reset_state()
        for output, value in cross.items():
            request.set_tensor(output.get_any_name(), self._tensor(np.ascontiguousarray(value)))
        logits = self._step(prompt, 0)
        logits[list(begin_suppress)] = -np.inf
        tokens: list[int] = []
        limit = self._positions - len(prompt) - 1
        while True:
            token = int(np.argmax(logits + self._suppressed))
            if token == self._eos or len(tokens) >= limit:
                return tokens
            tokens.append(token)
            logits = self._step([token], len(prompt) + len(tokens) - 1)

    def _step(self, ids: list[int], start: int) -> np.ndarray:
        embeds = widen_bf16(self._embedding[np.asarray(ids, np.int64)]).reshape(1, len(ids), -1)
        self._request.set_tensor("inputs_embeds", self._tensor(embeds))
        self._request.set_tensor("position_ids", self._tensor(np.arange(start, start + len(ids), dtype=np.int64)[None]))
        self._request.infer()
        return np.array(self._request.get_tensor("logits").data[0, -1], np.float32)

    def dispose(self) -> None:
        self._encoder = self._decoder = self._request = None
        self._embedding = None
