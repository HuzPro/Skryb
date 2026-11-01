"""Qwen3-ASR-1.7B on an Intel NPU through OpenVINO, with no torch.

The audio encoder runs one 8 s window at a time at fp16 and Qwen3's text decoder runs with int8 weights in
the NPU plugin's own LLM pipeline (NPUW), which keeps the K/V cache on the device. Both graphs are built from
``model.safetensors`` in seconds the first time (``openvino_qwen3_asr_graph``), into
``snapshots/<sha>/.openvino``, and compiled once into OpenVINO's cache beside them.

A decoder pass reads all 1.7 GB of weights whether it checks one token or sixteen, so decoding is
speculative: when another engine has already transcribed the clip (Phonon, for dictation on this machine)
its text is the draft, each pass checks up to ``PASS_WIDTH`` drafted tokens, and the ones the decoder would
have written itself are kept. The text is exactly what greedy decoding gives; the draft only saves passes.
On a Core 5 320 a 10 s clip takes about 0.6 s this way against 2.5 s token by token, and 9 s through
transformers on the CPU.

Without an NPU, or when the NPU refuses the decoder, it all runs on OpenVINO's CPU device and the engine says
so in ``device_note``.
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.hf_model_manager import loadable_snapshot_dir
from skryb.transcription.openvino_parakeet_adapter import import_openvino, split_at_pauses
from skryb.transcription.openvino_graph import Safetensors, widen_bf16
from skryb.transcription.openvino_qwen3_asr_graph import (
    DECODER_FILENAME, EMBEDDING, ENCODER_FILENAME, WINDOW_FRAMES, build, window_tokens,
)
from skryb.transcription.qwen3_asr_adapter import new_token_budget
from skryb.transcription.qwen3_asr_prompt import LANGUAGE_NAMES, prompt_text
from skryb.transcription.result import TranscriptionResult
from skryb.transcription.roman_urdu import romanize
from skryb.transcription.speculative_draft import Draft
from skryb.transcription.whisper_features import log_mel_features

_logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_HOP_SAMPLES = 160
# Qwen3-ASR's feature extractor pads anything shorter than half a second with silence.
_SHORTEST_SAMPLES = 8_000
EXPORT_DIRECTORY_NAME = ".openvino"
CACHE_DIRECTORY_NAME = "compiled-npu"
# Tokens checked per decoder pass. On a Core 5 320 a pass costs about 55 ms at 1 or 16 tokens; 32 is refused.
PASS_WIDTH = 16
# The prompt is prefilled 128 tokens at a time: a 3 s clip's 70-token prompt takes 0.14 s against 0.23 s at
# 256. Past three chunks NPUW's default "pyramid" attention gives wrong logits, so attention stays static.
PREFILL_CHUNK = 128
MAX_PROMPT_TOKENS = 1024
MAX_RESPONSE_TOKENS = 512
# 13 audio tokens a second: 45 s leaves room in the prompt for names to spell and the drafted tokens.
PIECE_SECONDS = 45
_NPU_DECODER_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_MAX_PROMPT_LEN": MAX_PROMPT_TOKENS,
    "NPUW_LLM_MIN_RESPONSE_LEN": MAX_RESPONSE_TOKENS,
    "NPUW_LLM_MAX_GENERATION_TOKEN_LEN": PASS_WIDTH,
    "NPUW_LLM_PREFILL_CHUNK_SIZE": PREFILL_CHUNK,
    "NPUW_LLM_PREFILL_ATTENTION_HINT": "STATIC",
}
_ASR_TEXT_MARK = "<asr_text>"


class OpenVinoQwen3AsrAdapter(TranscriptionEngine):
    """Batch and meeting Qwen3-ASR. One call at a time, like every adapter here.

    *drafter*, when set, transcribes the same clip with another engine; its text drafts English decodes.
    """

    def __init__(self, model_dir: Path, repo_id: str, revision: str = "", language: str = "en") -> None:
        self._model_dir = model_dir
        self._repo_id = repo_id
        self._revision = revision
        self._language = language if language in LANGUAGE_NAMES else "en"
        self.drafter: Callable[[AudioSegment], str] | None = None
        self._encoder: object = None
        self._decoder: _Decoder | None = None
        self._embedding: np.ndarray | None = None
        self._tokenizer: object = None
        self._eos: tuple[int, ...] = ()
        self._device: str | None = None
        self._note: str | None = None
        self._lock = threading.Lock()

    @property
    def encoder_device(self) -> str | None:
        return self._device

    @property
    def device_note(self) -> str | None:
        return self._note

    def load_model(self) -> None:
        started_at = time.monotonic()
        openvino = import_openvino()
        snapshot = loadable_snapshot_dir(self._model_dir, self._repo_id, self._revision)
        files = snapshot / EXPORT_DIRECTORY_NAME
        if not (files / DECODER_FILENAME).is_file() or not (files / ENCODER_FILENAME).is_file():
            _logger.info("Building Qwen3-ASR's OpenVINO graphs (once per download) in %s", files)
            build_started = time.monotonic()
            build(openvino, snapshot, files)
            _logger.info("Built in %.1fs", time.monotonic() - build_started)
        core = openvino.Core()
        cache = {"CACHE_DIR": str(files / CACHE_DIRECTORY_NAME)}
        with ThreadPoolExecutor(2) as pool:
            encoder = pool.submit(_compile_encoder, core, files / ENCODER_FILENAME, cache)
            decoder = pool.submit(_compile_decoder, core, files / DECODER_FILENAME, cache)
            (self._encoder, encoder_device), (compiled, self._device, self._note) = encoder.result(), decoder.result()
        if encoder_device != self._device:
            self._note = self._note or f"Qwen3-ASR's audio encoder runs on the {encoder_device}."
        self._decoder = _Decoder(compiled, on_npu=self._device == "NPU")
        self._embedding = Safetensors(snapshot / "model.safetensors").bf16_rows(EMBEDDING)
        self._tokenizer, self._eos = _tokenizer(snapshot)
        _logger.info("[load_model qwen3-asr openvino] on %s in %.2fs", self._device, time.monotonic() - started_at)

    def is_model_loaded(self) -> bool:
        return self._decoder is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        return self.transcribe_prompted(audio, None, self._language)

    def transcribe_prompted(
        self, audio: AudioSegment, context: str | None, language: str | None,
    ) -> TranscriptionResult:
        if self._decoder is None:
            raise TranscriptionError("Model not loaded")
        if audio.sample_rate != _SAMPLE_RATE:
            raise TranscriptionError(f"Qwen3-ASR needs {_SAMPLE_RATE} Hz audio, got {audio.sample_rate}")
        started_at = time.monotonic()
        samples = np.ascontiguousarray(audio.data, dtype=np.float32)
        with self._lock:
            pieces = split_at_pauses(samples, PIECE_SECONDS * _SAMPLE_RATE) or [samples]
            text = " ".join(part for part in (self._transcribe_piece(p, context, language) for p in pieces) if part)
        return TranscriptionResult(
            # Qwen3-ASR has no Urdu: it writes spoken Urdu in Hindi script.
            text=romanize(text),
            language=language or "",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - started_at,
        )

    def _transcribe_piece(self, samples: np.ndarray, context: str | None, language: str | None) -> str:
        prefix, suffix = (self._ids(text) for text in prompt_text(context, language))
        prompt = np.concatenate([self._embeds(prefix), self._audio(samples), self._embeds(suffix)])
        draft = self._draft(samples, language)
        budget = min(new_token_budget(samples.size / _SAMPLE_RATE), MAX_RESPONSE_TOKENS - PASS_WIDTH)
        tokens = self._decoder.decode(prompt, draft, self._embeds, budget, self._eos)
        text = self._tokenizer.decode(tokens, skip_special_tokens=False)
        return text.split(_ASR_TEXT_MARK)[-1].strip()

    def _audio(self, samples: np.ndarray) -> np.ndarray:
        samples = np.pad(samples, (0, max(0, _SHORTEST_SAMPLES - samples.size)))
        mel = log_mel_features(samples)
        out = []
        for start in range(0, mel.shape[1], WINDOW_FRAMES):
            window = mel[:, start:start + WINDOW_FRAMES]
            real = window.shape[1]
            window = np.pad(window, ((0, 0), (0, WINDOW_FRAMES - real)))
            valid = window_tokens(real)
            result = self._encoder.infer_new_request({"mel": window[None], "valid": np.array([valid], np.int64)})
            out.append(np.asarray(next(iter(result.values())))[:valid])
        return np.concatenate(out)

    def _draft(self, samples: np.ndarray, language: str | None) -> list[int]:
        """Another engine's transcript of the clip as token ids; nothing for other languages or without one."""
        if self.drafter is None or language != "en":
            return []
        try:
            text = self.drafter(AudioSegment(samples, _SAMPLE_RATE))
        except Exception:
            _logger.warning("The draft transcript failed; decoding without one", exc_info=True)
            return []
        return self._ids(text.strip()) if text.strip() else []

    def _ids(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def _embeds(self, ids: list[int]) -> np.ndarray:
        return widen_bf16(self._embedding[np.asarray(ids, np.int64)]).reshape(len(ids), -1)

    def dispose(self) -> None:
        with self._lock:
            self._encoder = None
            self._decoder = None
            self._embedding = None


class _Decoder:
    """Greedy decoding through the stateful decoder, checking drafted tokens a pass at a time.

    On the NPU, NPUW takes the position of the first token fed as where the cache now ends, so dropping
    rejected tokens is free; OpenVINO's CPU device appends whatever it is fed, so its state is cut back.
    """

    def __init__(self, compiled: object, on_npu: bool) -> None:
        self._compiled = compiled
        self._request = compiled.create_infer_request()
        self._on_npu = on_npu

    def decode(
        self, prompt: np.ndarray, draft_ids: list[int], embeds: Callable[[list[int]], np.ndarray], budget: int,
        eos: tuple[int, ...],
    ) -> list[int]:
        draft = Draft(draft_ids + [eos[-1]] if draft_ids else [])
        # The draft starts after the prompt: NPUW fails when a rejected draft is cut back out of the prompt's chunk.
        self._request.reset_state()
        rows = self._run(prompt, 0)[-1:]
        kept, proposal, tokens = len(prompt), [], []
        while True:
            accepted = 0
            while accepted < len(proposal) and int(rows[accepted].argmax()) == proposal[accepted]:
                accepted += 1
            new = proposal[:accepted] + [int(rows[accepted].argmax())]
            ended = next((i for i, token in enumerate(new) if token in eos), None)
            if ended is not None:
                return tokens + new[:ended]
            tokens += new
            if len(tokens) >= budget:
                return tokens[:budget]
            draft.advance(new)
            kept += accepted
            proposal = draft.propose(PASS_WIDTH - 1)
            feed = [new[-1]] + proposal
            rows = self._run(embeds(feed), kept)[-len(feed):]
            kept += 1

    def _run(self, embeds: np.ndarray, start: int) -> np.ndarray:
        if not self._on_npu and start:
            self._cut_cache(start)
        end = start + len(embeds)
        result = self._request.infer({
            "inputs_embeds": embeds[None], "attention_mask": np.ones((1, end), np.int64),
            "position_ids": np.arange(start, end, dtype=np.int64)[None], "beam_idx": np.zeros(1, np.int32),
        })
        return result[self._compiled.output("logits")][0]

    def _cut_cache(self, length: int) -> None:
        for state in self._request.query_state():
            cache = state.state.data
            if cache.shape[2] > length:
                state.state = type(state.state)(np.ascontiguousarray(cache[:, :, :length]))


def _compile_encoder(core: object, path: Path, cache: dict) -> tuple[object, str]:
    try:
        return core.compile_model(str(path), "NPU", cache), "NPU"
    except RuntimeError:
        _logger.warning("The NPU refused Qwen3-ASR's audio encoder; it runs on the CPU", exc_info=True)
        return core.compile_model(str(path), "CPU"), "CPU"


def _compile_decoder(core: object, path: Path, cache: dict) -> tuple[object, str, str | None]:
    try:
        return core.compile_model(str(path), "NPU", {**cache, **_NPU_DECODER_CONFIG}), "NPU", None
    except RuntimeError as error:
        _logger.warning("The NPU refused Qwen3-ASR's decoder; it runs on the CPU", exc_info=True)
        return core.compile_model(str(path), "CPU"), "CPU", f"Qwen3-ASR runs on the CPU: the NPU refused it ({error})."[:300]


def _tokenizer(snapshot: Path) -> tuple[object, tuple[int, ...]]:
    import json

    from tokenizers import Tokenizer

    generation = json.loads((snapshot / "generation_config.json").read_text(encoding="utf-8"))
    eos = generation["eos_token_id"]
    return Tokenizer.from_file(str(snapshot / "tokenizer.json")), tuple(eos if isinstance(eos, list) else [eos])
