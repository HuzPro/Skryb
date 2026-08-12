"""Regression: the cache-aware session must emit token IDS from encoder frames.

The session had two methods named ``_decode`` - the RNN-T greedy loop
(frames -> token ids) and the SentencePiece call (token ids -> text). The second
shadowed the first, so ``_run_chunk`` handed a multi-element encoder TENSOR to
``sp.DecodeIds`` and crashed with "Boolean value of Tensor ... is ambiguous".

These tests drive the real session through ``feed``/``finalize`` over a fake
model whose encoder has zero layers (chunk frames pass straight to the decoder)
and whose fake SentencePiece refuses anything but ``list[int]`` - exactly the
contract real sentencepiece enforces.
"""
from __future__ import annotations

import numpy as np
import torch

from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.streaming_engine import StreamingUpdate
from voicetotext.transcription.streaming.cache_aware.model import BLANK_ID
from voicetotext.transcription.streaming.cache_aware.streaming_session import (
    CacheAwareStreamingSession,
)

_SAMPLE_RATE = 16000
_ENOUGH_AUDIO_SAMPLES = 24000  # 1.5s -> ~18 subsampled frames -> fills a chunk
_HOP = 160                     # mel hop; T_mel = samples // hop
_SUBSAMPLE_FACTOR = 8          # 8x dw-striding; T_sub = T_mel // 8
_ENC_DIM = 4
_NON_BLANK_TOKEN = 7


class _FakePreprocessor:
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        t_mel = audio.shape[-1] // _HOP
        return torch.zeros(1, 128, t_mel)  # [1, n_mels, T_mel]


class _FakeEncoder:
    def __init__(self) -> None:
        self.layers: list = []          # zero layers: chunk frames pass through
        self.pos_enc = None             # never used while layers is empty
        self._param = torch.zeros(1)

    def parameters(self):
        return iter([self._param])      # supplies the encoder dtype

    def pre_encode(self, feats: torch.Tensor, lengths: torch.Tensor):
        t_sub = feats.shape[1] // _SUBSAMPLE_FACTOR
        return torch.zeros(1, t_sub, _ENC_DIM), lengths


class _FakeDecoder:
    def __call__(self, last_label: torch.Tensor, hidden: object):
        return torch.zeros(1, 1, 1), hidden


class _FakeJoint:
    """Emits one non-blank token then a blank, alternately: one token per frame."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, enc_frame: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        logits = torch.full((1, 1, 1, BLANK_ID + 1), -1.0)
        chosen = _NON_BLANK_TOKEN if self.calls % 2 == 1 else BLANK_ID
        logits[0, 0, 0, chosen] = 1.0
        return logits


class _FakeSentencePiece:
    """Stand-in that enforces real sentencepiece's ``list[int]`` contract."""

    def __init__(self) -> None:
        self.decode_calls: list[list[int]] = []

    def IdToPiece(self, token_id: int) -> str:  # noqa: N802 (sentencepiece name)
        return "▁hello"  # leading word-boundary marker

    def DecodeIds(self, ids: object) -> str:  # noqa: N802 (sentencepiece name)
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            raise TypeError(f"DecodeIds expects list[int], got {type(ids).__name__}")
        self.decode_calls.append(list(ids))
        return " ".join("hello" for _ in ids)


class _FakeStreamingModel:
    def __init__(self) -> None:
        self.preprocessor = _FakePreprocessor()
        self.encoder = _FakeEncoder()
        self.decoder = _FakeDecoder()
        self.joint = _FakeJoint()
        self.sp = _FakeSentencePiece()
        self._param = torch.zeros(1)

    def parameters(self):
        return iter([self._param])      # supplies the session device


def _silence(samples: int = _ENOUGH_AUDIO_SAMPLES) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


class TestCacheAwareSessionDecode:
    def test_feed_runs_a_chunk_without_decoding_a_tensor(self) -> None:
        model = _FakeStreamingModel()
        session = CacheAwareStreamingSession(model)

        update = session.feed(_silence())

        assert isinstance(update, StreamingUpdate)

    def test_finalize_returns_decoded_text_from_emitted_tokens(self) -> None:
        model = _FakeStreamingModel()
        session = CacheAwareStreamingSession(model)

        session.feed(_silence())
        result = session.finalize()

        assert isinstance(result, TranscriptionResult)
        assert "hello" in result.text

    def test_sentencepiece_is_only_ever_handed_token_id_lists(self) -> None:
        model = _FakeStreamingModel()
        session = CacheAwareStreamingSession(model)

        session.feed(_silence())
        session.finalize()

        assert model.sp.decode_calls  # the RNN-T loop actually emitted tokens
        for ids in model.sp.decode_calls:
            assert all(token == _NON_BLANK_TOKEN for token in ids)
