"""True cache-aware streaming over the FastConformer + RNN-T model.

Unlike a sliding-window re-encode (which drifts at window edges), this carries
an EXACT per-layer cache of prior activations:
  * attention: the last ``ATT_LEFT`` frames of each layer's attention input
    (norm_self_att output) serve as extra K/V for the current chunk.
  * conv: the last ``CONV_KERNEL-1`` frames of each layer's conv input
    (norm_conv output) serve as the causal left context for the depthwise conv.
Each frame is encoded exactly once; the chunk's output equals the offline
chunked-attention forward, with no boundary drift.

Milestone B scope: mel + dw-striding subsampling are computed offline-exact on
the supplied audio (both are local convs, no drift); the 24 conformer layers and
the RNN-T decode are streamed with the caches above. Per-chunk audio mel +
subsampling caching is a Milestone-C concern for the live mic path.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from skryb.transcription.result import TranscriptionResult
from skryb.transcription.streaming.streaming_engine import (
    StreamingTranscriptionSession, StreamingUpdate,
)
from skryb.transcription.streaming.cache_aware.model import (
    ATT_LEFT, ATT_RIGHT, BLANK_ID, CONV_KERNEL, MAX_SYMBOLS_PER_STEP,
)

_CONV_LEFT = CONV_KERNEL - 1  # 8
_CHUNK_FRAMES = ATT_RIGHT + 1  # 14; one attention chunk (right context + current)
_SAMPLES_PER_SUB_FRAME = 8 * 160  # 1280 = one encoder frame = 80ms
# Subsampling is a local causal conv (small receptive field). Keeping this many
# encoder frames of left audio context makes the committed frames numerically
# exact; the last frame is held back one step for the conv's right lookahead.
_SUBSAMPLE_LEFT_FRAMES = 32
_SUBSAMPLE_RIGHT_FRAMES = 2
_WORD_BOUNDARY = "▁"  # SentencePiece word-start marker ('▁')


def _conv_chunk(conv_mod, cin_full: torch.Tensor) -> torch.Tensor:
    """Causal conv over [conv_cache(8) ++ chunk]; returns the chunk frames only."""
    x = cin_full.transpose(1, 2)          # [1, d, 8+M]
    x = conv_mod.pointwise_conv1(x)
    x = F.glu(x, dim=1)
    x = conv_mod.depthwise_conv(x)        # kernel CONV_KERNEL, no pad -> [1, d, M]
    x = conv_mod.norm(x.transpose(1, 2)).transpose(1, 2)
    x = conv_mod.act(x)
    x = conv_mod.pointwise_conv2(x)
    return x.transpose(1, 2)              # [1, M, d]


def _layer_chunk(layer, x, att_cache, conv_cache, pos_enc):
    """Stream one conformer layer over a chunk, updating its two caches."""
    m = x.shape[1]
    x = x + 0.5 * layer.feed_forward1(layer.norm_feed_forward1(x))

    a_in = layer.norm_self_att(x)         # [1, M, d]
    kv = a_in if att_cache is None else torch.cat([att_cache, a_in], dim=1)
    _, pos_emb = pos_enc(kv)
    x = x + layer.self_attn(kv, pos_emb, mask=None)[:, -m:, :]
    new_att_cache = kv[:, -ATT_LEFT:, :].clone()

    c_in = layer.norm_conv(x)             # [1, M, d]
    if conv_cache is None:
        conv_cache = torch.zeros(
            1, _CONV_LEFT, c_in.shape[-1], dtype=c_in.dtype, device=c_in.device,
        )
    cin_full = torch.cat([conv_cache, c_in], dim=1)
    x = x + _conv_chunk(layer.conv, cin_full)
    new_conv_cache = cin_full[:, -_CONV_LEFT:, :].clone()

    x = x + 0.5 * layer.feed_forward2(layer.norm_feed_forward2(x))
    return layer.norm_out(x), new_att_cache, new_conv_cache


def words_committed_text(tokens: list[int], sp: object) -> str:
    """Decode only COMPLETE words, holding back the in-progress (last) word.

    The model emits a word as subword tokens across chunks ('▁iss' then 'ue'),
    so a live snapshot must exclude the current word until the next word starts
    (a new '▁' piece) - otherwise the word-count output writer types it
    half-finished ('issue' -> 'iss'). At finalize the caller decodes ALL tokens.
    """
    last_word_start = 0
    for i, tok in enumerate(tokens):
        if sp.IdToPiece(tok).startswith(_WORD_BOUNDARY):
            last_word_start = i
    return sp.DecodeIds(tokens[:last_word_start]).strip()


class CacheAwareStreamingSession(StreamingTranscriptionSession):
    """Live cache-aware streaming over audio chunks (no re-encode, no drift).

    ``feed`` accepts arbitrary-size mic frames; it buffers them, computes mel +
    subsampling on a small bounded window (exact, since both are local convs),
    and streams complete attention chunks (14 frames) through the per-layer
    caches, decoding new frames once with persistent RNN-T state. ``finalize``
    flushes the trailing partial chunk. Output text grows monotonically.
    """

    def __init__(self, model: object) -> None:
        self._model = model
        self._device = next(model.parameters()).device
        self._audio = np.zeros(0, dtype=np.float32)
        self._base_sub = 0          # global subsampled-frame index of audio[0]
        self._committed_sub = 0     # global subsampled frames fed to the layers
        n_layers = len(model.encoder.layers)
        self._att_caches: list = [None] * n_layers
        self._conv_caches: list = [None] * n_layers
        self._hidden = None
        self._last = torch.tensor([[BLANK_ID]], dtype=torch.long, device=self._device)
        self._pending: torch.Tensor | None = None  # stable sub frames awaiting a full chunk
        self._tokens: list[int] = []
        self._started_at = time.monotonic()

    def feed(self, chunk: np.ndarray) -> StreamingUpdate:
        self._audio = np.concatenate([self._audio, chunk.astype(np.float32)])
        self._consume(right_hold=_SUBSAMPLE_RIGHT_FRAMES, flush_chunks=False)
        return self._snapshot(is_final=False)

    def finalize(self) -> TranscriptionResult:
        self._consume(right_hold=0, flush_chunks=True)
        return TranscriptionResult(
            text=self._decode_text(self._tokens), language="en",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - self._started_at,
        )

    @torch.inference_mode()
    def _consume(self, right_hold: int, flush_chunks: bool) -> None:
        sub = self._subsample(self._audio)              # [Tsub, d], frames from base_sub
        target = self._base_sub + sub.shape[0] - right_hold
        if target > self._committed_sub:
            lo = self._committed_sub - self._base_sub
            hi = target - self._base_sub
            self._enqueue(sub[lo:hi])
            self._committed_sub = target
        self._drain_chunks(flush_chunks)
        self._trim_audio()

    def _enqueue(self, frames: torch.Tensor) -> None:
        frames = frames.unsqueeze(0)                    # [1, n, d]
        self._pending = frames if self._pending is None else torch.cat(
            [self._pending, frames], dim=1,
        )

    def _drain_chunks(self, flush: bool) -> None:
        if self._pending is None:
            return
        while self._pending.shape[1] >= _CHUNK_FRAMES:
            self._run_chunk(self._pending[:, :_CHUNK_FRAMES, :])
            self._pending = self._pending[:, _CHUNK_FRAMES:, :]
        if flush and self._pending.shape[1] > 0:
            self._run_chunk(self._pending)
            self._pending = None

    def _run_chunk(self, h: torch.Tensor) -> None:
        for i, layer in enumerate(self._model.encoder.layers):
            h, self._att_caches[i], self._conv_caches[i] = _layer_chunk(
                layer, h, self._att_caches[i], self._conv_caches[i],
                self._model.encoder.pos_enc,
            )
        self._emit_tokens(h[0].float())

    def _emit_tokens(self, enc: torch.Tensor) -> None:
        for tf in range(enc.shape[0]):
            f = enc[tf].unsqueeze(0).unsqueeze(0)
            emitted = 0
            while emitted < MAX_SYMBOLS_PER_STEP:
                pred_out, hidden_next = self._model.decoder(self._last, self._hidden)
                k = int(self._model.joint(f, pred_out)[0, 0, 0, :].argmax())
                if k == BLANK_ID:
                    break
                self._tokens.append(k)
                self._hidden = hidden_next
                self._last = torch.tensor([[k]], dtype=torch.long, device=self._device)
                emitted += 1

    def _subsample(self, audio: np.ndarray) -> torch.Tensor:
        feats = self._model.preprocessor(torch.from_numpy(audio).to(self._device))
        lengths = torch.tensor([feats.shape[-1]], device=self._device, dtype=torch.long)
        enc_dtype = next(self._model.encoder.parameters()).dtype
        sub, _ = self._model.encoder.pre_encode(feats.transpose(1, 2).to(enc_dtype), lengths)
        return sub[0]

    def _trim_audio(self) -> None:
        new_base = max(0, self._committed_sub - _SUBSAMPLE_LEFT_FRAMES)
        trim = (new_base - self._base_sub) * _SAMPLES_PER_SUB_FRAME
        if trim > 0:
            self._audio = self._audio[trim:]
            self._base_sub = new_base

    def _decode_text(self, tokens: list[int]) -> str:
        return self._model.sp.DecodeIds(tokens).strip()

    def _snapshot(self, is_final: bool) -> StreamingUpdate:
        # Live snapshot commits only complete words (hold back the current word).
        return StreamingUpdate(
            committed_text=words_committed_text(self._tokens, self._model.sp),
            tentative_text="", is_final=is_final,
        )
