"""Unit specs for the cache-aware model's tensor logic.

These run on CPU with random init and tiny inputs — no real checkpoint, no GPU,
no vendor-specific runtime. They pin the dimension/length arithmetic that is
otherwise only exercised when the real 2.4GB model is loaded on a GPU: the
8x causal subsampling, the chunked-limited attention mask, the RNN-T joint's
vocab width, and the greedy-decode emission loop.
"""
from __future__ import annotations

import torch

from voicetotext.transcription.streaming.cache_aware.model import (
    BLANK_ID,
    D_MODEL,
    MAX_SYMBOLS_PER_STEP,
    N_MELS,
    PRED_HIDDEN,
    VOCAB_SIZE,
    CausalConvSubsampling,
    RNNTJoint,
    StreamingLogMelPreprocessor,
    chunked_limited_mask,
    rnnt_greedy_decode,
)

_NON_BLANK_TOKEN = 7


class TestCausalConvSubsampling:
    """8x dw-striding: each stride-2 conv takes L -> floor(L/2)+1 in time."""

    def test_collapses_128_frames_to_17_with_model_dim_output(self) -> None:
        subsampling = CausalConvSubsampling()
        features = torch.randn(1, 128, N_MELS)  # [B, T, n_mels]

        out, lengths = subsampling(features, torch.tensor([128]))

        assert out.shape == (1, 17, D_MODEL)
        assert lengths.tolist() == [17]

    def test_time_and_freq_subsample_independently(self) -> None:
        # 64 time frames: 64 -> 33 -> 17 -> 9, while the 128 freq bins still
        # collapse to 17 internally. A T/F mix-up would change the frame count.
        subsampling = CausalConvSubsampling()
        features = torch.randn(1, 64, N_MELS)

        out, lengths = subsampling(features, torch.tensor([64]))

        assert out.shape == (1, 9, D_MODEL)
        assert lengths.tolist() == [9]


class TestChunkedLimitedMask:
    def test_groups_frames_into_chunks_with_left_context(self) -> None:
        # 6 frames, left=2, chunk_size=2 -> chunks [0,0,1,1,2,2]. Each frame
        # attends to its whole chunk plus 2 frames before the chunk start.
        ignore = True
        attend = False
        expected = torch.tensor([[
            [attend, attend, ignore, ignore, ignore, ignore],
            [attend, attend, ignore, ignore, ignore, ignore],
            [attend, attend, attend, attend, ignore, ignore],
            [attend, attend, attend, attend, ignore, ignore],
            [ignore, ignore, attend, attend, attend, attend],
            [ignore, ignore, attend, attend, attend, attend],
        ]])

        mask = chunked_limited_mask(6, left=2, chunk_size=2, device=torch.device("cpu"))

        assert torch.equal(mask, expected)

    def test_every_frame_attends_to_itself(self) -> None:
        mask = chunked_limited_mask(7, left=3, chunk_size=2, device=torch.device("cpu"))

        diagonal = torch.diagonal(mask[0])

        assert not diagonal.any()  # the self-position is never masked out


class TestRNNTJoint:
    def test_emits_one_logit_per_vocab_token_plus_blank(self) -> None:
        joint = RNNTJoint()
        enc_frame = torch.randn(1, 1, D_MODEL)
        pred_out = torch.randn(1, 1, PRED_HIDDEN)

        logits = joint(enc_frame, pred_out)

        assert logits.shape == (1, 1, 1, VOCAB_SIZE + 1)
        assert logits[0, 0, 0, :].numel() == VOCAB_SIZE + 1


class _PassthroughDecoder:
    """Stand-in RNN-T prediction net: ignores input, threads hidden state."""

    def __call__(self, last_label: torch.Tensor, hidden: object) -> tuple[torch.Tensor, object]:
        return torch.zeros(1, 1, 1), hidden


class _AlternatingJoint:
    """Emits one non-blank then a blank, alternately — one token per frame."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, enc_frame: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        logits = torch.full((1, 1, 1, BLANK_ID + 1), -1.0)
        chosen = _NON_BLANK_TOKEN if self.calls % 2 == 1 else BLANK_ID
        logits[0, 0, 0, chosen] = 1.0
        return logits


class _AlwaysNonBlankJoint:
    """Never emits blank — forces the per-frame emission cap to engage."""

    def __call__(self, enc_frame: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, 1, 1, BLANK_ID + 1), -1.0)
        logits[0, 0, 0, _NON_BLANK_TOKEN] = 1.0
        return logits


class TestRNNTGreedyDecode:
    def test_emits_one_token_per_frame_until_blank(self) -> None:
        encoder_out = torch.zeros(3, 4)  # 3 frames, arbitrary enc dim

        tokens = rnnt_greedy_decode(
            encoder_out, enc_len=3, decoder=_PassthroughDecoder(), joint=_AlternatingJoint(),
        )

        assert tokens == [_NON_BLANK_TOKEN, _NON_BLANK_TOKEN, _NON_BLANK_TOKEN]

    def test_caps_emissions_per_frame_at_max_symbols(self) -> None:
        encoder_out = torch.zeros(1, 4)  # a single frame

        tokens = rnnt_greedy_decode(
            encoder_out, enc_len=1, decoder=_PassthroughDecoder(), joint=_AlwaysNonBlankJoint(),
        )

        assert len(tokens) == MAX_SYMBOLS_PER_STEP


class TestStreamingLogMelPreprocessor:
    def test_produces_128_mel_bins_over_time(self) -> None:
        preprocessor = StreamingLogMelPreprocessor()
        one_second = torch.zeros(16000)

        mel = preprocessor(one_second)

        assert mel.ndim == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == N_MELS
        assert mel.shape[2] > 0
