"""Clean-room cache-aware streaming Parakeet model (FastConformer + RNN-T).

Milestone A target: an OFFLINE (full-context) forward + RNN-T greedy decode
that loads the streaming checkpoint's weights and transcribes correctly. The
cache-aware streaming loop (limited-context attention + per-layer caches) is
layered on in Milestone B; the modules here are written so that step only adds
state threading, not new weights.

Reuses nano_parakeet's proven blocks (preprocessor, relative-position
encoding/attention, macaron feed-forward, RNN-T prediction net) and swaps in
the three streaming-model differences:
  * causal dw-striding subsampling (freq 128->65->33->17 => 256*17=4352 proj),
  * LayerNorm + causal depthwise conv module (NeMo conv_norm_type=layer_norm),
  * an RNN-T joint with vocab 1024 (+blank), no TDT durations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano_parakeet.model import (
    ConformerFeedForward,
    RelPositionalEncoding,
    RelPositionMHA,
    RNNTDecoder,
)


class StreamingLogMelPreprocessor(nn.Module):
    """Log-mel features WITHOUT per-utterance normalization.

    The streaming model's preprocessor config is ``normalize: NA`` - it was
    trained on raw log-mel, not the per-utterance mean/std-normalized features
    nano_parakeet's offline preprocessor produces. Skipping normalization both
    matches training (better accuracy) and makes the features fully causal, so
    they can be computed chunk-by-chunk for live streaming. STFT/mel params and
    the pre-emphasis match NeMo's AudioToMelSpectrogramPreprocessor; ``window``
    and ``fb`` are loaded from the checkpoint.
    """

    _PREEMPH = 0.97

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 128,
        log_zero_guard: float = 5.960464e-8,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.log_zero_guard = log_zero_guard
        self.register_buffer("window", torch.zeros(win_length))
        self.register_buffer("fb", torch.zeros(1, n_mels, n_fft // 2 + 1))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(0)  # [1, T]
        x = torch.cat([x[:, :1], x[:, 1:] - self._PREEMPH * x[:, :-1]], dim=1)
        with torch.amp.autocast(x.device.type, enabled=False):
            stft = torch.stft(
                x.float(), n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length, window=self.window.float(),
                center=True, pad_mode="reflect", return_complex=True,
            )
        power = torch.view_as_real(stft).pow(2).sum(-1)
        mel = torch.matmul(self.fb.to(power.dtype), power)
        return torch.log(mel + self.log_zero_guard)  # [1, n_mels, T'] - no norm

N_MELS = 128
D_MODEL = 1024
N_LAYERS = 24
N_HEADS = 8
D_FF = 4096
CONV_KERNEL = 9
SUBSAMPLING_CHANNELS = 256
SUBSAMPLED_FREQ = 17  # 128 -> 65 -> 33 -> 17 under causal (left=2,right=1) padding
VOCAB_SIZE = 1024
BLANK_ID = VOCAB_SIZE  # 1024
PRED_HIDDEN = 640
JOINT_HIDDEN = 640
MAX_SYMBOLS_PER_STEP = 10  # RNN-T greedy: cap non-blank emissions per encoder frame
# chunked_limited attention context in subsampled (80ms) frames. The model was
# trained with these bounds; running full-context attention produces garbage on
# audio longer than the left window. [70,13] is the highest-accuracy preset.
ATT_LEFT = 70
ATT_RIGHT = 13
_SUBSAMPLE_STEPS = 3
# Causal subsampling padding per stride-2 conv: left = kernel-1, right = stride-1.
_SS_PAD_LEFT = 2
_SS_PAD_RIGHT = 1


class CausalConvSubsampling(nn.Module):
    """8x dw-striding subsampling with causal (left=2, right=1) padding.

    Each stride-2 conv reduces a length L to floor(L/2)+1 (NeMo ``calc_length``
    with add_pad = (left+right) - kernel = 0), so 128 mel bins become
    65 -> 33 -> 17. The flattened 256*17 = 4352 matches the streaming
    checkpoint's ``pre_encode.out`` projection (the offline model's symmetric
    padding yields 16 / 4096 instead).
    """

    def __init__(
        self, d_model: int = D_MODEL, channels: int = SUBSAMPLING_CHANNELS,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=0),               # 0
            nn.ReLU(inplace=True),                                         # 1
            nn.Conv2d(channels, channels, 3, stride=2, padding=0, groups=channels),  # 2
            nn.Conv2d(channels, channels, 1),                             # 3
            nn.ReLU(inplace=True),                                         # 4
            nn.Conv2d(channels, channels, 3, stride=2, padding=0, groups=channels),  # 5
            nn.Conv2d(channels, channels, 1),                             # 6
            nn.ReLU(inplace=True),                                         # 7
        )
        self.out = nn.Linear(channels * SUBSAMPLED_FREQ, d_model)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, n_mels] -> [B, 1, T, F]
        x = x.unsqueeze(1)
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d) and layer.stride == (2, 2):
                # F.pad order for [B,C,T,F] is (F_left, F_right, T_left, T_right).
                x = F.pad(x, (_SS_PAD_LEFT, _SS_PAD_RIGHT, _SS_PAD_LEFT, _SS_PAD_RIGHT))
            x = layer(x)
        b, c, t, f = x.shape
        x = self.out(x.transpose(1, 2).reshape(b, t, c * f))
        for _ in range(_SUBSAMPLE_STEPS):
            lengths = torch.div(lengths, 2, rounding_mode="floor") + 1
        return x, lengths.to(torch.long)


class LayerNormCausalConvModule(nn.Module):
    """Conformer conv sub-layer: PW -> GLU -> causal DW -> LayerNorm -> SiLU -> PW.

    Matches NeMo ``conv_norm_type=layer_norm`` + ``conv_context_size=causal``.
    The norm is named ``batch_norm`` in the checkpoint but is a LayerNorm
    (weight+bias only, no running stats); the loader remaps the name.
    """

    def __init__(self, d_model: int = D_MODEL, kernel: int = CONV_KERNEL) -> None:
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel, padding=0, groups=d_model, bias=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.act = nn.SiLU()
        self._left_pad = kernel - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)            # [B, C, T]
        x = self.pointwise_conv1(x)      # [B, 2C, T]
        x = F.glu(x, dim=1)              # [B, C, T]
        x = F.pad(x, (self._left_pad, 0))
        x = self.depthwise_conv(x)       # [B, C, T]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.pointwise_conv2(x)      # [B, C, T]
        return x.transpose(1, 2)         # [B, T, C]


class StreamingFastConformerLayer(nn.Module):
    """FastConformer block (FF1 -> Self-Attn -> Conv -> FF2 -> LayerNorm).

    Identical to nano_parakeet's layer except the conv sub-layer uses the
    LayerNorm causal module above.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(D_MODEL)
        self.feed_forward1 = ConformerFeedForward(D_MODEL, D_FF)
        self.norm_conv = nn.LayerNorm(D_MODEL)
        self.conv = LayerNormCausalConvModule(D_MODEL, CONV_KERNEL)
        self.norm_self_att = nn.LayerNorm(D_MODEL)
        self.self_attn = RelPositionMHA(D_MODEL, N_HEADS)
        self.norm_feed_forward2 = nn.LayerNorm(D_MODEL)
        self.feed_forward2 = ConformerFeedForward(D_MODEL, D_FF)
        self.norm_out = nn.LayerNorm(D_MODEL)

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


def chunked_limited_mask(
    n_frames: int, left: int, chunk_size: int, device: torch.device,
) -> torch.Tensor:
    """NeMo ``chunked_limited`` attention mask [1, T, T]; True = ignore.

    Frames are grouped into non-overlapping chunks of ``chunk_size`` (= right
    context + 1). Every frame in a chunk attends to the whole chunk (its
    ``right`` lookahead) plus ``left`` frames before the chunk start, so all
    frames in a chunk share one context window - matching how the model was
    trained (and how the cache-aware streaming pass will batch frames).
    """
    idx = torch.arange(n_frames, device=device)
    chunk = torch.div(idx, chunk_size, rounding_mode="floor")
    right_bound = (chunk + 1) * chunk_size - 1   # last frame of the query's chunk
    left_bound = chunk * chunk_size - left        # left frames before chunk start
    j = idx.unsqueeze(0)                          # keys  [1, T]
    rb = right_bound.unsqueeze(1)                 # [T, 1]
    lb = left_bound.unsqueeze(1)                  # [T, 1]
    return ((j < lb) | (j > rb)).unsqueeze(0)


class StreamingEncoder(nn.Module):
    """24-layer FastConformer encoder with chunked-limited attention."""

    def __init__(self, att_left: int = ATT_LEFT, att_right: int = ATT_RIGHT) -> None:
        super().__init__()
        self.pre_encode = CausalConvSubsampling()
        self.pos_enc = RelPositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList(
            [StreamingFastConformerLayer() for _ in range(N_LAYERS)]
        )
        self._att_left = att_left
        self._chunk_size = att_right + 1

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = features.transpose(1, 2)  # [B, n_mels, T] -> [B, T, n_mels]
        x, lengths = self.pre_encode(x, lengths)
        x, pos_emb = self.pos_enc(x)
        mask = chunked_limited_mask(
            x.size(1), self._att_left, self._chunk_size, x.device,
        )
        for layer in self.layers:
            x = layer(x, pos_emb, mask=mask)
        return x, lengths


class RNNTJoint(nn.Module):
    """RNN-T joint: enc + pred projections -> ReLU -> Linear(vocab+blank).

    No TDT duration outputs (the streaming model is plain RNN-T, 1025 classes).
    """

    def __init__(
        self,
        enc_dim: int = D_MODEL,
        pred_dim: int = PRED_HIDDEN,
        joint_dim: int = JOINT_HIDDEN,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.enc = nn.Linear(enc_dim, joint_dim)
        self.pred = nn.Linear(pred_dim, joint_dim)
        self.joint_net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),  # no-op in eval
            nn.Linear(joint_dim, vocab_size + 1),
        )

    def forward(
        self, enc_out: torch.Tensor, pred_out: torch.Tensor,
    ) -> torch.Tensor:
        return self.joint_net(self.enc(enc_out) + self.pred(pred_out)).unsqueeze(2)


@torch.inference_mode()
def rnnt_greedy_decode(
    encoder_out: torch.Tensor,
    enc_len: int,
    decoder: RNNTDecoder,
    joint: RNNTJoint,
    blank_id: int = BLANK_ID,
    max_symbols_per_step: int = MAX_SYMBOLS_PER_STEP,
) -> list[int]:
    """Plain RNN-T greedy: per encoder frame, emit non-blanks until blank."""
    device = encoder_out.device
    tokens: list[int] = []
    hidden = None
    last_label = torch.tensor([[blank_id]], dtype=torch.long, device=device)
    for time_idx in range(enc_len):
        f = encoder_out[time_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, enc_dim]
        emitted = 0
        while emitted < max_symbols_per_step:
            pred_out, hidden_next = decoder(last_label, hidden)
            logits = joint(f, pred_out)[0, 0, 0, :]
            k = int(logits.argmax())
            if k == blank_id:
                break
            tokens.append(k)
            hidden = hidden_next
            last_label = torch.tensor([[k]], dtype=torch.long, device=device)
            emitted += 1
    return tokens


class CacheAwareStreamingParakeet(nn.Module):
    """Full model: preprocessor + encoder + RNN-T decoder/joint.

    ``sp`` (SentencePieceProcessor) is attached by the loader.
    """

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor = StreamingLogMelPreprocessor()
        self.encoder = StreamingEncoder()
        self.decoder = RNNTDecoder(
            vocab_size=VOCAB_SIZE, d_model=PRED_HIDDEN, n_layers=2,
        )
        self.joint = RNNTJoint()
        self.sp: object | None = None

    @torch.inference_mode()
    def transcribe_offline(self, audio: torch.Tensor) -> str:
        if self.sp is None:
            raise RuntimeError("No tokenizer loaded; use the loader.")
        token_ids = self.decode_offline(audio)
        return self.sp.DecodeIds(token_ids).strip()

    @torch.inference_mode()
    def decode_offline(self, audio: torch.Tensor) -> list[int]:
        enc_out, enc_len = self.encode(audio)
        return rnnt_greedy_decode(enc_out, enc_len, self.decoder, self.joint)

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Run preprocessor + encoder on one audio window. Returns (enc_out[T,d], T)."""
        device = next(self.parameters()).device
        features = self.preprocessor(audio.to(device))  # [1, n_mels, T]
        lengths = torch.tensor([features.shape[-1]], device=device, dtype=torch.long)
        enc_dtype = next(self.encoder.parameters()).dtype
        enc_out, enc_lengths = self.encoder(features.to(enc_dtype), lengths)
        return enc_out[0].float(), int(enc_lengths[0])
