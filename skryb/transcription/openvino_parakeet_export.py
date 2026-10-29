"""The one-time export behind the NPU engine: the encoder to OpenVINO IR and the decoder to NumPy.

The only part of the engine that needs torch. It runs once per download, when ``.openvino`` is missing
or incomplete; every later load reads the IR and ``decoder.npz`` without importing torch.
"""

import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from skryb.transcription.parakeet_hf_checkpoint import remap_hf_parakeet_state_dict
from skryb.transcription.parakeet_numpy import DECODING_TENSORS

_logger = logging.getLogger(__name__)

_N_MELS = 128
_EXAMPLE_FRAMES = 201
_EXAMPLE_LENGTH = 180


def export(openvino: object, weights: Path, encoder_path: Path, decoder_path: Path) -> None:
    """Write whichever of the encoder IR and the decoder tensors is missing."""
    state = _remapped_state_dict(weights, with_encoder=not encoder_path.is_file())
    if not encoder_path.is_file():
        export_encoder(openvino, state, encoder_path)
    if not decoder_path.is_file():
        export_decoder(state, decoder_path)


def export_decoder(state: dict, path: Path) -> None:
    """The prediction net and joint as fp32 NumPy arrays, written atomically."""
    partial = path.with_name(path.name + ".partial.npz")
    np.savez(partial, **{name: state[name].numpy() for name in DECODING_TENSORS})
    os.replace(partial, path)


def _remapped_state_dict(weights: Path, with_encoder: bool) -> dict[str, torch.Tensor]:
    """nano-parakeet's parameters in fp32; the encoder's only when it is to be exported."""
    from safetensors.torch import load_file

    state = remap_hf_parakeet_state_dict(load_file(str(weights)))
    return {
        name: tensor.float() if tensor.is_floating_point() else tensor
        for name, tensor in state.items()
        if with_encoder or not name.startswith("encoder.")
    }


def _part(state: dict[str, object], prefix: str) -> dict[str, object]:
    return {name[len(prefix):]: tensor for name, tensor in state.items() if name.startswith(prefix)}


def export_encoder(openvino: object, state: dict, ir_path: Path) -> None:
    """Write the encoder as OpenVINO IR at *ir_path*, for any input length.

    Staged in a sibling directory and moved into place last, so an
    interrupted export never leaves IR that looks finished.
    """
    from nano_parakeet import model as parakeet

    started_at = time.monotonic()
    encoder = parakeet.FastConformerEncoder()
    encoder.load_state_dict(_part(state, "encoder."))
    wrapper = MaskedEncoder(encoder).eval()
    example = (torch.zeros(1, _N_MELS, _EXAMPLE_FRAMES), torch.tensor([_EXAMPLE_LENGTH]))
    with torch.no_grad():
        converted = openvino.convert_model(
            wrapper, example_input=example,
            input=[("features", [1, _N_MELS, -1]), ("length", [1])],
        )
    staging = ir_path.parent.with_name(ir_path.parent.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    openvino.save_model(converted, str(staging / ir_path.name))
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    # The .xml, whose presence means "exported", goes last.
    for exported in sorted(staging.iterdir(), key=lambda path: path.name == ir_path.name):
        os.replace(exported, ir_path.parent / exported.name)
    staging.rmdir()
    _logger.info("Exported the encoder to OpenVINO IR in %.1fs", time.monotonic() - started_at)


class MaskedEncoder(torch.nn.Module):
    """nano-parakeet's encoder, masking padded frames the way NeMo does.

    nano-parakeet runs one unpadded clip at a time and never masks. In a
    clip padded to a fixed length, every convolution must see zeros past
    the end, as it would at the edge of an unpadded clip, and attention
    must skip the padded keys; otherwise the padding leaks into the last
    real frames.
    """

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self, features: torch.Tensor, length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, length = self._subsample(features.transpose(1, 2), length)
        x, pos_emb = self.encoder.pos_enc(x)
        frames = x.shape[1]
        valid = torch.arange(frames).unsqueeze(0) < length.unsqueeze(1)
        padded_keys = (~valid).unsqueeze(1).expand(-1, frames, frames)
        keep = valid.unsqueeze(-1).to(x.dtype)
        for layer in self.encoder.layers:
            x = x + 0.5 * layer.feed_forward1(layer.norm_feed_forward1(x))
            x = x + layer.self_attn(layer.norm_self_att(x), pos_emb, padded_keys)
            x = x + layer.conv(layer.norm_conv(x) * keep)
            x = x + 0.5 * layer.feed_forward2(layer.norm_feed_forward2(x))
            x = layer.norm_out(x)
        return x, length

    def _subsample(
        self, x: torch.Tensor, length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``ConvSubsampling.forward``, with frames past the end zeroed after each step."""
        pre_encode = self.encoder.pre_encode
        x = x.unsqueeze(1)
        for module in pre_encode.conv:
            x = module(x)
            if isinstance(module, torch.nn.Conv2d) and module.stride[0] == 2:
                length = torch.div(length - 1, 2, rounding_mode="floor") + 1
            valid = torch.arange(x.shape[2]).unsqueeze(0) < length.unsqueeze(1)
            x = x * valid[:, None, :, None].to(x.dtype)
        batch, channels, frames, mels = x.shape
        return pre_encode.out(x.transpose(1, 2).reshape(batch, frames, channels * mels)), length
