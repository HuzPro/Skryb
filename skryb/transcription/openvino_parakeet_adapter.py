"""Parakeet TDT with its encoder on an Intel NPU, through OpenVINO.

The encoder is nearly all of the work. It is exported once from the
transformers-layout weights to OpenVINO IR beside the download
(``snapshots/<sha>/.openvino``) and compiled for the NPU, or for the CPU
when there is no NPU or it refuses the graph. The mel features, the LSTM
prediction network and the greedy TDT decode stay in PyTorch on the CPU:
they are small, and the decode runs once per token.

The NPU compiles static shapes only, so features are padded up to one of
two fixed lengths and the padding is masked out the way NeMo masks a
padded batch: attention ignores padded keys, and the convolution module
sees zeros there. OpenVINO's ``CACHE_DIR`` keeps the compiled blobs, so
only the first load pays for compiling.
"""

import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.backend_imports import make_engine_dependency_error
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.hf_model_manager import loadable_snapshot_dir
from skryb.transcription.parakeet_hf_checkpoint import (
    preprocessor_buffers,
    remap_hf_parakeet_state_dict,
)
from skryb.transcription.parakeet_loader import transformers_layout_weights
from skryb.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_FRAMES_PER_SECOND = 100
_N_MELS = 128
# 68% of logged dictations fit in 10 s and 92% in 30 s. Longer audio is cut
# at a pause into pieces of at most the longest length.
BUCKET_SECONDS = (10, 30)
_PAUSE_SEARCH_SECONDS = 5
_PAUSE_WINDOW_SAMPLES = _SAMPLE_RATE // 10
_IR_DIRECTORY_NAME = ".openvino"
_IR_FILENAME = "encoder.xml"
_CACHE_DIRECTORY_NAME = "compiled"
_NPU = "NPU"
_CPU = "CPU"
_BLANK_ID = 8192
_DURATIONS = [0, 1, 2, 3, 4]
_INSTALL_HINT = "The NPU engine needs OpenVINO. Install it with: pip install openvino"


class OpenVinoParakeetAdapter(TranscriptionEngine):
    """Parakeet TDT (Bark, parakeet-ultra) with the encoder on the NPU."""

    def __init__(self, model_dir: Path, repo_id: str) -> None:
        self._model_dir = model_dir
        self._repo_id = repo_id
        self._model: _NpuParakeet | None = None

    def load_model(self) -> None:
        started_at = time.monotonic()
        openvino = import_openvino()
        nano_parakeet = _import_nano_parakeet()
        weights = transformers_layout_weights(self._model_dir, self._repo_id)
        if weights is None:
            raise TranscriptionError(f"{self._repo_id} has no transformers-layout weights for OpenVINO")
        snapshot = loadable_snapshot_dir(self._model_dir, self._repo_id)
        ir_path = snapshot / _IR_DIRECTORY_NAME / _IR_FILENAME
        state = _remapped_state_dict(weights, with_encoder=not ir_path.is_file())
        if not ir_path.is_file():
            export_encoder(openvino, state, ir_path)
        encoder = compile_encoder(openvino, ir_path, _devices_to_try(openvino))
        self._model = _NpuParakeet(encoder, *_decoding_parts(nano_parakeet, state))
        _logger.info(
            "[load_model openvino] encoder on %s, ready in %.2fs",
            encoder.device, time.monotonic() - started_at,
        )

    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def encoder_device(self) -> str | None:
        """The OpenVINO device the encoder runs on, once loaded: "NPU" or "CPU"."""
        return None if self._model is None else self._model.encoder_device

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if self._model is None:
            raise TranscriptionError("Model not loaded")
        start = time.monotonic()
        pieces = split_at_pauses(audio.data, int(max(BUCKET_SECONDS) * _SAMPLE_RATE))
        text = " ".join(part for part in map(self._model.transcribe, pieces) if part)
        return TranscriptionResult(
            text=text,
            language="en",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - start,
        )

    def dispose(self) -> None:
        self._model = None


class CompiledEncoder:
    """The encoder compiled for one device: per padded length, or for any length."""

    def __init__(self, device: str, compiled_by_frames: dict[int, object]) -> None:
        self.device = device
        self._compiled_by_frames = compiled_by_frames

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Encoder frames for ``[1, 128, T]`` features, padding dropped."""
        frames = features.shape[-1]
        size, compiled = self._compiled_for(frames)
        padded = np.zeros((1, _N_MELS, size), np.float32)
        padded[..., :frames] = features
        result = compiled([padded, np.array([frames], np.int64)])
        encoded, length = result[0], int(result[1][0])
        return encoded[0, :length]

    def _compiled_for(self, frames: int) -> tuple[int, object]:
        if 0 in self._compiled_by_frames:
            return frames, self._compiled_by_frames[0]
        fitting = [size for size in self._compiled_by_frames if size >= frames]
        if not fitting:
            raise TranscriptionError(f"{frames} feature frames exceed the longest compiled length")
        return min(fitting), self._compiled_by_frames[min(fitting)]


class _NpuParakeet:
    def __init__(
        self,
        encoder: CompiledEncoder,
        preprocessor: object,
        decoder: object,
        joint: object,
        tokenizer: object,
        greedy_decode: object,
    ) -> None:
        self._encoder = encoder
        self._preprocessor = preprocessor
        self._decoder = decoder
        self._joint = joint
        self._tokenizer = tokenizer
        self._greedy_decode = greedy_decode

    @property
    def encoder_device(self) -> str:
        return self._encoder.device

    def transcribe(self, samples: np.ndarray) -> str:
        with torch.inference_mode():
            features = self._preprocessor(torch.from_numpy(samples.astype(np.float32)))
            encoded = torch.from_numpy(self._encoder(features.numpy()))
            tokens = self._greedy_decode(
                encoder_out=encoded, enc_len=encoded.shape[0],
                decoder=self._decoder, joint=self._joint,
                blank_id=_BLANK_ID, durations=_DURATIONS,
            )
        return str(self._tokenizer.DecodeIds(tokens)).strip()


def split_at_pauses(samples: np.ndarray, longest: int) -> list[np.ndarray]:
    """Cut *samples* into pieces of at most *longest*, each at its quietest 100 ms.

    The cut is looked for in the last five seconds before the limit, so a
    word is cut through only when the speaker never paused there.
    """
    pieces = []
    search = _PAUSE_SEARCH_SECONDS * _SAMPLE_RATE
    while len(samples) > longest:
        tail = samples[longest - search:longest]
        windows = len(tail) // _PAUSE_WINDOW_SAMPLES
        energy = np.square(tail[:windows * _PAUSE_WINDOW_SAMPLES]).reshape(windows, -1).sum(axis=1)
        cut = longest - search + (int(np.argmin(energy)) + 1) * _PAUSE_WINDOW_SAMPLES
        pieces.append(samples[:cut])
        samples = samples[cut:]
    if len(samples):
        pieces.append(samples)
    return pieces


def import_openvino() -> object:
    """OpenVINO, with its converter's usage reporting kept out.

    OpenVINO's model converter sends usage events to Google Analytics
    unless the user has opted out. With its telemetry package unimportable
    it falls back to its own no-op stub, so Skryb hides the package before
    OpenVINO first loads.
    """
    sys.modules.setdefault("openvino_telemetry", None)
    try:
        import openvino
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="openvino", install_hint=_INSTALL_HINT, underlying=exc,
        ) from exc
    return openvino


def _import_nano_parakeet() -> object:
    try:
        import nano_parakeet
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="nano_parakeet",
            install_hint="Nano-Parakeet not installed: pip install nano-parakeet",
            underlying=exc,
        ) from exc
    return nano_parakeet


def _remapped_state_dict(weights: Path, with_encoder: bool) -> dict[str, torch.Tensor]:
    """nano-parakeet's parameters in fp32; the encoder's only when it is to be exported."""
    from safetensors.torch import load_file

    state = remap_hf_parakeet_state_dict(load_file(str(weights)))
    state.update(preprocessor_buffers(torch))
    return {
        name: tensor.float() if tensor.is_floating_point() else tensor
        for name, tensor in state.items()
        if with_encoder or not name.startswith("encoder.")
    }


def _part(state: dict[str, object], prefix: str) -> dict[str, object]:
    return {name[len(prefix):]: tensor for name, tensor in state.items() if name.startswith(prefix)}


def _decoding_parts(nano_parakeet_module: object, state: dict) -> tuple:
    import sentencepiece
    from nano_parakeet import model as parakeet

    preprocessor = parakeet.LogMelPreprocessor()
    preprocessor.load_state_dict(_part(state, "preprocessor."))
    decoder = parakeet.RNNTDecoder()
    decoder.load_state_dict(_part(state, "decoder."))
    joint = parakeet.TDTJoint()
    joint.load_state_dict(_part(state, "joint."))
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.LoadFromSerializedProto(nano_parakeet_module.get_bundled_tokenizer_proto())
    return (
        preprocessor.eval(), decoder.eval(), joint.eval(), tokenizer, parakeet.tdt_greedy_decode,
    )


def export_encoder(openvino: object, state: dict, ir_path: Path) -> None:
    """Write the encoder as OpenVINO IR at *ir_path*, for any input length.

    Staged in a sibling directory and renamed into place last, so an
    interrupted export never leaves IR that looks finished.
    """
    from nano_parakeet import model as parakeet

    started_at = time.monotonic()
    encoder = parakeet.FastConformerEncoder()
    encoder.load_state_dict(_part(state, "encoder."))
    wrapper = MaskedEncoder(encoder).eval()
    example = (torch.zeros(1, _N_MELS, bucket_frames(2)), torch.tensor([180]))
    with torch.no_grad():
        converted = openvino.convert_model(
            wrapper, example_input=example,
            input=[("features", [1, _N_MELS, -1]), ("length", [1])],
        )
    staging = ir_path.parent.with_name(ir_path.parent.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    openvino.save_model(converted, str(staging / ir_path.name))
    os.replace(staging, ir_path.parent)
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


def bucket_frames(seconds: int) -> int:
    """Feature frames for *seconds* of audio: the STFT is centred, so one extra."""
    return seconds * _FRAMES_PER_SECOND + 1


def _devices_to_try(openvino: object) -> list[str]:
    return ([_NPU] if _NPU in openvino.Core().available_devices else []) + [_CPU]


def compile_encoder(openvino: object, ir_path: Path, devices: list[str]) -> CompiledEncoder:
    """The encoder compiled for the first device that runs it cleanly."""
    for device in devices:
        try:
            return compile_for(openvino, ir_path, device, padded=device == _NPU)
        except RuntimeError as refusal:
            _logger.warning("OpenVINO could not run the encoder on %s: %s", device, refusal)
    raise TranscriptionError(f"OpenVINO could not run the encoder on {', '.join(devices)}")


def compile_for(openvino: object, ir_path: Path, device: str, padded: bool) -> CompiledEncoder:
    """Compile for *device*, at each fixed length when *padded*, and check the output.

    A warm-up on silence must come back finite: a device that computes in
    fp16 and overflows is caught here and the next device is tried.
    """
    core = openvino.Core()
    config = {
        "CACHE_DIR": str(ir_path.parent / _CACHE_DIRECTORY_NAME),
        "PERFORMANCE_HINT": "LATENCY",
    }
    compiled_by_frames = {}
    for frames in ([bucket_frames(s) for s in BUCKET_SECONDS] if padded else [0]):
        model = core.read_model(str(ir_path))
        if frames:
            model.reshape({"features": [1, _N_MELS, frames], "length": [1]})
        compiled_by_frames[frames] = core.compile_model(model, device, config)
    encoder = CompiledEncoder(device, compiled_by_frames)
    warm = encoder(np.zeros((1, _N_MELS, bucket_frames(1)), np.float32))
    if not np.isfinite(warm).all():
        raise RuntimeError(f"the encoder gave non-finite values on {device}")
    return encoder
