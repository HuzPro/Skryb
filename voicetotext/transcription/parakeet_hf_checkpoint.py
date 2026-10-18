"""Load a Parakeet TDT checkpoint saved in transformers' ``ParakeetForTDT`` layout.

nano-parakeet reads only NeMo ``.nemo`` archives. ``moondream/parakeet-ultra``
is the same network saved as ``model.safetensors`` under transformers'
parameter names, so this renames the parameters back to nano-parakeet's and
supplies the two fixed preprocessor buffers the file does not carry.
"""

from collections.abc import Mapping
from pathlib import Path


# Moondream's checkpoint adds a voice-activity head that nano-parakeet has
# no module for.
_DROPPED_PREFIXES = ("vad_head.",)

_PREFIX_RENAMES = (
    ("encoder.subsampling.layers.", "encoder.pre_encode.conv."),
    ("encoder.subsampling.linear.", "encoder.pre_encode.out."),
    ("encoder_projector.", "joint.enc."),
    ("decoder.decoder_projector.", "joint.pred."),
    ("decoder.embedding.", "decoder.embed."),
    ("joint.head.", "joint.joint_net.2."),
)

# Whole dotted segments inside ``encoder.layers.<n>.*``. ``norm`` on its own
# is only the convolution module's batch norm; the layer norms are
# ``norm_conv``, ``norm_out`` and so on, which a segment match leaves alone.
_ENCODER_LAYER_SEGMENT_RENAMES = {
    "norm": "batch_norm",
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_out",
    "relative_k_proj": "linear_pos",
    "bias_u": "pos_bias_u",
    "bias_v": "pos_bias_v",
}

# NeMo's featurizer settings for every Parakeet TDT 0.6B checkpoint.
_SAMPLE_RATE = 16_000
_N_FFT = 512
_WIN_LENGTH = 400
_N_MELS = 128

_LOADED_MODELS: dict[tuple[str, str, object], object] = {}


def remap_hf_parakeet_state_dict(state_dict: Mapping[str, object]) -> dict[str, object]:
    """Rename transformers ``ParakeetForTDT`` parameters to nano-parakeet's."""
    return {
        _nano_parakeet_key(key): tensor
        for key, tensor in state_dict.items()
        if not key.startswith(_DROPPED_PREFIXES)
    }


def _nano_parakeet_key(key: str) -> str:
    for old_prefix, new_prefix in _PREFIX_RENAMES:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    if key.startswith("encoder.layers."):
        return ".".join(
            _ENCODER_LAYER_SEGMENT_RENAMES.get(segment, segment)
            for segment in key.split(".")
        )
    return key


def preprocessor_buffers(torch_module: object) -> dict[str, object]:
    """The STFT window and mel filterbank NeMo stores inside a ``.nemo``.

    They are fixed by the featurizer settings, not learned, so they are
    rebuilt here: a symmetric Hann window and a Slaney-scale, Slaney-
    normalised mel filterbank, which is what NeMo gets from librosa.
    """
    import torchaudio

    filterbank = torchaudio.functional.melscale_fbanks(
        n_freqs=_N_FFT // 2 + 1,
        f_min=0.0,
        f_max=_SAMPLE_RATE / 2,
        n_mels=_N_MELS,
        sample_rate=_SAMPLE_RATE,
        norm="slaney",
        mel_scale="slaney",
    )
    return {
        "preprocessor.window": torch_module.hann_window(
            _WIN_LENGTH, periodic=False,
        ),
        "preprocessor.fb": filterbank.T.unsqueeze(0),
    }


def load_hf_parakeet(
    nano_parakeet_module: object,
    torch_module: object,
    weights_path: Path,
    device: str,
    dtype: object,
) -> object:
    """Build a ready ``ParakeetTDT`` from a transformers-layout safetensors file.

    Mirrors ``nano_parakeet.from_pretrained`` step for step, including its
    cache, so the batch and streaming engines share one loaded model.
    """
    cache_key = (str(weights_path), device, dtype)
    if cache_key not in _LOADED_MODELS:
        _LOADED_MODELS[cache_key] = _build_model(
            nano_parakeet_module, torch_module, weights_path, device, dtype,
        )
    return _LOADED_MODELS[cache_key]


def _build_model(
    nano_parakeet_module: object,
    torch_module: object,
    weights_path: Path,
    device: str,
    dtype: object,
) -> object:
    import sentencepiece
    from safetensors.torch import load_file

    state_dict = remap_hf_parakeet_state_dict(load_file(str(weights_path)))
    state_dict.update(preprocessor_buffers(torch_module))
    model = nano_parakeet_module.ParakeetTDT()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    # The preprocessor stays fp32 because the STFT needs it, as in
    # nano-parakeet's own loader.
    model.encoder.to(dtype)
    model.decoder.to(dtype)
    model.joint.to(dtype)
    # Same 8,192-piece vocabulary as parakeet-tdt-0.6b-v3, so the tokenizer
    # bundled with nano-parakeet decodes it.
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.LoadFromSerializedProto(
        nano_parakeet_module.get_bundled_tokenizer_proto(),
    )
    model.sp = tokenizer
    model.warmup()
    return model
