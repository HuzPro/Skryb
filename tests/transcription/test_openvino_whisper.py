"""The OpenVINO Whisper graphs against transformers' Whisper, on a tiny random model and OpenVINO's CPU device."""

import json

import numpy as np
import pytest
import torch

openvino = pytest.importorskip("openvino")

from transformers import WhisperConfig, WhisperForConditionalGeneration  # noqa: E402

from skryb.transcription import openvino_whisper as module  # noqa: E402
from skryb.transcription.openvino_graph import Safetensors  # noqa: E402
from skryb.transcription.openvino_whisper import OpenVinoWhisper  # noqa: E402
from skryb.transcription.openvino_whisper_graph import build_decoder, build_encoder  # noqa: E402
from skryb.transcription.whisper_features import log_mel_spectrogram  # noqa: E402
from skryb.recording.audio_segment import AudioSegment  # noqa: E402

_VOCAB = 600
_EOS = 599
_SUPPRESSED = [5, 6, 7, 300]
_BEGIN_SUPPRESSED = [220, _EOS]


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    torch.manual_seed(0)
    config = WhisperConfig(
        vocab_size=_VOCAB, num_mel_bins=128, d_model=64, encoder_layers=2, decoder_layers=2, encoder_attention_heads=4,
        decoder_attention_heads=4, encoder_ffn_dim=128, decoder_ffn_dim=128, max_source_positions=1500,
        max_target_positions=448, eos_token_id=_EOS, pad_token_id=_EOS, decoder_start_token_id=1, bos_token_id=_EOS,
    )
    folder = tmp_path_factory.mktemp("whisper")
    WhisperForConditionalGeneration(config).eval().to(torch.bfloat16).save_pretrained(folder)
    (folder / "generation_config.json").write_text(json.dumps(
        {"suppress_tokens": _SUPPRESSED, "begin_suppress_tokens": _BEGIN_SUPPRESSED, "eos_token_id": _EOS}))
    return folder


@pytest.fixture(scope="module")
def reference(snapshot):
    return WhisperForConditionalGeneration.from_pretrained(snapshot, dtype=torch.float32).eval()


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(seconds * 16000)
    return (rng.normal(0, 0.1, count) * (0.5 + 0.5 * np.sin(np.linspace(0, seconds * 6, count)))).astype(np.float32)


def _compiled(model: object) -> object:
    return openvino.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})


def test_the_encoder_projects_transformers_cross_attention_keys_and_values_for_every_decoder_layer(snapshot, reference) -> None:
    config = json.loads((snapshot / "config.json").read_text())
    mel = log_mel_spectrogram(AudioSegment(_speechlike(7.5, 1), 16000))[None]

    got = _compiled(build_encoder(openvino, Safetensors(snapshot / "model.safetensors"), config)).infer_new_request({"mel": mel})
    with torch.no_grad():
        hidden = reference.model.encoder(torch.from_numpy(mel)).last_hidden_state
        for i, layer in enumerate(reference.model.decoder.layers):
            for kind, proj in (("k", layer.encoder_attn.k_proj), ("v", layer.encoder_attn.v_proj)):
                expected = proj(hidden).view(1, 1500, 4, 16).transpose(1, 2).numpy()
                np.testing.assert_allclose(got[f"cross_{kind}.{i}"], expected, atol=1e-4)


def test_the_int8_decoder_gives_transformers_logits_token_after_token(snapshot, reference) -> None:
    config = json.loads((snapshot / "config.json").read_text())
    weights = Safetensors(snapshot / "model.safetensors")
    mel = log_mel_spectrogram(AudioSegment(_speechlike(4.0, 2), 16000))[None]
    cross = _compiled(build_encoder(openvino, weights, config)).infer_new_request({"mel": mel})
    request = _compiled(build_decoder(openvino, weights, config)).create_infer_request()
    for output, value in cross.items():
        request.set_tensor(output.get_any_name(), openvino.Tensor(np.ascontiguousarray(value)))
    embedding = reference.model.decoder.embed_tokens.weight.detach().numpy()
    tokens = np.random.default_rng(3).integers(0, _VOCAB, 40)

    def run(start: int, end: int) -> np.ndarray:
        request.set_tensor("inputs_embeds", openvino.Tensor(embedding[tokens[start:end]][None]))
        request.set_tensor("position_ids", openvino.Tensor(np.arange(start, end)[None]))
        request.infer()
        return np.array(request.get_tensor("logits").data[0])

    request.reset_state()
    got = np.concatenate([run(0, 4)] + [run(i, i + 1) for i in range(4, 40)])
    with torch.no_grad():
        expected = reference(input_features=torch.from_numpy(mel), decoder_input_ids=torch.from_numpy(tokens)[None]).logits[0].numpy()

    assert np.abs(got - expected).max() < 0.03 * np.abs(expected).max()
    assert (got.argmax(-1) == expected.argmax(-1)).mean() > 0.95


def test_generating_never_writes_a_suppressed_token_nor_starts_with_a_blank_or_the_end(snapshot, monkeypatch) -> None:
    monkeypatch.setattr(module, "_NPU_ENCODER_CONFIG", {})
    original = openvino.Core.compile_model
    monkeypatch.setattr(openvino.Core, "compile_model", lambda self, path, device, config=None: original(self, path, "CPU"))
    whisper = OpenVinoWhisper(snapshot)
    whisper.load()
    prompt = [1, 2, 3, 4]

    for seed in range(5):
        tokens = whisper.generate(_speechlike(3.0 + seed, seed), prompt, whisper.begin_suppress_tokens())
        assert not set(tokens) & set(_SUPPRESSED)
        assert len(tokens) <= 448 - len(prompt) - 1
        assert not tokens or tokens[0] not in _BEGIN_SUPPRESSED
