"""Jais-ASR's OpenVINO decoder against a reference on tiny random weights: no real weights, no GPU, no NPU.

Jais-ASR's decoder is Llama's (no per-head norms, rope on whole heads, SwiGLU, an untied head) with RMSNorms
taken over two halves of each row. With one group it is exactly transformers' Llama, which is the reference.
"""

import json

import numpy as np
import pytest
import torch

openvino = pytest.importorskip("openvino")

from safetensors.torch import save_file  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from skryb.transcription.openvino_graph import Graph, open_checkpoint  # noqa: E402
from skryb.transcription.openvino_jais_asr_adapter import OpenVinoJaisAsrAdapter  # noqa: E402
from skryb.transcription.openvino_qwen3_asr_graph import build_decoder  # noqa: E402

_VOCAB = 96


def _llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, vocab_size=_VOCAB, rms_norm_eps=1e-6,
                         rope_theta=1e7, tie_word_embeddings=False, attention_bias=False, mlp_bias=False)
    return LlamaForCausalLM(config).eval()


@pytest.fixture(scope="module")
def jais_shaped(tmp_path_factory):
    """The tiny Llama saved under Jais-ASR's tensor names, sharded in two like the real checkpoint."""
    model = _llama()
    folder = tmp_path_factory.mktemp("jais")
    state = {("lm_head.weight" if k == "lm_head.weight" else "model.language_model." + k.removeprefix("model.")): v.contiguous()
             for k, v in model.state_dict().items()}
    names = sorted(state)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    for shard, keys in shards.items():
        save_file({k: state[k].to(torch.bfloat16) for k in keys}, str(folder / shard))
    weight_map = {k: shard for shard, keys in shards.items() for k in keys}
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return model, folder


def _text_config(groups: int) -> dict:
    return {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"},
            "query_key_norm": False, "layernorm_num_groups": groups}


def _logits(compiled, embeds: np.ndarray) -> np.ndarray:
    n = embeds.shape[0]
    request = compiled.create_infer_request()
    result = request.infer({"inputs_embeds": embeds[None], "attention_mask": np.ones((1, n), np.int64),
                            "position_ids": np.arange(n, dtype=np.int64)[None], "beam_idx": np.zeros(1, np.int32)})
    return result[compiled.output("logits")][0]


def test_the_jais_decoder_graph_gives_llamas_logits_through_an_untied_head_read_from_shards(jais_shaped) -> None:
    model, folder = jais_shaped
    graph = build_decoder(openvino, open_checkpoint(folder), _text_config(groups=1), head="lm_head.weight")
    compiled = openvino.Core().compile_model(graph, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    embeds = np.random.default_rng(1).normal(0, 1, (9, 64)).astype(np.float32)

    got = _logits(compiled, embeds)
    with torch.no_grad():
        bf16_model = model.to(torch.bfloat16).float()
        expected = bf16_model(inputs_embeds=torch.from_numpy(embeds)[None]).logits[0].numpy()

    assert got.shape == (9, _VOCAB)
    assert np.abs(got - expected).max() <= 0.05 * np.abs(expected).max()
    assert (got.argmax(-1) == expected.argmax(-1)).mean() >= 8 / 9


class _Weights:
    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self._tensors = tensors

    def shape(self, name: str) -> list[int]:
        return list(self._tensors[name].shape)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._tensors[name]


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_a_grouped_rms_norm_normalises_each_slice_of_a_row_on_its_own(groups: int) -> None:
    rng = np.random.default_rng(groups)
    x = rng.normal(0, 3, (1, 5, 16)).astype(np.float32)
    x[0, 0, :8] *= 100  # one half far louder than the other
    weight = rng.normal(1, 0.1, 16).astype(np.float32)
    g = Graph(openvino, _Weights({"w": weight}))
    param = g.parameter([1, -1, 16], openvino.Type.f32, "x")
    model = openvino.Model([g.ops.result(g.rms(param, "w", 1e-6, groups))], [param], "rms")

    got = openvino.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"}).infer_new_request({"x": x})
    got = np.asarray(next(iter(got.values())))

    sliced = x.reshape(1, 5, groups, 16 // groups)
    expected = (sliced / np.sqrt((sliced ** 2).mean(-1, keepdims=True) + 1e-6)).reshape(1, 5, 16) * weight
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_the_openvino_jais_engine_asks_for_arabic_in_code_switch_mode_and_drops_the_tag(tmp_path) -> None:
    adapter = OpenVinoJaisAsrAdapter(tmp_path, "IFM/Jais-ASR")

    prefix, _suffix = adapter._prompt("Names and terms that may come up: Northwind.", "ar")

    assert "[Code-switch]" in prefix and "Northwind" not in prefix
    assert adapter._finish("[AE] تمام، نشوفكم بكرة") == "تمام، نشوفكم بكرة"
    assert adapter._finish("[IFM-NoTranscript]") == ""
    assert adapter.name == "Jais-ASR"
