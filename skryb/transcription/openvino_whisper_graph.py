"""Whisper (large-v3 and its turbo fine-tunes) as OpenVINO graphs, built straight from ``model.safetensors``.

- The encoder over one 30 s window at fp16, for the NPU. It also projects its output through every decoder
  layer's cross-attention keys and values, which then stay fixed for the whole window, so the decoder never
  touches the encoder's 1500 frames again beyond reading them.
- The decoder with int8 weights, for OpenVINO's CPU device: its self-attention K/V is OpenVINO state, the
  cross-attention K/V are inputs set once a window, and it takes token embeddings looked up outside.
"""

import json
from pathlib import Path

import numpy as np

from skryb.transcription.openvino_graph import Graph, Safetensors, opset, save

ENCODER_FILENAME = "whisper-encoder.xml"
DECODER_FILENAME = "whisper-decoder.xml"
EMBEDDING = "model.decoder.embed_tokens.weight"
WINDOW_FRAMES = 3000
_ENCODER = "model.encoder."
_DECODER = "model.decoder."


def build(openvino: object, snapshot: Path, target: Path) -> None:
    """Write both graphs for *snapshot* into *target*; each ``.xml`` lands last."""
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    weights = Safetensors(snapshot / "model.safetensors")
    target.mkdir(parents=True, exist_ok=True)
    save(openvino, build_encoder(openvino, weights, config), target / ENCODER_FILENAME, compress_to_fp16=True)
    save(openvino, build_decoder(openvino, weights, config), target / DECODER_FILENAME, compress_to_fp16=False)


def _attention_heads(g: Graph, x: object, heads: int, head_dim: int) -> object:
    ops = g.ops
    return ops.transpose(ops.reshape(x, g.const([1, -1, heads, head_dim]), False), g.const([0, 2, 1, 3]))


def _merged_heads(g: Graph, x: object, width: int) -> object:
    ops = g.ops
    return ops.reshape(ops.transpose(x, g.const([0, 2, 1, 3])), g.const([1, -1, width]), False)


def _linear(g: Graph, x: object, name: str, int8: bool, bias: bool = True) -> object:
    if not int8:
        return g.dense(x, name, bias=bias)
    y = g.int8_linear(x, name + ".weight")
    return g.ops.add(y, g.const(g.weights[name + ".bias"])) if bias else y


def build_encoder(openvino: object, weights: Safetensors, config: dict) -> object:
    """mel [1, 128, 3000] -> cross_k.{i}, cross_v.{i} [1, heads, 1500, head_dim] for every decoder layer."""
    g, ops = Graph(openvino, weights), opset(openvino)
    width, heads = config["d_model"], config["encoder_attention_heads"]
    head_dim = width // heads
    mel = g.parameter([1, config["num_mel_bins"], WINDOW_FRAMES], openvino.Type.f32, "mel")
    x = mel
    for conv, stride in (("conv1", 1), ("conv2", 2)):
        x = ops.convolution(x, g.const(weights[f"{_ENCODER}{conv}.weight"]), [stride], [1], [1], [1])
        x = ops.gelu(ops.add(x, g.const(weights[f"{_ENCODER}{conv}.bias"].reshape(1, -1, 1))), "erf")
    x = ops.add(ops.transpose(x, g.const([0, 2, 1])), g.const(weights[f"{_ENCODER}embed_positions.weight"]))
    for i in range(config["encoder_layers"]):
        p = f"{_ENCODER}layers.{i}."
        h = g.layer_norm(x, p + "self_attn_layer_norm")
        q = _attention_heads(g, g.dense(h, p + "self_attn.q_proj"), heads, head_dim)
        k = _attention_heads(g, g.dense(h, p + "self_attn.k_proj", bias=False), heads, head_dim)
        v = _attention_heads(g, g.dense(h, p + "self_attn.v_proj"), heads, head_dim)
        x = ops.add(x, g.dense(_merged_heads(g, ops.scaled_dot_product_attention(q, k, v, causal=False), width),
                               p + "self_attn.out_proj"))
        h = g.layer_norm(x, p + "final_layer_norm")
        x = ops.add(x, g.dense(ops.gelu(g.dense(h, p + "fc1"), "erf"), p + "fc2"))
    x = g.layer_norm(x, f"{_ENCODER}layer_norm")
    results = []
    for i in range(config["decoder_layers"]):
        p = f"{_DECODER}layers.{i}.encoder_attn."
        for kind, bias in (("k", False), ("v", True)):
            out = _attention_heads(g, g.dense(x, f"{p}{kind}_proj", bias=bias), heads, head_dim)
            out.output(0).get_tensor().set_names({f"cross_{kind}.{i}"})
            results.append(ops.result(out))
    return openvino.Model(results, [mel], "whisper-encoder")


def build_decoder(openvino: object, weights: Safetensors, config: dict) -> object:
    """inputs_embeds, position_ids and each layer's cross K/V -> logits, the self-attention K/V kept as state."""
    from openvino.op.util import Variable, VariableInfo

    g, ops, f32, i64 = Graph(openvino, weights), opset(openvino), openvino.Type.f32, openvino.Type.i64
    width, heads, layers = config["d_model"], config["decoder_attention_heads"], config["decoder_layers"]
    head_dim = width // heads
    embeds = g.parameter([1, -1, width], f32, "inputs_embeds")
    positions = g.parameter([1, -1], i64, "position_ids")
    cross = [(g.parameter([1, heads, -1, head_dim], f32, f"cross_k.{i}"), g.parameter([1, heads, -1, head_dim], f32, f"cross_v.{i}"))
             for i in range(layers)]

    x = ops.add(embeds, ops.gather(g.const(weights[f"{_DECODER}embed_positions.weight"]), positions, g.const(0)))
    sinks, variables = [], []

    def cached(new: object, layer: int, kind: str) -> object:
        info = VariableInfo()
        info.data_shape = openvino.PartialShape([1, heads, -1, head_dim])
        info.data_type = f32
        info.variable_id = f"self.{layer}.{kind}"
        variable = Variable(info)
        variables.append(variable)
        empty = ops.broadcast(g.const(np.float32(0)), g.const(np.array([1, heads, 0, head_dim])))
        joined = ops.concat([ops.read_value(empty, variable), new], 2)
        sinks.append(ops.assign(joined, variable))
        return joined

    def counting(n: object) -> object:
        return ops.range(g.const(np.int64(0)), ops.squeeze(n, g.const([0])), g.const(np.int64(1)), i64)

    length = ops.gather(ops.shape_of(embeds), g.const([1]), g.const(0))
    keys_at = None
    for i in range(layers):
        p = f"{_DECODER}layers.{i}."
        h = g.layer_norm(x, p + "self_attn_layer_norm")
        q = _attention_heads(g, _linear(g, h, p + "self_attn.q_proj", True), heads, head_dim)
        k = cached(_attention_heads(g, _linear(g, h, p + "self_attn.k_proj", True, bias=False), heads, head_dim), i, "key")
        v = cached(_attention_heads(g, _linear(g, h, p + "self_attn.v_proj", True), heads, head_dim), i, "value")
        if keys_at is None:
            past = ops.subtract(ops.gather(ops.shape_of(k), g.const([2]), g.const(0)), length)
            query_at = ops.add(ops.unsqueeze(counting(length), g.const([1])), past)
            keys_at = ops.less_equal(ops.unsqueeze(counting(ops.add(past, length)), g.const([0])), query_at)
        o = ops.scaled_dot_product_attention(q, k, v, keys_at, causal=False)
        x = ops.add(x, _linear(g, _merged_heads(g, o, width), p + "self_attn.out_proj", True))
        h = g.layer_norm(x, p + "encoder_attn_layer_norm")
        q = _attention_heads(g, _linear(g, h, p + "encoder_attn.q_proj", True), heads, head_dim)
        o = ops.scaled_dot_product_attention(q, cross[i][0], cross[i][1], causal=False)
        x = ops.add(x, _linear(g, _merged_heads(g, o, width), p + "encoder_attn.out_proj", True))
        h = g.layer_norm(x, p + "final_layer_norm")
        x = ops.add(x, _linear(g, ops.gelu(_linear(g, h, p + "fc1", True), "erf"), p + "fc2", True))
    logits = g.int8_linear(g.layer_norm(x, f"{_DECODER}layer_norm"), EMBEDDING)
    logits.output(0).get_tensor().set_names({"logits"})
    parameters = [embeds, positions] + [p for pair in cross for p in pair]
    return openvino.Model([ops.result(logits)], sinks, parameters, variables, "whisper-decoder")
