"""Qwen3-ASR as OpenVINO graphs, built straight from ``model.safetensors`` with no torch.

Two graphs, both written beside the download once and compiled for the NPU from then on:

- the audio encoder and projector for one fixed window of ``WINDOW_CHUNKS`` one-second chunks, at fp16.
  Qwen3-ASR's encoder attends within 8 s windows of 104 tokens (13 per chunk), so a clip is that many
  independent windows; the last is padded and told how many of its tokens are real.
- Qwen3's text decoder, stateful (its K/V cache is OpenVINO state) and fed embeddings rather than token ids,
  so audio embeddings drop straight into the prompt. Weights are int8, one scale per output row, in the
  int8 -> Convert -> Multiply form OpenVINO keeps compressed. The state layout, ``beam_idx`` input and names
  follow optimum-intel's, which is what the NPU plugin's LLM pipeline (NPUW) expects.

Weights are read one tensor at a time (``openvino_graph``), so the build needs about the size of the int8
graph in memory, and takes seconds.
"""

import json
from pathlib import Path

import numpy as np

from skryb.transcription.openvino_graph import Graph, Safetensors, opset, save

# Qwen3-ASR's audio windowing, from the model's config: 100 mel frames a chunk, 13 tokens a chunk, 8 chunks a window.
CHUNK_FRAMES = 100
TOKENS_PER_CHUNK = 13
WINDOW_CHUNKS = 8
WINDOW_FRAMES = WINDOW_CHUNKS * CHUNK_FRAMES
ENCODER_FILENAME = "audio-encoder.xml"
DECODER_FILENAME = "text-decoder.xml"
_AUDIO_PREFIX = "model.audio_tower."
_PROJECTOR_PREFIX = "model.multi_modal_projector."
TEXT_PREFIX = "model.language_model."
EMBEDDING = TEXT_PREFIX + "embed_tokens.weight"


def post_cnn_length(frames: int) -> int:
    """Tokens one chunk of *frames* real mel frames gives after the three stride-2 convolutions."""
    for _ in range(3):
        frames = (frames - 1) // 2 + 1 if frames > 0 else 0
    return frames


def window_tokens(frames: int) -> int:
    """Real tokens in a window whose first *frames* mel frames are real."""
    return sum(post_cnn_length(max(0, min(CHUNK_FRAMES, frames - c * CHUNK_FRAMES))) for c in range(WINDOW_CHUNKS))


def build(openvino: object, snapshot: Path, target: Path) -> None:
    """Write both graphs for *snapshot* into *target*; each ``.xml`` lands last, so a half-written one never loads."""
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    weights = Safetensors(snapshot / "model.safetensors")
    target.mkdir(parents=True, exist_ok=True)
    save(openvino, build_audio(openvino, weights, config), target / ENCODER_FILENAME, compress_to_fp16=True)
    save(openvino, build_decoder(openvino, weights, config["text_config"]), target / DECODER_FILENAME,
          compress_to_fp16=False)


def build_audio(openvino: object, weights: Safetensors, config: dict) -> object:
    """mel [1, 128, 800] and the window's real token count -> [104, 2048] audio embeddings for the decoder."""
    g, ops = Graph(openvino, weights), opset(openvino)
    audio = config["audio_config"]
    bins, d_model, heads = audio["num_mel_bins"], audio["d_model"], audio["encoder_attention_heads"]
    head_dim = d_model // heads
    mel = g.parameter([1, bins, WINDOW_FRAMES], openvino.Type.f32, "mel")
    valid = g.parameter([1], openvino.Type.i64, "valid")

    x = ops.reshape(mel, g.const([1, bins, WINDOW_CHUNKS, CHUNK_FRAMES]), False)
    x = ops.reshape(ops.transpose(x, g.const([0, 2, 1, 3])), g.const([WINDOW_CHUNKS, 1, bins, CHUNK_FRAMES]), False)
    for conv in ("conv2d1", "conv2d2", "conv2d3"):
        x = ops.convolution(x, g.const(weights[f"{_AUDIO_PREFIX}{conv}.weight"]), [2, 2], [1, 1], [1, 1], [1, 1])
        x = ops.gelu(ops.add(x, g.const(weights[f"{_AUDIO_PREFIX}{conv}.bias"].reshape(1, -1, 1, 1))), "erf")
    channels = weights.shape(f"{_AUDIO_PREFIX}conv2d3.weight")[0]
    freq = (((bins + 1) // 2 + 1) // 2 + 1) // 2
    x = ops.reshape(ops.transpose(x, g.const([0, 3, 1, 2])), g.const([WINDOW_CHUNKS, TOKENS_PER_CHUNK, channels * freq]), False)
    x = g.dense(x, f"{_AUDIO_PREFIX}conv_out", bias=False)
    x = ops.add(x, g.const(_sinusoids(audio["max_position_embeddings"], d_model)[:TOKENS_PER_CHUNK]))
    length = WINDOW_CHUNKS * TOKENS_PER_CHUNK
    x = ops.reshape(x, g.const([1, length, d_model]), False)

    keep = ops.less(g.const(np.arange(length, dtype=np.int64)[None, :]), ops.unsqueeze(valid, g.const([1])))
    keep = ops.unsqueeze(keep, g.const([1, 2]))
    heads_first = lambda t: ops.transpose(ops.reshape(t, g.const([1, length, heads, head_dim]), False), g.const([0, 2, 1, 3]))
    for i in range(audio["encoder_layers"]):
        p = f"{_AUDIO_PREFIX}layers.{i}."
        h = g.layer_norm(x, p + "self_attn_layer_norm")
        q, k, v = (heads_first(g.dense(h, p + f"self_attn.{n}_proj")) for n in "qkv")
        o = ops.scaled_dot_product_attention(q, k, v, keep, causal=False)
        o = ops.reshape(ops.transpose(o, g.const([0, 2, 1, 3])), g.const([1, length, d_model]), False)
        x = ops.add(x, g.dense(o, p + "self_attn.out_proj"))
        h = g.layer_norm(x, p + "final_layer_norm")
        x = ops.add(x, g.dense(ops.gelu(g.dense(h, p + "fc1"), "erf"), p + "fc2"))
    x = g.layer_norm(x, f"{_AUDIO_PREFIX}ln_post")
    x = g.dense(ops.gelu(g.dense(x, f"{_PROJECTOR_PREFIX}linear_1"), "erf"), f"{_PROJECTOR_PREFIX}linear_2")
    out = ops.reshape(x, g.const([length, -1]), False)
    out.output(0).get_tensor().set_names({"audio"})
    return openvino.Model([ops.result(out)], [mel, valid], "qwen3-asr-audio")


def _sinusoids(length: int, channels: int) -> np.ndarray:
    """Qwen3-ASR's per-chunk position embedding, computed in float32 as transformers does."""
    increment = np.float32(np.log(10000) / (channels // 2 - 1))
    inv = np.exp(-increment * np.arange(channels // 2, dtype=np.float32))
    scaled = np.arange(length, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)


def build_decoder(openvino: object, weights: Safetensors, config: dict, prefix: str = TEXT_PREFIX) -> object:
    """inputs_embeds, attention_mask, position_ids, beam_idx -> logits, with each layer's K/V kept as state."""
    from openvino.op.util import Variable, VariableInfo

    g, ops, f32, i64 = Graph(openvino, weights), opset(openvino), openvino.Type.f32, openvino.Type.i64
    hidden, layers = config["hidden_size"], config["num_hidden_layers"]
    heads, kv_heads, head_dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    eps, theta = config["rms_norm_eps"], config["rope_parameters"]["rope_theta"]

    embeds = g.parameter([1, -1, hidden], f32, "inputs_embeds")
    mask = g.parameter([1, -1], i64, "attention_mask")
    positions = g.parameter([1, -1], i64, "position_ids")
    beam = g.parameter([-1], openvino.Type.i32, "beam_idx")

    inv_freq = (1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)).astype(np.float32)
    freqs = ops.multiply(ops.unsqueeze(ops.convert(positions, f32), g.const([-1])), g.const(inv_freq))
    angles = ops.unsqueeze(ops.concat([freqs, freqs], -1), g.const([1]))
    cos, sin = ops.cos(angles), ops.sin(angles)

    batch = ops.gather(ops.shape_of(embeds), g.const([0]), g.const(0))
    empty_shape = ops.concat([batch, g.const(np.array([kv_heads, 0, head_dim]))], 0)
    sinks, variables = [], []

    def past(layer: int, kind: str) -> tuple[object, object]:
        info = VariableInfo()
        info.data_shape = openvino.PartialShape([-1, kv_heads, -1, head_dim])
        info.data_type = f32
        info.variable_id = f"past_key_values.{layer}.{kind}present.{layer}.{kind}"
        variable = Variable(info)
        variables.append(variable)
        # NPUW finds the K/V cache by these names, and needs each state to start from its own empty tensor.
        read = ops.read_value(ops.broadcast(g.const(np.float32(0)), empty_shape), variable)
        read.output(0).get_tensor().set_names({f"past_key_values.{layer}.{kind}"})
        return ops.gather(read, beam, g.const(0)), variable

    caches = [(past(i, "key"), past(i, "value")) for i in range(layers)]

    # Query i sits at past + i and sees every key up to itself that the mask keeps.
    def counting_to(n: object) -> object:
        return ops.range(g.const(np.int64(0)), ops.squeeze(n, g.const([0])), g.const(np.int64(1)), i64)

    length = ops.gather(ops.shape_of(embeds), g.const([1]), g.const(0))
    past_length = ops.gather(ops.shape_of(caches[0][0][0]), g.const([2]), g.const(0))
    kv_length = ops.add(past_length, length)
    query_at = ops.add(ops.unsqueeze(counting_to(length), g.const([1])), past_length)
    key_at = ops.unsqueeze(counting_to(kv_length), g.const([0]))
    allowed = ops.bitwise_and(ops.less_equal(key_at, query_at),
                              ops.unsqueeze(ops.convert(mask, openvino.Type.boolean), g.const([1, 2])))

    split = lambda t, n: ops.transpose(ops.reshape(t, g.const([1, -1, n, head_dim]), False), g.const([0, 2, 1, 3]))
    repeat = g.const(np.array([1, kv_heads, heads // kv_heads, 1, 1]))
    expand = lambda t: ops.reshape(ops.broadcast(ops.unsqueeze(t, g.const([2])), repeat, broadcast_spec="BIDIRECTIONAL"),
                                   g.const([1, heads, -1, head_dim]), False)

    rotate = lambda t: g.rotate(t, cos, sin, head_dim)

    def cached(new: object, cache: tuple[object, object]) -> object:
        joined = ops.concat([cache[0], new], 2)
        sinks.append(ops.assign(joined, cache[1]))
        return joined

    x = embeds
    for i in range(layers):
        p = f"{prefix}layers.{i}."
        h = g.rms(x, p + "input_layernorm.weight", eps)
        q = rotate(g.rms(split(g.int8_linear(h, p + "self_attn.q_proj.weight"), heads), p + "self_attn.q_norm.weight", eps))
        k = rotate(g.rms(split(g.int8_linear(h, p + "self_attn.k_proj.weight"), kv_heads), p + "self_attn.k_norm.weight", eps))
        v = split(g.int8_linear(h, p + "self_attn.v_proj.weight"), kv_heads)
        keys, values = cached(k, caches[i][0]), cached(v, caches[i][1])
        o = ops.scaled_dot_product_attention(q, expand(keys), expand(values), allowed, causal=False)
        o = ops.reshape(ops.transpose(o, g.const([0, 2, 1, 3])), g.const([1, -1, heads * head_dim]), False)
        x = ops.add(x, g.int8_linear(o, p + "self_attn.o_proj.weight"))
        h = g.rms(x, p + "post_attention_layernorm.weight", eps)
        gate, up = g.int8_linear(h, p + "mlp.gate_proj.weight"), g.int8_linear(h, p + "mlp.up_proj.weight")
        x = ops.add(x, g.int8_linear(ops.multiply(ops.swish(gate), up), p + "mlp.down_proj.weight"))
    # Qwen3-ASR ties its output head to the embedding table.
    logits = g.int8_linear(g.rms(x, f"{prefix}norm.weight", eps), f"{prefix}embed_tokens.weight")
    logits.output(0).get_tensor().set_names({"logits"})
    return openvino.Model([ops.result(logits)], sinks, [embeds, mask, positions, beam], variables, "qwen3-text-decoder")
