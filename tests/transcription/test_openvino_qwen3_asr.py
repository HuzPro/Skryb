"""Qwen3-ASR's OpenVINO graphs and NPU engine, on a tiny random Qwen3-ASR and OpenVINO's CPU device.

The graphs are checked against transformers' own Qwen3-ASR modules; the speculative decoding against plain
greedy decoding through the same graph, which it must match token for token whatever the draft says.
"""

import json
import logging

import numpy as np
import pytest
import torch

openvino = pytest.importorskip("openvino")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from tokenizers import AddedToken  # noqa: E402
from transformers import Qwen3ASRConfig, Qwen3ASRFeatureExtractor, Qwen3ASRForConditionalGeneration  # noqa: E402
from transformers.models.qwen3_asr.modeling_qwen3_asr import _get_feat_extract_output_lengths  # noqa: E402

from skryb.recording.audio_segment import AudioSegment  # noqa: E402
from skryb.transcription import openvino_qwen3_asr_adapter as adapter_module  # noqa: E402
from skryb.transcription.openvino_qwen3_asr_adapter import OpenVinoQwen3AsrAdapter, _Decoder  # noqa: E402
from skryb.transcription.openvino_graph import Safetensors, widen_bf16  # noqa: E402
from skryb.transcription.openvino_qwen3_asr_graph import (  # noqa: E402
    EMBEDDING, WINDOW_FRAMES, build_audio, build_decoder, window_tokens,
)
from skryb.transcription.speculative_draft import Draft  # noqa: E402
from skryb.transcription.whisper_features import log_mel_features  # noqa: E402

_REPO = "test/tiny-qwen3-asr"
_SHA = "0" * 40
_VOCAB = 512
_EOS = (_VOCAB - 2, _VOCAB - 1)
_SPECIALS = ["<|im_start|>", "<|im_end|>", "<|audio_start|>", "<|audio_end|>", "<asr_text>"]


def _tiny_model() -> Qwen3ASRForConditionalGeneration:
    torch.manual_seed(0)
    config = Qwen3ASRConfig(
        audio_config={"num_mel_bins": 128, "d_model": 64, "encoder_layers": 2, "encoder_attention_heads": 4,
                      "encoder_ffn_dim": 128, "downsample_hidden_size": 32, "output_dim": 64, "n_window": 50,
                      "n_window_infer": 800, "max_position_embeddings": 13},
        text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 16, "vocab_size": _VOCAB, "tie_word_embeddings": True,
                     "rope_parameters": {"rope_theta": 1e6, "rope_type": "default"}},
        audio_token_id=_VOCAB - 3,
    )
    return Qwen3ASRForConditionalGeneration(config).eval().to(torch.bfloat16)


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    """A models folder holding the tiny model the way the downloader lays out a snapshot."""
    root = tmp_path_factory.mktemp("models")
    repo = root / f"models--{_REPO.replace('/', '--')}"
    snapshot = repo / "snapshots" / _SHA
    _tiny_model().save_pretrained(snapshot)
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(_SHA)
    (snapshot / "generation_config.json").write_text(json.dumps({"eos_token_id": list(_EOS)}))
    words = [f"w{i}" for i in range(_VOCAB - len(_SPECIALS) - 1)]
    vocab = {word: i for i, word in enumerate(["[UNK]", *_SPECIALS, *words])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in _SPECIALS])
    tokenizer.save(str(snapshot / "tokenizer.json"))
    return root


@pytest.fixture(scope="module")
def snapshot(models_dir):
    return next((models_dir / f"models--{_REPO.replace('/', '--')}" / "snapshots").iterdir())


@pytest.fixture(scope="module")
def reference(snapshot):
    return Qwen3ASRForConditionalGeneration.from_pretrained(snapshot, dtype=torch.float32).eval()


@pytest.fixture(scope="module")
def decoder_graph(snapshot):
    config = json.loads((snapshot / "config.json").read_text())
    model = build_decoder(openvino, Safetensors(snapshot / "model.safetensors"), config["text_config"])
    return openvino.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(seconds * 16000)
    envelope = 0.5 + 0.5 * np.sin(np.linspace(0, seconds * 6, count))
    return (rng.normal(0, 0.1, count) * envelope).astype(np.float32)


@pytest.mark.parametrize("seconds", [0.5, 3.0, 7.99, 8.0, 8.01, 37.7])
def test_the_mel_front_end_matches_qwen3_asrs_own_feature_extractor(seconds: float) -> None:
    samples = _speechlike(seconds, seed=int(seconds * 10))

    # padding=True, as Qwen3-ASR's processor asks for: one clip is not padded out to 30 s.
    expected = Qwen3ASRFeatureExtractor()(samples, sampling_rate=16000, padding=True, return_attention_mask=True)
    frames = int(expected["attention_mask"].sum())
    mine = log_mel_features(samples)

    assert mine.shape == (128, frames)
    np.testing.assert_allclose(mine, expected["input_features"][0, :, :frames].numpy(), atol=1e-4)


def test_window_by_window_token_counts_add_up_to_transformers_count_for_every_length_to_ten_minutes() -> None:
    lengths = np.arange(1, 60_001, 7)

    mine = [sum(window_tokens(min(WINDOW_FRAMES, n - start)) for start in range(0, n, WINDOW_FRAMES)) for n in lengths]

    assert mine == _get_feat_extract_output_lengths(torch.from_numpy(lengths)).tolist()


@pytest.mark.parametrize("seconds", [1.3, 8.0, 11.3, 20.7])
def test_the_audio_graph_gives_transformers_audio_embeddings_one_window_at_a_time(snapshot, reference, seconds) -> None:
    config = json.loads((snapshot / "config.json").read_text())
    encoder = openvino.Core().compile_model(build_audio(openvino, Safetensors(snapshot / "model.safetensors"), config), "CPU",
                                            {"INFERENCE_PRECISION_HINT": "f32"})
    mel = log_mel_features(_speechlike(seconds, seed=3))
    frames = mel.shape[1]

    pieces = []
    for start in range(0, frames, WINDOW_FRAMES):
        window = mel[:, start:start + WINDOW_FRAMES]
        valid = window_tokens(window.shape[1])
        window = np.pad(window, ((0, 0), (0, WINDOW_FRAMES - window.shape[1])))
        pieces.append(encoder.infer_new_request({"mel": window[None], "valid": np.array([valid])})["audio"][:valid])
    padded = np.pad(mel, ((0, 0), (0, (-frames) % 100)))
    mask = torch.zeros(1, padded.shape[1], dtype=torch.long)
    mask[0, :frames] = 1
    with torch.no_grad():
        expected = reference.model.get_audio_features(torch.from_numpy(padded[None]), mask).pooler_output.numpy()

    np.testing.assert_allclose(np.concatenate(pieces), expected, atol=1e-4)


def test_the_int8_text_decoder_gives_transformers_logits_through_its_cache(snapshot, reference, decoder_graph) -> None:
    embedding = widen_bf16(Safetensors(snapshot / "model.safetensors").bf16_rows(EMBEDDING))
    request = decoder_graph.create_infer_request()
    request.reset_state()
    ids = np.random.default_rng(0).integers(0, _VOCAB, 300)

    def run(start: int, end: int) -> np.ndarray:
        return request.infer({"inputs_embeds": embedding[ids[start:end]][None], "attention_mask": np.ones((1, end), np.int64),
                              "position_ids": np.arange(start, end)[None], "beam_idx": np.zeros(1, np.int32)})["logits"][0]

    got = np.concatenate([run(0, 280), run(280, 281), run(281, 300)])
    with torch.no_grad():
        hidden = reference.model.language_model(inputs_embeds=torch.from_numpy(embedding[ids][None])).last_hidden_state
        expected = reference.lm_head(hidden)[0].numpy()

    scale = np.abs(expected).max()
    assert np.abs(got - expected).max() < 0.03 * scale
    assert (got.argmax(-1) == expected.argmax(-1)).mean() > 0.97


def _greedy(decoder: _Decoder, prompt: np.ndarray, embed, budget: int) -> list[int]:
    return decoder.decode(prompt, [], embed, budget, _EOS)


def _edited(tokens: list[int], rng: np.random.Generator) -> list[int]:
    """*tokens* with roughly one in eight swapped, dropped, or doubled with a stray token after it."""
    out = []
    for token in tokens:
        roll = rng.random()
        if roll < 0.04:
            out.append(int(rng.integers(0, _VOCAB - 3)))
        elif roll < 0.08:
            continue
        elif roll < 0.12:
            out += [token, int(rng.integers(0, _VOCAB - 3))]
        else:
            out.append(token)
    return out


@pytest.mark.parametrize("seed", range(4))
def test_drafted_decoding_writes_exactly_what_plain_greedy_decoding_writes_whatever_the_draft(snapshot, decoder_graph, seed) -> None:
    embedding = widen_bf16(Safetensors(snapshot / "model.safetensors").bf16_rows(EMBEDDING))
    embed = lambda ids: embedding[np.asarray(ids, np.int64)].reshape(len(ids), -1)
    rng = np.random.default_rng(seed)
    prompt = embed(rng.integers(0, _VOCAB - 3, 150 + 40 * seed).tolist())
    decoder = _Decoder(decoder_graph, on_npu=False)
    plain = _greedy(decoder, prompt, embed, 80)

    drafts = [plain, _edited(plain, rng), rng.integers(0, _VOCAB - 3, 60).tolist(), plain[:5], plain * 2, []]

    assert len(plain) > 20
    assert all(decoder.decode(prompt, draft, embed, 80, _EOS) == plain for draft in drafts)


def test_a_draft_follows_the_output_through_a_swapped_a_dropped_and_an_added_word() -> None:
    draft = Draft([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    draft.advance([1, 2, 99])
    assert draft.propose(2) == [4, 5]
    draft.advance([4, 6])
    assert draft.propose(2) == [7, 8]
    draft.advance([7, 98, 8])
    assert draft.propose(3) == [9, 10]
    draft.advance([9, 10])
    assert draft.propose(3) == []


def test_a_draft_of_thousands_of_tokens_with_scattered_edits_lets_most_of_them_through() -> None:
    rng = np.random.default_rng(7)
    output = rng.integers(0, 50_000, 3_000).tolist()
    draft = Draft(_edited(output, rng))
    proposed_right = 0

    for token in output:
        proposed_right += draft.propose(1) == [token]
        draft.advance([token])

    assert proposed_right > 0.8 * len(output)


def _on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(adapter_module, "_compile_encoder", lambda core, path, cache, name: (core.compile_model(str(path), "CPU"), "CPU"))
    monkeypatch.setattr(adapter_module, "_compile_decoder",
                        lambda core, path, cache, name: (core.compile_model(str(path), "CPU"), "CPU", None))


@pytest.fixture
def engine(models_dir, monkeypatch):
    _on_cpu(monkeypatch)
    engine = OpenVinoQwen3AsrAdapter(models_dir, _REPO)
    engine.load_model()
    return engine


def test_loading_builds_the_graphs_once_beside_the_download_and_reuses_them(models_dir, snapshot, monkeypatch) -> None:
    _on_cpu(monkeypatch)
    OpenVinoQwen3AsrAdapter(models_dir, _REPO).load_model()
    built = {path.name: path.stat().st_mtime_ns for path in (snapshot / ".openvino").glob("*.xml")}

    OpenVinoQwen3AsrAdapter(models_dir, _REPO).load_model()

    assert set(built) == {"audio-encoder.xml", "text-decoder.xml"}
    assert {path.name: path.stat().st_mtime_ns for path in (snapshot / ".openvino").glob("*.xml")} == built


def test_a_drafter_changes_no_word_of_the_transcript_even_when_it_is_wrong_or_fails(engine, caplog) -> None:
    clip = AudioSegment(_speechlike(6.0, seed=11), 16000)
    plain = engine.transcribe(clip).text

    def failing(audio: AudioSegment) -> str:
        raise RuntimeError("phonon fell over")

    texts = []
    for drafter in (lambda audio: plain, lambda audio: "w1 w2 w3 w4 w5", lambda audio: "", failing):
        engine.drafter = drafter
        with caplog.at_level(logging.WARNING):
            texts.append(engine.transcribe(clip).text)

    assert texts == [plain] * 4
    assert "draft transcript failed" in caplog.text


def test_only_english_is_drafted_since_the_drafting_engine_writes_english(engine) -> None:
    asked = []
    engine.drafter = lambda audio: asked.append(audio.duration_seconds) or "w1"

    engine.transcribe_prompted(AudioSegment(_speechlike(2.0, seed=1), 16000), None, "hi")
    engine.transcribe_prompted(AudioSegment(_speechlike(2.0, seed=1), 16000), None, None)
    engine.transcribe_prompted(AudioSegment(_speechlike(2.0, seed=1), 16000), None, "en")

    assert asked == [2.0]


def test_long_audio_is_cut_at_pauses_and_each_piece_drafted_on_its_own(engine, monkeypatch) -> None:
    monkeypatch.setattr(adapter_module, "PIECE_SECONDS", 6)
    asked = []
    engine.drafter = lambda audio: asked.append(round(audio.duration_seconds)) or ""
    samples = np.concatenate([_speechlike(5.0, 1), np.zeros(8000, np.float32), _speechlike(5.0, 2), np.zeros(8000, np.float32),
                              _speechlike(4.0, 3)])

    engine.transcribe(AudioSegment(samples, 16000))

    assert len(asked) == 3 and sum(asked) == 15


def test_transcribing_before_loading_or_at_the_wrong_rate_is_refused(models_dir, engine) -> None:
    from skryb.core.exceptions import TranscriptionError

    with pytest.raises(TranscriptionError, match="not loaded"):
        OpenVinoQwen3AsrAdapter(models_dir, _REPO).transcribe(AudioSegment(np.zeros(16000, np.float32), 16000))
    with pytest.raises(TranscriptionError, match="16000 Hz"):
        engine.transcribe(AudioSegment(np.zeros(8000, np.float32), 8000))


def test_settings_learn_where_it_runs_and_why(engine) -> None:
    assert engine.encoder_device == "CPU"
    assert engine.device_note is None
