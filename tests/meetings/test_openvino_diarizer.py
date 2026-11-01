"""The NPU diarizer's model step against transformers' Nemotron 3 Diarization, on a tiny random model and the CPU."""

import numpy as np
import pytest
import torch

openvino = pytest.importorskip("openvino")

from transformers import Nemotron3DiarizationConfig, Nemotron3DiarizationForAudioFrameClassification  # noqa: E402

from skryb.meetings.openvino_diarizer import build_step  # noqa: E402
from skryb.transcription.openvino_graph import Safetensors  # noqa: E402

_ROWS = 48


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    config = Nemotron3DiarizationConfig(
        audio_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
                      "num_key_value_heads": 4, "num_mel_bins": 128, "subsampling_factor": 8,
                      "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0}},
        head_config={"audio_hidden_size": 64, "hidden_size": 32, "num_speakers": 8, "subsampling_factor": 8},
    )
    model = Nemotron3DiarizationForAudioFrameClassification(config).eval()
    folder = tmp_path_factory.mktemp("diarizer")
    model.save_pretrained(folder)
    graph = build_step(openvino, Safetensors(folder / "model.safetensors"), config.to_dict(), rows=_ROWS)
    return model, openvino.Core().compile_model(graph, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})


@pytest.mark.parametrize("rows", [1, 13, 47, 48])
def test_a_padded_step_gives_the_unpadded_steps_speaker_logits_for_its_real_rows(tiny, rows: int) -> None:
    model, compiled = tiny
    step = (np.random.default_rng(rows).standard_normal((rows, 64)) * 0.5).astype(np.float32)
    padded = np.zeros((1, _ROWS, 64), np.float32)
    padded[0, :rows] = step
    padded[0, rows:] = 7.0  # whatever sits past the real rows must not reach them

    got = compiled.create_infer_request().infer({"embeds": padded, "valid": np.array([rows])})["logits"][:rows * 8]
    with torch.no_grad():
        expected = model.classifier(model.model(inputs_embeds=torch.from_numpy(step)[None]).last_hidden_state)[0].numpy()

    np.testing.assert_allclose(got, expected, atol=1e-4)
