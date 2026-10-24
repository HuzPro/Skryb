from pathlib import Path

import torch

from skryb.transcription import parakeet_hf_checkpoint
from skryb.transcription.parakeet_hf_checkpoint import (
    load_hf_parakeet,
    preprocessor_buffers,
    remap_hf_parakeet_state_dict,
)


def _shapes(state_dict: dict) -> dict[str, tuple[int, ...]]:
    return {key: tuple(tensor.shape) for key, tensor in state_dict.items()}


def _transformers_parakeet_state_dict() -> dict:
    """Parameter names and shapes of transformers' ``ParakeetForTDT``.

    Built on the meta device, so no weights are allocated or downloaded.
    The config matches ``moondream/parakeet-ultra``'s ``config.json``.
    """
    from transformers import ParakeetForTDT, ParakeetTDTConfig

    config = ParakeetTDTConfig(encoder_config={
        "num_mel_bins": 128,
        "attention_bias": False,
        "convolution_bias": False,
    })
    with torch.device("meta"):
        return ParakeetForTDT(config).state_dict()


def _nano_parakeet_state_dict() -> dict:
    from nano_parakeet import ParakeetTDT

    with torch.device("meta"):
        return ParakeetTDT().state_dict()


class TestRemapHfParakeetStateDict:
    def test_transformers_parameters_become_exactly_the_parameters_nano_parakeet_expects(
        self,
    ) -> None:
        remapped = remap_hf_parakeet_state_dict(
            _transformers_parakeet_state_dict(),
        )
        expected = _shapes(_nano_parakeet_state_dict())
        # The STFT window and mel filterbank are not in a transformers
        # checkpoint; preprocessor_buffers supplies them.
        del expected["preprocessor.window"], expected["preprocessor.fb"]

        assert _shapes(remapped) == expected

    def test_moondreams_voice_activity_head_is_dropped(self) -> None:
        remapped = remap_hf_parakeet_state_dict({
            "vad_head.proj.weight": torch.zeros(1),
            "joint.head.weight": torch.zeros(1),
        })

        assert set(remapped) == {"joint.joint_net.2.weight"}

    def test_the_convolution_batch_norm_is_renamed_but_layer_norms_keep_their_names(
        self,
    ) -> None:
        remapped = remap_hf_parakeet_state_dict({
            "encoder.layers.3.conv.norm.running_var": torch.zeros(1),
            "encoder.layers.3.norm_conv.weight": torch.zeros(1),
        })

        assert set(remapped) == {
            "encoder.layers.3.conv.batch_norm.running_var",
            "encoder.layers.3.norm_conv.weight",
        }


class TestPreprocessorBuffers:
    def test_buffers_have_the_shapes_nano_parakeet_registers(self) -> None:
        buffers = preprocessor_buffers(torch)
        expected = _shapes(_nano_parakeet_state_dict())

        assert _shapes(buffers) == {
            key: expected[key] for key in ("preprocessor.window", "preprocessor.fb")
        }

    def test_window_is_a_symmetric_hann_window(self) -> None:
        window = preprocessor_buffers(torch)["preprocessor.window"]

        assert torch.allclose(window, window.flip(0), atol=1e-7)
        assert window[0] == 0.0

    def test_every_mel_filter_covers_some_frequency(self) -> None:
        filterbank = preprocessor_buffers(torch)["preprocessor.fb"][0]

        assert bool((filterbank.sum(dim=1) > 0).all())


class TestLoadHfParakeet:
    def test_asking_twice_for_the_same_file_device_and_dtype_builds_the_model_once(
        self, monkeypatch,
    ) -> None:
        builds: list[tuple] = []

        def fake_build(nano_parakeet_module, torch_module, path, device, dtype):
            builds.append((path, device, dtype))
            return object()

        monkeypatch.setattr(parakeet_hf_checkpoint, "_LOADED_MODELS", {})
        monkeypatch.setattr(parakeet_hf_checkpoint, "_build_model", fake_build)
        weights = Path("X:/models/model.safetensors")

        first = load_hf_parakeet(None, torch, weights, "cuda", torch.bfloat16)
        second = load_hf_parakeet(None, torch, weights, "cuda", torch.bfloat16)
        on_cpu = load_hf_parakeet(None, torch, weights, "cpu", torch.float32)

        assert first is second
        assert on_cpu is not first
        assert len(builds) == 2
