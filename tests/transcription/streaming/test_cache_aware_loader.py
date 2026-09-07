"""Unit specs for the clean-room ``.nemo`` key remapper.

``remap_state_dict`` rewrites NeMo's checkpoint key names onto our module
hierarchy. It is pure string-keyed dict work (no torch ops, no GPU, no real
checkpoint), so a wrong rename rule can be caught here in milliseconds instead
of only surfacing as a strict-load mismatch when the 2.4GB weights are present.
"""
from __future__ import annotations

from pathlib import Path
import pickle
from unittest.mock import patch

import pytest
import torch

from voicetotext.transcription.streaming.cache_aware.loader import _load_checkpoint

from voicetotext.transcription.streaming.cache_aware.loader import remap_state_dict

_W = "WINDOW_BUFFER"
_FB = "FILTERBANK_BUFFER"
_V = "PARAM_VALUE"


def test_rejects_unsafe_checkpoint_without_unrestricted_retry() -> None:
    with patch(
        "voicetotext.transcription.streaming.cache_aware.loader.torch.load",
        side_effect=ValueError("unsafe checkpoint"),
    ) as load:
        with pytest.raises(ValueError, match="unsafe checkpoint"):
            _load_checkpoint(Path("model.ckpt"), "cpu")

    load.assert_called_once_with("model.ckpt", map_location="cpu", weights_only=True)


def test_rejects_pickled_objects_in_a_real_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"unexpected": Path("private")}, checkpoint)

    with pytest.raises(pickle.UnpicklingError):
        _load_checkpoint(checkpoint, "cpu")


def test_loads_tensor_weights_without_unrestricted_pickle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    expected = {"encoder.weight": torch.arange(3)}
    torch.save(expected, checkpoint)

    loaded = _load_checkpoint(checkpoint, "cpu")

    assert isinstance(loaded, dict)
    assert torch.equal(loaded["encoder.weight"], expected["encoder.weight"])


class TestRemapStateDict:
    def test_renames_preprocessor_featurizer_window_and_fb(self) -> None:
        sd = {
            "preprocessor.featurizer.window": _W,
            "preprocessor.featurizer.fb": _FB,
        }

        remapped = remap_state_dict(sd)

        assert remapped == {"preprocessor.window": _W, "preprocessor.fb": _FB}

    def test_renames_prediction_embedding_to_decoder_embed(self) -> None:
        sd = {"decoder.prediction.embed.weight": _V}

        remapped = remap_state_dict(sd)

        assert remapped == {"decoder.embed.weight": _V}

    def test_lstm_prefix_maps_to_decoder_lstm_not_generic_prediction(self) -> None:
        # The LSTM rule must win over the generic ``decoder.prediction.*`` rule,
        # otherwise the key would become ``decoder.dec_rnn.lstm.*`` and miss.
        sd = {"decoder.prediction.dec_rnn.lstm.weight_ih_l0": _V}

        remapped = remap_state_dict(sd)

        assert remapped == {"decoder.lstm.weight_ih_l0": _V}

    def test_conv_batch_norm_is_renamed_to_conv_norm(self) -> None:
        sd = {"encoder.layers.3.conv.batch_norm.weight": _V}

        remapped = remap_state_dict(sd)

        assert remapped == {"encoder.layers.3.conv.norm.weight": _V}

    def test_other_prediction_params_drop_the_prediction_segment(self) -> None:
        sd = {"decoder.prediction.foo.weight": _V}

        remapped = remap_state_dict(sd)

        assert remapped == {"decoder.foo.weight": _V}

    def test_unmatched_keys_pass_through_unchanged(self) -> None:
        sd = {"encoder.layers.0.self_attn.linear_q.weight": _V}

        remapped = remap_state_dict(sd)

        assert remapped == {"encoder.layers.0.self_attn.linear_q.weight": _V}

    def test_preserves_values_by_identity(self) -> None:
        value = object()
        sd = {"decoder.prediction.embed.weight": value}

        remapped = remap_state_dict(sd)

        assert remapped["decoder.embed.weight"] is value
