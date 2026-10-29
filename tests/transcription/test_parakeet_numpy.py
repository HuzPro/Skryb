"""The NumPy front and back end against nano-parakeet's torch modules they replace."""

import subprocess
import sys

import numpy as np
import pytest
import torch

from skryb.transcription.parakeet_hf_checkpoint import preprocessor_buffers
from skryb.transcription.parakeet_numpy import DECODING_TENSORS, TdtDecoder, log_mel_features

_RATE = 16_000


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * _RATE)) / _RATE
    tone = 0.3 * np.sin(2 * np.pi * (180 + 40 * np.sin(3 * t)) * t) * (np.sin(2 * np.pi * 2.5 * t) > 0)
    return (tone + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


def _nano_features(samples: np.ndarray) -> np.ndarray:
    from nano_parakeet.model import LogMelPreprocessor

    preprocessor = LogMelPreprocessor()
    buffers = preprocessor_buffers(torch)
    preprocessor.load_state_dict({"window": buffers["preprocessor.window"], "fb": buffers["preprocessor.fb"]})
    with torch.inference_mode():
        return preprocessor.eval()(torch.from_numpy(samples)).numpy()


@pytest.mark.parametrize("seconds", [257 / _RATE, 0.5, 1.0, 5.4, 30.0])
def test_the_mel_features_match_nano_parakeets(seconds: float) -> None:
    samples = _speechlike(seconds, seed=int(seconds * 100))

    expected = _nano_features(samples)
    features = log_mel_features(samples)

    assert features.shape == expected.shape
    assert np.abs(features - expected).max() < 2e-3


def test_silence_gives_finite_features() -> None:
    assert np.isfinite(log_mel_features(np.zeros(_RATE, np.float32))).all()


def _random_decoding_parts(seed: int, encoded: torch.Tensor | None = None) -> tuple:
    from nano_parakeet.model import RNNTDecoder, TDTJoint

    torch.manual_seed(seed)
    decoder, joint = RNNTDecoder().eval(), TDTJoint().eval()
    out = joint.joint_net[2]
    with torch.no_grad():
        out.weight *= 20
        out.bias[8193] += 3.0  # some zero durations, so a frame can emit several tokens
        if encoded is not None:
            # Blank about half the time, so every branch of the walk is taken.
            start = decoder(torch.tensor([[8192]]))[0].expand(len(encoded), 1, 640)
            logits = joint(encoded[:, None], start)[:, 0, 0, :8192]
            out.weight[8192] = 0
            out.bias[8192] = logits.max(dim=1).values.median()
    tensors = {f"decoder.{k}": v.numpy() for k, v in decoder.state_dict().items()}
    tensors |= {f"joint.{k}": v.numpy() for k, v in joint.state_dict().items()}
    return decoder, joint, tensors


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_greedy_decoding_matches_nano_parakeets_token_for_token(seed: int) -> None:
    from nano_parakeet.model import tdt_greedy_decode

    encoded = torch.randn(150, 1024, generator=torch.Generator().manual_seed(seed))
    decoder, joint, tensors = _random_decoding_parts(seed, encoded)

    with torch.inference_mode():
        expected = tdt_greedy_decode(
            encoder_out=encoded, enc_len=encoded.shape[0], decoder=decoder, joint=joint,
            blank_id=8192, durations=[0, 1, 2, 3, 4],
        )
    tokens = TdtDecoder(tensors).decode(encoded.numpy())

    assert 20 < len(expected) < 8 * len(encoded)
    assert 8192 not in expected
    assert tokens == expected


def test_the_decoder_needs_only_the_listed_tensors() -> None:
    _, _, tensors = _random_decoding_parts(0)

    TdtDecoder({name: tensors[name] for name in DECODING_TENSORS})


def test_an_empty_encoding_decodes_to_nothing() -> None:
    _, _, tensors = _random_decoding_parts(0)

    assert TdtDecoder(tensors).decode(np.zeros((0, 1024), np.float32)) == []


def test_the_npu_engine_runs_without_importing_torch() -> None:
    probe = (
        "import sys, skryb.transcription.openvino_parakeet_adapter;"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False"
