import pytest

from skryb.core.exceptions import TranscriptionError
from skryb.transcription.streaming.cache_aware.engine import (
    CacheAwareStreamingEngine,
)


def _engine(device: str = "cuda") -> CacheAwareStreamingEngine:
    return CacheAwareStreamingEngine(
        repo_id="nvidia/nemotron-speech-streaming-en-0.6b",
        model_dir="models",
        device=device,
        model_factory=lambda _device: object(),  # sentinel; never run on GPU
    )


class TestCacheAwareStreamingEngine:
    def test_should_reject_cpu_device_as_gpu_only(self):
        engine = _engine(device="cpu")

        with pytest.raises(ValueError, match="GPU-only"):
            engine.load_model()

    def test_should_raise_when_starting_session_before_load(self):
        with pytest.raises(TranscriptionError, match="not loaded"):
            _engine().start_session()

    def test_should_load_via_model_factory(self):
        engine = _engine()

        engine.load_model()

        assert engine.is_model_loaded() is True

    def test_should_clear_model_on_dispose(self):
        engine = _engine()
        engine.load_model()

        engine.dispose()

        assert engine.is_model_loaded() is False
