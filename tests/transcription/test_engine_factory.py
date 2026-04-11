from pathlib import Path
from unittest.mock import MagicMock, patch

from voicetotext.transcription.engine_factory import (
    _resolve_device,
    create_engine,
    create_model_manager,
)


class TestResolveDevice:
    def test_should_return_configured_device_when_not_auto(self):
        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("cuda") == "cuda"

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_should_detect_cuda_when_available(self):
        import torch
        torch.cuda.is_available.return_value = True

        assert _resolve_device("auto") == "cuda"

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_should_fall_back_to_cpu_when_cuda_unavailable(self):
        import torch
        torch.cuda.is_available.return_value = False

        assert _resolve_device("auto") == "cpu"


def _make_config(**overrides):
    defaults = {
        "transcription.model_size": "base",
        "transcription.device": "auto",
        "transcription.language": None,
        "transcription.beam_size": 5,
    }
    defaults.update(overrides)
    config = MagicMock()
    config.get.side_effect = lambda k, d=None: defaults.get(k, d)
    return config


class TestCreateEngine:
    @patch("voicetotext.transcription.engine_factory._resolve_device", return_value="cpu")
    def test_should_create_transformers_engine(self, _dev):
        from voicetotext.transcription.transformers_whisper_adapter import (
            TransformersWhisperAdapter,
        )

        paths = MagicMock()
        paths.models_dir = Path("/tmp/models")

        engine = create_engine(_make_config(), paths)

        assert isinstance(engine, TransformersWhisperAdapter)

    @patch("voicetotext.transcription.engine_factory._resolve_device", return_value="cuda")
    def test_device_override_should_force_cpu(self, _dev):
        from voicetotext.transcription.transformers_whisper_adapter import (
            TransformersWhisperAdapter,
        )

        paths = MagicMock()
        paths.models_dir = Path("/tmp/models")

        engine = create_engine(_make_config(), paths, device_override="cpu")

        assert isinstance(engine, TransformersWhisperAdapter)
        assert engine._device == "cpu"
        _dev.assert_not_called()


class TestCreateModelManager:
    def test_should_create_huggingface_manager_with_transformers_lineup(self):
        from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager

        download_manager = MagicMock()
        manager = create_model_manager(Path("/tmp/models"), download_manager)

        assert isinstance(manager, HuggingFaceModelManager)
        repo_ids = {m.repo_id for m in manager.list_models()}
        assert "openai/whisper-base" in repo_ids
