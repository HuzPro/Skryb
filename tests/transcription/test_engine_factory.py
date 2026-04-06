from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voicetotext.transcription.engine_factory import (
    _auto_select_engine,
    _resolve_device,
    create_engine,
    create_model_manager,
    resolve_engine_name,
)


class TestAutoSelectEngine:
    @patch("voicetotext.transcription.engine_factory.detect_available_backends")
    def test_should_prefer_faster_whisper_when_available(self, mock_detect):
        mock_detect.return_value = ["faster_whisper", "transformers", "cpu"]

        assert _auto_select_engine() == "faster_whisper"

    @patch("voicetotext.transcription.engine_factory.detect_available_backends")
    def test_should_pick_transformers_when_no_ctranslate2(self, mock_detect):
        mock_detect.return_value = ["transformers", "cpu"]

        assert _auto_select_engine() == "transformers"

    @patch("voicetotext.transcription.engine_factory.detect_available_backends")
    def test_should_fall_back_to_cpu(self, mock_detect):
        mock_detect.return_value = ["cpu"]

        assert _auto_select_engine() == "cpu"


class TestResolveDevice:
    def test_should_return_configured_device_when_not_auto(self):
        assert _resolve_device("faster_whisper", "cpu") == "cpu"
        assert _resolve_device("transformers", "cuda") == "cuda"

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_should_detect_cuda_for_transformers(self):
        import torch
        torch.cuda.is_available.return_value = True

        assert _resolve_device("transformers", "auto") == "cuda"

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_should_fall_back_to_cpu_for_transformers(self):
        import torch
        torch.cuda.is_available.return_value = False

        assert _resolve_device("transformers", "auto") == "cpu"

    @patch.dict("sys.modules", {"ctranslate2": MagicMock()})
    def test_should_detect_cuda_for_faster_whisper(self):
        import ctranslate2
        ctranslate2.get_supported_compute_types.return_value = {"float16"}

        assert _resolve_device("faster_whisper", "auto") == "cuda"

    @patch.dict("sys.modules", {"ctranslate2": MagicMock()})
    def test_should_fall_back_to_cpu_for_faster_whisper(self):
        import ctranslate2
        ctranslate2.get_supported_compute_types.side_effect = RuntimeError()

        assert _resolve_device("faster_whisper", "auto") == "cpu"


class TestResolveEngineName:
    @patch("voicetotext.transcription.engine_factory._auto_select_engine")
    def test_should_resolve_auto(self, mock_auto):
        mock_auto.return_value = "transformers"
        config = MagicMock()
        config.get.return_value = "auto"

        assert resolve_engine_name(config) == "transformers"

    def test_should_return_explicit_engine(self):
        config = MagicMock()
        config.get.return_value = "faster_whisper"

        assert resolve_engine_name(config) == "faster_whisper"


def _make_config(**overrides):
    defaults = {
        "transcription.model_size": "base",
        "transcription.device": "auto",
        "transcription.compute_type": "float16",
        "transcription.language": None,
        "transcription.beam_size": 5,
        "transcription.vad_filter": True,
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

        engine = create_engine("transformers", _make_config(), paths)

        assert isinstance(engine, TransformersWhisperAdapter)

    @patch.dict("sys.modules", {"faster_whisper": MagicMock()})
    @patch("voicetotext.transcription.engine_factory._resolve_device", return_value="cpu")
    def test_should_create_faster_whisper_engine(self, _dev):
        from voicetotext.transcription.faster_whisper_adapter import FasterWhisperAdapter

        paths = MagicMock()
        paths.models_dir = Path("/tmp/models")

        engine = create_engine("faster_whisper", _make_config(), paths)

        assert isinstance(engine, FasterWhisperAdapter)

    @patch.dict("sys.modules", {"faster_whisper": MagicMock()})
    @patch("voicetotext.transcription.engine_factory._resolve_device", return_value="cpu")
    @patch("voicetotext.transcription.engine_factory._auto_select_engine", return_value="faster_whisper")
    def test_cpu_should_create_faster_whisper_with_cpu(self, _auto, _dev):
        from voicetotext.transcription.faster_whisper_adapter import FasterWhisperAdapter

        paths = MagicMock()
        paths.models_dir = Path("/tmp/models")

        engine = create_engine("cpu", _make_config(), paths)

        assert isinstance(engine, FasterWhisperAdapter)


class TestCreateModelManager:
    def test_should_create_manager_for_faster_whisper(self):
        from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
        from voicetotext.transcription.model_info import FASTER_WHISPER_MODELS

        manager = create_model_manager("faster_whisper", Path("/tmp/models"))

        assert isinstance(manager, HuggingFaceModelManager)
        model_names = {m.name for m in manager.list_models()}
        assert model_names == set(FASTER_WHISPER_MODELS.keys())

    def test_should_create_manager_for_transformers(self):
        from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
        from voicetotext.transcription.model_info import TRANSFORMERS_WHISPER_MODELS

        manager = create_model_manager("transformers", Path("/tmp/models"))

        assert isinstance(manager, HuggingFaceModelManager)
        models = manager.list_models()
        repo_ids = {m.repo_id for m in models}
        assert "openai/whisper-base" in repo_ids
