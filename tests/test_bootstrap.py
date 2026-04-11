from unittest.mock import MagicMock, call, patch

import pytest

from voicetotext.bootstrap import _load_engine_with_fallback
from voicetotext.transcription.model_info import migrate_removed_models


class TestLoadEngineWithFallback:
    @patch("voicetotext.bootstrap.create_engine")
    def test_should_return_engine_when_first_attempt_succeeds(self, mock_create):
        engine = MagicMock()
        mock_create.return_value = engine
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback(config, paths)

        assert result is engine
        engine.load_model.assert_called_once()

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_retry_on_cpu_when_default_load_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("CUDA fail")
        cpu_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback(config, paths)

        assert result is cpu_engine
        assert mock_create.call_args_list == [
            call(config, paths),
            call(config, paths, device_override="cpu"),
        ]

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_catch_os_error_from_missing_model_files(self, mock_create):
        broken_engine = MagicMock()
        broken_engine.load_model.side_effect = OSError("Can't load feature extractor")
        cpu_engine = MagicMock()
        mock_create.side_effect = [broken_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback(config, paths)

        assert result is cpu_engine

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_propagate_error_when_cpu_fallback_also_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("fail")
        cpu_engine = MagicMock()
        cpu_engine.load_model.side_effect = RuntimeError("fail")
        mock_create.side_effect = [gpu_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        with pytest.raises(RuntimeError):
            _load_engine_with_fallback(config, paths)


class TestMigrateRemovedModels:
    def test_should_rewrite_medium_to_large_v3_turbo(self):
        config = MagicMock()
        config.get.return_value = "medium"

        migrate_removed_models(config)

        config.set.assert_called_once_with(
            "transcription.model_size", "large-v3-turbo"
        )
        config.save.assert_called_once()

    def test_should_leave_other_models_alone(self):
        config = MagicMock()
        config.get.return_value = "small"

        migrate_removed_models(config)

        config.set.assert_not_called()
        config.save.assert_not_called()

    def test_should_leave_large_v3_turbo_alone(self):
        config = MagicMock()
        config.get.return_value = "large-v3-turbo"

        migrate_removed_models(config)

        config.set.assert_not_called()

    def test_should_leave_large_v3_alone(self):
        config = MagicMock()
        config.get.return_value = "large-v3"

        migrate_removed_models(config)

        config.set.assert_not_called()
