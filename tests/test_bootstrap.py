from unittest.mock import MagicMock, call, patch

import pytest

from voicetotext.bootstrap import _load_engine_with_fallback


class TestLoadEngineWithFallback:
    @patch("voicetotext.bootstrap.create_engine")
    def test_should_return_engine_when_first_attempt_succeeds(self, mock_create):
        engine = MagicMock()
        mock_create.return_value = engine
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback("transformers", config, paths)

        assert result is engine
        engine.load_model.assert_called_once()

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_retry_same_engine_on_cpu_when_gpu_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("CUDA fail")
        cpu_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback("transformers", config, paths)

        assert result is cpu_engine
        assert mock_create.call_args_list == [
            call("transformers", config, paths),
            call("transformers", config, paths, device_override="cpu"),
        ]

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_fall_back_to_alt_engine_when_same_engine_cpu_also_fails(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("CUDA fail")
        cpu_engine = MagicMock()
        cpu_engine.load_model.side_effect = RuntimeError("CPU fail")
        alt_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine, alt_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback("transformers", config, paths)

        assert result is alt_engine
        assert mock_create.call_args_list == [
            call("transformers", config, paths),
            call("transformers", config, paths, device_override="cpu"),
            call("faster_whisper", config, paths),
        ]

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_skip_cpu_retry_when_engine_is_already_cpu(self, mock_create):
        cpu_engine = MagicMock()
        cpu_engine.load_model.side_effect = RuntimeError("broken")
        alt_engine = MagicMock()
        mock_create.side_effect = [cpu_engine, alt_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback("cpu", config, paths)

        assert result is alt_engine
        assert mock_create.call_args_list == [
            call("cpu", config, paths),
            call("transformers", config, paths),
        ]

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_use_transformers_as_alt_for_faster_whisper(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("fail")
        cpu_engine = MagicMock()
        cpu_engine.load_model.side_effect = RuntimeError("fail")
        alt_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine, alt_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback("faster_whisper", config, paths)

        assert mock_create.call_args_list[2] == call("transformers", config, paths)

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_use_faster_whisper_as_alt_for_transformers(self, mock_create):
        gpu_engine = MagicMock()
        gpu_engine.load_model.side_effect = RuntimeError("fail")
        cpu_engine = MagicMock()
        cpu_engine.load_model.side_effect = RuntimeError("fail")
        alt_engine = MagicMock()
        mock_create.side_effect = [gpu_engine, cpu_engine, alt_engine]
        config = MagicMock()
        paths = MagicMock()

        result = _load_engine_with_fallback("transformers", config, paths)

        assert mock_create.call_args_list[2] == call("faster_whisper", config, paths)

    @patch("voicetotext.bootstrap.create_engine")
    def test_should_propagate_error_when_all_fail(self, mock_create):
        engines = [MagicMock() for _ in range(3)]
        for e in engines:
            e.load_model.side_effect = RuntimeError("fail")
        mock_create.side_effect = engines
        config = MagicMock()
        paths = MagicMock()

        with pytest.raises(RuntimeError):
            _load_engine_with_fallback("transformers", config, paths)
