from unittest.mock import MagicMock, patch

from voicetotext.bootstrap import _resolve_device


class TestResolveDevice:
    @patch.dict("sys.modules", {"ctranslate2": MagicMock()})
    def test_should_return_cuda_when_ctranslate2_succeeds(self):
        import ctranslate2

        ctranslate2.get_supported_compute_types.return_value = {
            "float16",
            "int8",
            "float32",
        }

        result = _resolve_device("auto")

        assert result == "cuda"

    @patch.dict("sys.modules", {"ctranslate2": MagicMock()})
    def test_should_return_cpu_when_ctranslate2_raises(self):
        import ctranslate2

        ctranslate2.get_supported_compute_types.side_effect = RuntimeError("no CUDA")

        result = _resolve_device("auto")

        assert result == "cpu"

    def test_should_return_configured_device_when_not_auto(self):
        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("cuda") == "cuda"
