from unittest.mock import MagicMock, patch

from voicetotext.transcription.gpu_detection import (
    detect_available_backends,
    get_device_info,
)


class TestDetectAvailableBackends:
    @patch.dict("sys.modules", {"ctranslate2": MagicMock(), "torch": MagicMock()})
    def test_should_return_all_backends_when_both_available(self):
        import ctranslate2
        import torch

        ctranslate2.get_supported_compute_types.return_value = {"float16"}
        torch.cuda.is_available.return_value = True

        result = detect_available_backends()

        assert result == ["faster_whisper", "transformers", "cpu"]

    @patch.dict("sys.modules", {"ctranslate2": MagicMock(), "torch": MagicMock()})
    def test_should_return_only_transformers_when_no_ctranslate2_cuda(self):
        import ctranslate2
        import torch

        ctranslate2.get_supported_compute_types.side_effect = RuntimeError("no CUDA")
        torch.cuda.is_available.return_value = True

        result = detect_available_backends()

        assert result == ["transformers", "cpu"]

    @patch.dict("sys.modules", {"ctranslate2": MagicMock(), "torch": MagicMock()})
    def test_should_return_only_faster_whisper_when_no_torch_gpu(self):
        import ctranslate2
        import torch

        ctranslate2.get_supported_compute_types.return_value = {"float16"}
        torch.cuda.is_available.return_value = False

        result = detect_available_backends()

        assert result == ["faster_whisper", "cpu"]

    @patch.dict("sys.modules", {"ctranslate2": MagicMock(), "torch": MagicMock()})
    def test_should_return_cpu_only_when_no_gpu(self):
        import ctranslate2
        import torch

        ctranslate2.get_supported_compute_types.side_effect = RuntimeError("no CUDA")
        torch.cuda.is_available.return_value = False

        result = detect_available_backends()

        assert result == ["cpu"]


class TestGetDeviceInfo:
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_should_return_gpu_name_when_available(self):
        import torch

        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "AMD Radeon RX 9060 XT"

        result = get_device_info()

        assert result == "AMD Radeon RX 9060 XT"

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_should_return_cpu_only_when_no_gpu(self):
        import torch

        torch.cuda.is_available.return_value = False

        result = get_device_info()

        assert result == "CPU only"
