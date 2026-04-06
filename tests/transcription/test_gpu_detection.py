from unittest.mock import MagicMock, patch

from voicetotext.transcription.gpu_detection import (
    detect_available_backends,
    get_device_info,
)


class TestDetectAvailableBackends:
    def test_should_always_return_transformers(self):
        result = detect_available_backends()

        assert result == ["transformers"]


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
