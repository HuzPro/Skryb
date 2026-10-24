import types

from skryb.transcription.hardware_detector import HardwareDetector
from skryb.transcription.torch_hardware_detector import TorchHardwareDetector


def _fake_torch(
    *,
    hip_version: str | None = None,
    cuda_available: bool = True,
    device_name: str = "GPU",
    gfx_arch: str = "",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        version=types.SimpleNamespace(hip=hip_version),
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_name=lambda idx: device_name,
            get_device_properties=lambda idx: types.SimpleNamespace(
                name=device_name, gcnArchName=gfx_arch,
            ),
        ),
    )


class TestTorchHardwareDetector:
    def test_should_implement_hardware_detector_interface(self):
        detector = TorchHardwareDetector(torch_loader=lambda: None)

        assert isinstance(detector, HardwareDetector)

    def test_should_return_cpu_profile_when_torch_unavailable(self):
        detector = TorchHardwareDetector(torch_loader=lambda: None)

        profile = detector.detect()

        assert profile.kind == "cpu"
        assert profile.device_name == "CPU"
        assert profile.gfx_arch is None

    def test_should_return_cpu_profile_when_torch_loader_raises(self):
        def raising_loader() -> object | None:
            raise RuntimeError("torch import blew up")

        detector = TorchHardwareDetector(torch_loader=raising_loader)

        profile = detector.detect()

        assert profile.kind == "cpu"

    def test_should_return_cpu_when_cuda_not_available(self):
        fake = _fake_torch(cuda_available=False)
        detector = TorchHardwareDetector(torch_loader=lambda: fake)

        profile = detector.detect()

        assert profile.kind == "cpu"

    def test_should_detect_nvidia_cuda(self):
        fake = _fake_torch(
            hip_version=None,
            device_name="NVIDIA GeForce RTX 4090",
        )
        detector = TorchHardwareDetector(torch_loader=lambda: fake)

        profile = detector.detect()

        assert profile.kind == "nvidia_cuda"
        assert profile.device_name == "NVIDIA GeForce RTX 4090"
        assert profile.gfx_arch is None

    def test_should_detect_amd_rocm_with_gfx_arch(self):
        fake = _fake_torch(
            hip_version="6.2.0",
            device_name="AMD Radeon RX 9060 XT",
            gfx_arch="gfx1200",
        )
        detector = TorchHardwareDetector(torch_loader=lambda: fake)

        profile = detector.detect()

        assert profile.kind == "amd_rocm"
        assert profile.device_name == "AMD Radeon RX 9060 XT"
        assert profile.gfx_arch == "gfx1200"
