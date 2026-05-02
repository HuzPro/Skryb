from collections.abc import Callable

from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.hardware_detector import HardwareDetector


TorchLoader = Callable[[], object | None]


_CPU_PROFILE = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)


def _import_torch() -> object | None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None
    return torch


class TorchHardwareDetector(HardwareDetector):

    def __init__(self, torch_loader: TorchLoader = _import_torch) -> None:
        self._torch_loader = torch_loader

    def detect(self) -> HardwareProfile:
        try:
            return self._probe()
        except Exception:
            return _CPU_PROFILE

    def _probe(self) -> HardwareProfile:
        torch = self._torch_loader()
        if torch is None or not torch.cuda.is_available():
            return _CPU_PROFILE
        if torch.version.hip is not None:
            return self._rocm_profile(torch)
        return self._cuda_profile(torch)

    @staticmethod
    def _rocm_profile(torch: object) -> HardwareProfile:
        props = torch.cuda.get_device_properties(0)
        return HardwareProfile(
            kind="amd_rocm",
            device_name=str(props.name),
            gfx_arch=str(props.gcnArchName) or None,
        )

    @staticmethod
    def _cuda_profile(torch: object) -> HardwareProfile:
        return HardwareProfile(
            kind="nvidia_cuda",
            device_name=str(torch.cuda.get_device_name(0)),
            gfx_arch=None,
        )
