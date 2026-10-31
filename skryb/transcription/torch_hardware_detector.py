import logging
from collections.abc import Callable

from skryb.core.rocm_env import prepare_rocm_environment
from skryb.transcription.engine_types import HardwareProfile
from skryb.transcription.hardware_detector import HardwareDetector


_logger = logging.getLogger(__name__)

TorchLoader = Callable[[], object | None]


_CPU_PROFILE = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)


def _import_torch() -> object | None:
    prepare_rocm_environment()
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None
    return torch


class TorchHardwareDetector(HardwareDetector):
    """Probes the GPU through torch, on the first call and only then.

    Importing torch costs over a second, so callers hold the detector and
    ask it when they need an answer rather than probing at startup. The
    first probe's result is kept: the app used to pass one profile
    around, and every consumer still expects the same answer for the
    life of the process.
    """

    def __init__(self, torch_loader: TorchLoader = _import_torch) -> None:
        self._torch_loader = torch_loader
        self._detected: HardwareProfile | None = None

    def detect(self) -> HardwareProfile:
        if self._detected is None:
            self._detected = self._detect_once()
            _logger.info(
                "Detected hardware: %s (%s, gfx=%s)",
                self._detected.kind,
                self._detected.device_name,
                self._detected.gfx_arch,
            )
        return self._detected

    def peek(self) -> HardwareProfile | None:
        return self._detected

    def _detect_once(self) -> HardwareProfile:
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
        props = torch.cuda.get_device_properties(0)
        return HardwareProfile(
            kind="nvidia_cuda",
            device_name=str(torch.cuda.get_device_name(0)),
            # The compute capability as nvcc names it: an RTX 5060 is "sm_120".
            gfx_arch=f"sm_{props.major}{props.minor}",
        )
