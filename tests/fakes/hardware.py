import threading

from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.hardware_detector import HardwareDetector

_DEFAULT_PROFILE = HardwareProfile(
    kind="amd_rocm", device_name="Fake GPU", gfx_arch="gfx1200",
)


class FakeHardwareDetector(HardwareDetector):
    """Reports a fixed profile and counts the probes asked of it.

    ``detect_calls`` is what makes the deferred-probe contract testable:
    the real detector imports torch on its first call, so a test that
    asserts this stayed at zero is asserting torch was never loaded.
    ``detect_threads`` names the thread each probe ran on. With
    ``probed=False`` the answer is unknown to :meth:`peek` until
    :meth:`detect` runs, like a real detector on a fresh launch.
    """

    def __init__(
        self,
        profile: HardwareProfile = _DEFAULT_PROFILE,
        *,
        probed: bool = True,
    ) -> None:
        self._profile = profile
        self._probed = probed
        self.detect_calls: int = 0
        self.detect_threads: list[str] = []

    def detect(self) -> HardwareProfile:
        self.detect_calls += 1
        self.detect_threads.append(threading.current_thread().name)
        self._probed = True
        return self._profile

    def peek(self) -> HardwareProfile | None:
        return self._profile if self._probed else None
