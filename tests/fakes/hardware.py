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
    """

    def __init__(self, profile: HardwareProfile = _DEFAULT_PROFILE) -> None:
        self._profile = profile
        self.detect_calls: int = 0

    def detect(self) -> HardwareProfile:
        self.detect_calls += 1
        return self._profile
