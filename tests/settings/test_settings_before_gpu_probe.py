"""Settings answers at once on a fresh launch and fills in the GPU answer later.

The GPU probe imports torch, which took 19 s on a cold ROCm machine, so
nothing may wait for it. These tests use a real :class:`SettingsService`
over a detector whose answer is not known yet.
"""
import threading
from pathlib import Path

from tests.fakes.config import FakeConfig
from tests.fakes.hardware import FakeHardwareDetector
from tests.fakes.licensing import AllowAllLicensePolicy
from tests.fakes.model_manager import FakeModelManager
from skryb.core.event_bus import SimpleEventBus
from skryb.core.state_machine import AppStateMachine
from skryb.settings.download_coordinator import DownloadCoordinator
from skryb.settings.engine_hot_swap import NullEngineHotSwap
from skryb.settings.model_installation_cache import ModelInstallationCache
from skryb.settings.service import SettingsService
from skryb.transcription.engine_factory import build_engine_registry
from skryb.transcription.engine_types import HardwareProfile

_TURBO = "crisperwhisper-2.0-turbo"
_NEMOTRON = "nemotron-speech-streaming-en-0.6b"
_ROCM = HardwareProfile(
    kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200",
)
_WAIT_SECONDS = 5.0


def _service(
    detector: FakeHardwareDetector, tmp_path: Path, engine: str = "auto",
) -> SettingsService:
    config = FakeConfig({
        "transcription.model_size": _TURBO,
        "transcription.engine": engine,
        "transcription.language": "en",
        "transcription.device": "auto",
        "hotkey.combination": "ctrl+alt+m",
        "hotkey.mode": "toggle",
        "output.mode": "direct_paste",
    })
    manager = FakeModelManager(installed={_TURBO: True})
    bus = SimpleEventBus()
    return SettingsService(
        config=config,
        model_manager=manager,
        event_bus=bus,
        install_cache=ModelInstallationCache(manager, bus),
        downloads=DownloadCoordinator(manager),
        hot_swap=NullEngineHotSwap(),
        state_machine=AppStateMachine(bus),
        dictionary_path=tmp_path / "dictionary.json",
        license_policy=AllowAllLicensePolicy(),
        registry=build_engine_registry(),
        hardware_detector=detector,
    )


def test_engine_choices_before_the_probe_come_from_the_ranking_that_ignores_hardware(
    tmp_path: Path,
) -> None:
    detector = FakeHardwareDetector(_ROCM, probed=False)
    service = _service(detector, tmp_path)

    turbo = service.engine_candidates_for(_TURBO)

    assert [c.engine_id for c in turbo] == ["crisperwhisper_ct2", "crisperwhisper"]
    assert [c.reason for c in turbo] == ["", ""]
    assert service.engine_candidates_for(_NEMOTRON) == ()
    assert service.supports_streaming_hardware() is False
    assert service.hardware_probe_pending() is True
    assert detector.detect_calls == 0


class _SlowDetector(FakeHardwareDetector):
    def __init__(self, profile: HardwareProfile, release: threading.Event) -> None:
        super().__init__(profile, probed=False)
        self._release = release

    def detect(self) -> HardwareProfile:
        self._release.wait(_WAIT_SECONDS)
        return super().detect()


def test_requests_made_while_a_probe_runs_share_it_and_all_hear_back(
    tmp_path: Path,
) -> None:
    release = threading.Event()
    detector = _SlowDetector(_ROCM, release)
    service = _service(detector, tmp_path)
    heard = [threading.Event() for _ in range(3)]

    for event in heard:
        service.request_hardware_probe(event.set)
    release.set()

    assert all(event.wait(_WAIT_SECONDS) for event in heard)
    assert detector.detect_calls == 1
    assert service.hardware_probe_pending() is False
    assert service.supports_streaming_hardware() is True
