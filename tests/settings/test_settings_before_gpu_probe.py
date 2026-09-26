"""Settings opens at once on a fresh launch and fills in the GPU answer later.

The GPU probe imports torch, which took 19 s on a cold ROCm machine, so
nothing on the GUI thread may wait for it. These tests use a real
:class:`SettingsService` over a detector whose answer is not known yet.
"""
import queue
import threading
import tkinter
from collections.abc import Callable
from pathlib import Path

from tests.fakes.config import FakeConfig
from tests.fakes.hardware import FakeHardwareDetector
from tests.fakes.licensing import AllowAllLicensePolicy
from tests.fakes.model_manager import FakeModelManager
from tests.settings.test_tk_window import FakeCaptureBridge
from tests.settings.tk_pages.test_tk_audio_page import FakeMicrophones
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.state_machine import AppStateMachine
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import NullEngineHotSwap
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.service import SettingsService
from voicetotext.settings.tk_pages.transcription_page import TkTranscriptionPage
from voicetotext.settings.tk_window import TkSettingsWindow
from voicetotext.transcription.engine_factory import build_engine_registry
from voicetotext.transcription.engine_types import HardwareProfile

_TURBO = "crisperwhisper-2.0-turbo"
_NEMOTRON = "nemotron-speech-streaming-en-0.6b"
_ROCM = HardwareProfile(
    kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200",
)
_CPU = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)
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


class _GuiQueue:
    """Work handed to the GUI thread, run when the test says so."""

    def __init__(self) -> None:
        self._actions: queue.Queue[Callable[[], None]] = queue.Queue()

    def schedule(self, action: Callable[[], None]) -> None:
        self._actions.put(action)

    def run_next(self) -> None:
        self._actions.get(timeout=_WAIT_SECONDS)()


def test_opening_settings_before_the_probe_leaves_the_probe_to_a_background_thread(
    tk_root: tkinter.Tk, tmp_path: Path,
) -> None:
    detector = FakeHardwareDetector(_ROCM, probed=False)
    gui = _GuiQueue()
    window = TkSettingsWindow(
        tk_root,
        _service(detector, tmp_path),
        capture_bridge=FakeCaptureBridge(),
        schedule=gui.schedule,
        microphones=FakeMicrophones([]),
    )
    try:
        selector = window.transcription_page().mode_selector()
        assert selector.is_streaming_enabled() is False
        assert selector.streaming_reason() == "Checking for a GPU..."

        gui.run_next()

        assert detector.detect_threads == ["hardware-probe"]
        assert selector.is_streaming_enabled() is True
        assert selector.streaming_reason() == ""
    finally:
        window.destroy()


def test_the_chosen_engine_survives_the_probe_and_then_says_why_it_is_recommended(
    tk_root: tkinter.Tk, tmp_path: Path,
) -> None:
    gui = _GuiQueue()
    page = TkTranscriptionPage(
        tk_root,
        _service(
            FakeHardwareDetector(_ROCM, probed=False), tmp_path,
            engine="crisperwhisper_ct2",
        ),
        schedule=gui.schedule,
    )
    try:
        assert page.engine_ids() == ["auto", "crisperwhisper_ct2", "crisperwhisper"]
        assert page.current_engine() == "crisperwhisper_ct2"
        assert page.engine_reason() == ""

        gui.run_next()

        assert page.current_engine() == "crisperwhisper_ct2"
        assert page.engine_reason() == (
            "Recommended for your AMD ROCm GPU (gfx1200)"
        )
    finally:
        page.destroy()


def test_a_probe_that_finds_no_gpu_keeps_streaming_off_and_says_why(
    tk_root: tkinter.Tk, tmp_path: Path,
) -> None:
    gui = _GuiQueue()
    page = TkTranscriptionPage(
        tk_root,
        _service(FakeHardwareDetector(_CPU, probed=False), tmp_path),
        schedule=gui.schedule,
    )
    try:
        gui.run_next()

        assert page.mode_selector().is_streaming_enabled() is False
        assert "none was detected" in page.mode_selector().streaming_reason()
    finally:
        page.destroy()


def test_a_probe_that_reports_after_the_page_closed_is_ignored(
    tk_root: tkinter.Tk, tmp_path: Path,
) -> None:
    gui = _GuiQueue()
    page = TkTranscriptionPage(
        tk_root,
        _service(FakeHardwareDetector(_ROCM, probed=False), tmp_path),
        schedule=gui.schedule,
    )
    page.destroy()

    gui.run_next()

    assert not page.winfo_exists()


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
