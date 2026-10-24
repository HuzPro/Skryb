"""A real API server over real services built from fakes, and plain urllib clients.

Every test talks to the server the way the shell does: HTTP on loopback
with the bearer token, and one ``EventSource``-style stream per client.
"""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from tests.fakes.config import FakeConfig
from tests.fakes.dispatcher import InlineActionDispatcher
from tests.fakes.hardware import FakeHardwareDetector
from tests.fakes.hotkeys import FakeHotkeyListener
from tests.fakes.model_manager import FakeModelManager
from tests.writing.test_service import Rig as WritingRig
from skryb.core.event_bus import SimpleEventBus
from skryb.core.state_machine import AppStateMachine
from skryb.history.service import HistoryService, RetryOutcome
from skryb.recording.input_devices import Microphones
from skryb.settings.download_coordinator import DownloadCoordinator
from skryb.settings.engine_hot_swap import EngineHotSwapCoordinator
from skryb.settings.model_installation_cache import ModelInstallationCache
from skryb.settings.service import SettingsService
from skryb.settings.settings_draft import ChangeSet
from skryb.settings.signal import Signal
from skryb.transcription.engine_factory import build_engine_registry
from skryb.transcription.license_policy import LicensePolicy
from skryb.ui.api_handlers import CoreApi, ShortcutCapture
from skryb.ui.api_server import ApiServer
from skryb.ui.event_hub import EventHub

PARAKEET = "parakeet-tdt-0.6b-v3"
TURBO = "crisperwhisper-2.0-turbo"
NEMOTRON = "nemotron-speech-streaming-en-0.6b"
CRISPERWHISPER_LICENCE = "crisperwhisper-nc"
_TIMEOUT_SECONDS = 10.0


class FakeCaptureBridge:
    def __init__(self) -> None:
        self.key_pressed = Signal()
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def press(self, *names: str) -> None:
        for name in names:
            self.key_pressed.emit(name)


class FakeMicrophones(Microphones):
    """Two microphones, one with a name MME cut at 31 characters, and a meter
    the test feeds levels into by hand."""

    def __init__(self) -> None:
        devices = [
            {"name": "Microphone (Maono PD300X)", "hostapi": 0, "max_input_channels": 1},
            {"name": "Headset (Jabra Evolve2 65 Mon", "hostapi": 0, "max_input_channels": 1},
            {"name": "Headset (Jabra Evolve2 65 Mono)", "hostapi": 1, "max_input_channels": 1},
            {"name": "Speakers (Realtek)", "hostapi": 0, "max_input_channels": 0},
        ]
        super().__init__(query=lambda: (devices, 0), refresh=self._rescanned)
        self.rescans = 0
        self.opens = True
        self.level: Callable[[float], None] | None = None
        self.metered = ""

    def _rescanned(self) -> None:
        self.rescans += 1

    def start_meter(self, name: str, on_level: Callable[[float], None]) -> bool:
        if not self.opens:
            return False
        self.metered, self.level = name, on_level
        return True

    def stop_meter(self) -> None:
        self.metered, self.level = "", None


class Rig:
    def __init__(
        self,
        tmp_path: Path,
        *,
        first_run: bool = False,
        installed: tuple[str, ...] = (PARAKEET,),
        hardware: FakeHardwareDetector | None = None,
        retranscribe: Callable[[Path, Callable[[RetryOutcome], None]], None] | None = None,
        play: Callable[[Path], None] | None = None,
        heartbeat_seconds: float = 15.0,
        on_save: Callable[[ChangeSet], None] | None = None,
        updates: Any = None,
    ) -> None:
        self.config = FakeConfig({
            "transcription.model_size": PARAKEET,
            "transcription.language": "en",
            "transcription.device": "auto",
            "output.mode": "direct_paste",
        })
        self.bus = SimpleEventBus()
        self.manager = FakeModelManager(installed={name: True for name in installed})
        self.state_machine = AppStateMachine(self.bus)
        self.use_case = MagicMock()
        self.service = SettingsService(
            config=self.config,
            model_manager=self.manager,
            event_bus=self.bus,
            install_cache=ModelInstallationCache(self.manager, self.bus),
            downloads=DownloadCoordinator(self.manager),
            hot_swap=EngineHotSwapCoordinator(
                use_case=self.use_case,
                event_bus=self.bus,
                loader=MagicMock,
                action_dispatcher=InlineActionDispatcher(),
            ),
            state_machine=self.state_machine,
            dictionary_path=tmp_path / "dictionary.json",
            license_policy=LicensePolicy(self.config),
            registry=build_engine_registry(),
            hardware_detector=hardware or FakeHardwareDetector(),
        )
        self.saved: list[ChangeSet] = []
        self.listener = FakeHotkeyListener()
        self.bridge = FakeCaptureBridge()
        self.microphones = FakeMicrophones()
        self.opened: list[Path] = []
        self.setup_done = threading.Event()
        self.history_dir = tmp_path / "diagnostics"
        self.history_dir.mkdir(parents=True)
        self.played: list[Path] = []
        self.stopped = 0
        self.writing_rig = WritingRig(tmp_path / "writing", [], locate_files=False)
        self.writing = self.writing_rig.service
        self.hub = EventHub(self.bus, heartbeat_seconds=heartbeat_seconds)
        self.server = ApiServer(self.hub, poll_seconds=0.02)
        self.server.start()
        CoreApi(
            self.service,
            self.hub,
            microphones=self.microphones,
            capture=ShortcutCapture(
                self.bridge,
                lambda captured: self.hub.publish("capture", captured),
                pause=self.listener.pause,
                resume=self.listener.resume,
                run_in_background=lambda action: action(),
            ),
            diagnostics_dir=self.history_dir,
            state=lambda: self.state_machine.state,
            on_save=on_save or self.saved.append,
            writing=None if first_run else self.writing,
            history=None if first_run else HistoryService(
                self.history_dir,
                retranscribe or _never_retranscribes,
                play=play or self.played.append,
                stop=self._count_stop,
            ),
            first_run=first_run,
            on_setup_done=self.setup_done.set,
            open_path=self.opened.append,
            updates=updates,
        ).serve_on(self.server)

    def _count_stop(self) -> None:
        self.stopped += 1

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.port}{path}"

    def call(
        self, method: str, path: str, body: object = None, *, token: str | None = None,
    ) -> tuple[int, Any]:
        headers = {"Content-Type": "application/json"}
        chosen = self.server.token if token is None else token
        if chosen:
            headers["Authorization"] = f"Bearer {chosen}"
        request = urllib.request.Request(
            self.url(path),
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return response.status, _parsed(response.read())
        except urllib.error.HTTPError as error:
            with error:
                return error.code, _parsed(error.read())

    def events(self, token: str | None = None) -> "EventStream":
        return EventStream(self.url(f"/api/events?token={token or self.server.token}"))

    def close(self) -> None:
        self.server.stop()
        self.writing.dispose()


class EventStream:
    """The shell's one ``EventSource``: named events, read in order."""

    def __init__(self, url: str) -> None:
        self._response = urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS)

    def next(self) -> tuple[str, Any]:
        name, data = "", None
        while True:
            line = self._response.readline()
            if not line:
                raise EOFError("the event stream ended")
            text = line.decode("utf-8").rstrip("\n")
            if text.startswith("event: "):
                name = text.removeprefix("event: ")
            elif text.startswith("data: "):
                data = json.loads(text.removeprefix("data: "))
            elif not text and name:
                return name, data

    def until(self, name: str) -> Any:
        while True:
            event, data = self.next()
            if event == name:
                return data

    def close(self) -> None:
        self._response.close()


def _parsed(raw: bytes) -> Any:
    return json.loads(raw) if raw else None


def _never_retranscribes(_wav: Path, on_done: Callable[[RetryOutcome], None]) -> None:
    on_done(RuntimeError("no engine in this test"))
