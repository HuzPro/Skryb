"""Project-wide pytest fixtures and marker registration.

Fixtures here are intentionally thin — each one returns a freshly
instantiated hand-rolled fake from :mod:`tests.fakes`.  Tests that
need a specific pre-programmed state should configure the fake inside
the test body so the setup is visible.
"""

import os
import sys

# Mirror ``voicetotext.__main__._apply_torch_env_overrides``: the ROCm
# AOTriton attention path must be enabled before torch is imported, or
# loading a real model on an AMD GPU can hard-crash the interpreter
# during a streaming/warmup pass. Harmless on CPU/CUDA boxes.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import pytest

from tests.fakes import (
    DeferredActionDispatcher,
    FakeAudioRecorder,
    FakeClipboardPreserver,
    FakeHotkeyListener,
    FakeTextOutputHandler,
    FakeTranscriptionEngine,
    InlineActionDispatcher,
    InstantSleeper,
    RecordingEventBus,
)
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.state_machine import AppStateMachine


@pytest.fixture()
def event_bus() -> SimpleEventBus:
    return SimpleEventBus()


@pytest.fixture()
def recording_event_bus() -> RecordingEventBus:
    return RecordingEventBus()


@pytest.fixture()
def state_machine(recording_event_bus: RecordingEventBus) -> AppStateMachine:
    return AppStateMachine(recording_event_bus)


@pytest.fixture()
def instant_sleeper() -> InstantSleeper:
    return InstantSleeper()


@pytest.fixture()
def inline_dispatcher() -> InlineActionDispatcher:
    return InlineActionDispatcher()


@pytest.fixture()
def deferred_dispatcher() -> DeferredActionDispatcher:
    return DeferredActionDispatcher()


@pytest.fixture()
def fake_recorder() -> FakeAudioRecorder:
    return FakeAudioRecorder()


@pytest.fixture()
def fake_engine() -> FakeTranscriptionEngine:
    return FakeTranscriptionEngine()


@pytest.fixture()
def fake_output() -> FakeTextOutputHandler:
    return FakeTextOutputHandler()


@pytest.fixture()
def fake_preserver() -> FakeClipboardPreserver:
    return FakeClipboardPreserver()


@pytest.fixture()
def fake_hotkey_listener() -> FakeHotkeyListener:
    return FakeHotkeyListener()


_captured_exit_status = 0


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    global _captured_exit_status
    _captured_exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Bypass a sentencepiece destructor crash on Windows + Python 3.12.

    Whisper's tokenizer pulls in ``_sentencepiece.cp312-win_amd64.pyd``
    whose destructor segfaults during ``Py_FinalizeEx``. This hook
    fires after the terminal summary is printed, so we flush and exit
    immediately rather than let the broken finalizer run.
    """
    if sys.platform != "win32":
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_captured_exit_status)
