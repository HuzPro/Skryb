"""Project-wide pytest fixtures and marker registration.

Fixtures here are intentionally thin — each one returns a freshly
instantiated hand-rolled fake from :mod:`tests.fakes`.  Tests that
need a specific pre-programmed state should configure the fake inside
the test body so the setup is visible.
"""

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
