"""Project-wide pytest fixtures and marker registration.

Fixtures here are intentionally thin — each one returns a freshly
instantiated hand-rolled fake from :mod:`tests.fakes`.  Tests that
need a specific pre-programmed state should configure the fake inside
the test body so the setup is visible.
"""

import logging
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


# Desktop/Windows-only test modules, dropped from collection off-Windows
# because they fail to *import* there (collection runs before marker
# deselection, so a marker alone cannot save them). Two causes:
#   * ``ctypes.windll`` — the ``win32_clipboard_*`` / ``win32_key_sender``
#     modules, and ``voicetotext.bootstrap`` (which imports them at module
#     top, so any test that imports bootstrap is affected), touch this
#     Windows-only attribute.
#   * ``APPDATA`` — ``tests/performance/conftest.py`` reads this
#     Windows-only environment variable at import time.
collect_ignore: list[str] = []
if sys.platform != "win32":
    collect_ignore = [
        "test_bootstrap.py",
        "test_mode_aware_loader.py",
        "integration/test_config_persistence.py",
        "output/test_win32_clipboard_open.py",
        "output/test_win32_clipboard_preserver.py",
        "output/test_win32_clipboard_writer.py",
        "output/test_win32_key_sender.py",
        "hotkeys/test_win32_keyboard_hook.py",
        "hotkeys/test_win32_hook_adapter.py",
        "settings/test_win32_capture_bridge.py",
        "tray/test_win32_tray_adapter.py",
        "performance",
    ]


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


@pytest.fixture(scope="session")
def tk_root():
    """One withdrawn Tk root shared by every Tk widget test in the session.

    Repeatedly creating and destroying ``tkinter.Tk()`` in a process that
    also imports PySide6 corrupts Tcl's library path (``invalid command
    name "tcl_findLibrary"`` on the second-or-later root). A single
    session-scoped root is created once and never recreated, sidestepping
    it; the workaround becomes unnecessary once Qt is removed. Skips
    cleanly where Tk cannot initialise (headless CI with no display).
    """
    import tkinter

    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:  # pragma: no cover - headless guard
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture()
def app_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """``caplog`` wired to see the non-propagating ``voicetotext`` logger tree.

    :mod:`voicetotext.core.debug_logger` sets ``propagate = False`` on the
    ``voicetotext`` logger at import, so its records never reach the root
    logger pytest captures from. Attaching caplog's handler to that logger and
    lifting its level lets tests assert on the app's own log output.
    """
    logger = logging.getLogger("voicetotext")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="voicetotext")
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


_captured_exit_status = 0


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    global _captured_exit_status
    _captured_exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Bypass a C-extension finalizer crash during interpreter shutdown.

    Whisper's tokenizer pulls in ``sentencepiece`` (and torch loads its
    own native libraries) whose destructors segfault during
    ``Py_FinalizeEx`` — on Windows (``_sentencepiece.cp312-win_amd64.pyd``)
    and on Linux alike — turning an all-green run into a 139 exit. This
    hook fires after the terminal summary is printed and after pytest-cov
    has written its data, so we flush and exit with the real status
    immediately, before the broken finalizers run.
    """
    if sys.platform not in ("win32", "linux"):
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_captured_exit_status)
