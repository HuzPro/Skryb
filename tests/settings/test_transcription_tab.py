"""Smoke tests for :class:`TranscriptionTab`.

Uses a withdrawn Tk root (the same pattern as
``test_auto_hide_scrollable_frame.py``) and a real
:class:`SettingsService` wired to fake config/model manager collaborators.
Tests exercise the tab's public surface — construction, draft accessors,
download-event handling, and dispose — without mocking the widget layer.

Creating many ``CTk()`` roots in the same process is flaky on Windows,
so the root is module-scoped and children are wiped between tests.
"""

import tkinter
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest

from tests.fakes.config import FakeConfig
from tests.fakes.dispatcher import InlineActionDispatcher
from tests.fakes.model_manager import FakeModelManager
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
)
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import EngineHotSwapCoordinator
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.service import SettingsService
from voicetotext.settings.transcription_tab import TranscriptionTab


def _destroy_safely(widget: tkinter.Misc) -> None:
    """Destroy a Tk widget, ignoring teardown noise from released vars.

    Tabs that own ``tkinter.Variable`` instances release the ``_tk``
    back-reference in their ``dispose()`` so the Variable's ``__del__``
    returns early at GC time (see ``voicetotext.settings.tk_cleanup``).
    But any widget still bound to that Variable — e.g. a
    ``CTkRadioButton`` — calls ``trace_remove`` during its own destroy,
    which explodes on the now-``None`` ``_tk``. In production the
    error surfaces during Python GC and is swallowed by the
    interpreter's ``Exception ignored`` wrapper; in tests we need to
    swallow it explicitly so teardown doesn't leak between cases.
    """
    try:
        widget.destroy()
    except (tkinter.TclError, AttributeError):
        pass


@pytest.fixture(scope="module")
def tk_root():
    # Tcl's init files sometimes refuse to reload after another test
    # module has created and destroyed its own Tk root in the same
    # process. Skip this module's tests gracefully so the full suite
    # stays green; tests still run reliably in isolation.
    try:
        root = ctk.CTk()
    except tkinter.TclError as exc:
        pytest.skip(f"cannot initialize fresh Tk root in this session: {exc}")
    root.withdraw()
    try:
        yield root
    finally:
        _destroy_safely(root)


@pytest.fixture
def clean_root(tk_root):
    for child in list(tk_root.winfo_children()):
        _destroy_safely(child)
    tk_root.update_idletasks()
    return tk_root


def _settle(root: tkinter.Tk) -> None:
    for _ in range(3):
        root.update()


def _build_service(
    installed: dict[str, bool] | None = None,
    config_overrides: dict | None = None,
) -> SettingsService:
    config_data = {
        "transcription.model_size": "base",
        "transcription.language": "en",
        "transcription.device": "cpu",
        "hotkey.combination": "ctrl+alt+m",
        "hotkey.mode": "toggle",
        "output.mode": "direct_paste",
    }
    config_data.update(config_overrides or {})
    config = FakeConfig(config_data)
    manager = FakeModelManager(installed=installed or {"base": True})
    bus = SimpleEventBus()
    cache = ModelInstallationCache(manager, bus)
    downloads = DownloadCoordinator(manager)
    hot_swap = EngineHotSwapCoordinator(
        use_case=MagicMock(),
        event_bus=bus,
        loader=lambda: MagicMock(),
        action_dispatcher=InlineActionDispatcher(),
    )
    return SettingsService(
        config=config,
        model_manager=manager,
        event_bus=bus,
        install_cache=cache,
        downloads=downloads,
        hot_swap=hot_swap,
    )


def _make_tab(parent: ctk.CTkFrame, **service_kwargs) -> TranscriptionTab:
    service = _build_service(**service_kwargs)
    return TranscriptionTab(parent=parent, service=service)


class TestTranscriptionTabConstruction:
    def test_should_build_tab_without_error_given_service_with_models(self, clean_root):
        frame = ctk.CTkFrame(clean_root)
        frame.pack(fill="both", expand=True)

        tab = _make_tab(frame)
        _settle(clean_root)

        assert tab is not None
        assert frame.winfo_children(), "tab should populate its parent frame"

    def test_should_expose_draft_values_via_current_accessors(self, clean_root):
        frame = ctk.CTkFrame(clean_root)
        frame.pack(fill="both", expand=True)

        tab = _make_tab(
            frame,
            config_overrides={
                "transcription.model_size": "tiny",
                "transcription.language": "fr",
                "transcription.device": "cuda",
            },
        )
        _settle(clean_root)

        assert tab.current_model() == "tiny"
        assert tab.current_language() == "fr"
        assert tab.current_device() == "cuda"


class TestTranscriptionTabDownloadEvents:
    def test_should_update_row_progress_on_download_progress_event(self, clean_root):
        frame = ctk.CTkFrame(clean_root)
        frame.pack(fill="both", expand=True)
        tab = _make_tab(frame, installed={})
        _settle(clean_root)

        tab.on_download_progress(
            DownloadProgressEvent(
                job_id="tiny",
                bytes_done=50,
                bytes_total=100,
                speed_bps=1024.0,
                eta_seconds=10.0,
            )
        )
        _settle(clean_root)

        row = tab._model_rows["tiny"]
        assert row._progress_bar is not None
        assert row._status_label is not None
        assert "50%" in row._status_label.cget("text")

    def test_should_mark_row_paused_on_download_state_event(self, clean_root):
        frame = ctk.CTkFrame(clean_root)
        frame.pack(fill="both", expand=True)
        tab = _make_tab(frame, installed={})
        _settle(clean_root)

        tab.on_download_progress(
            DownloadProgressEvent(
                job_id="tiny",
                bytes_done=10,
                bytes_total=100,
                speed_bps=0.0,
                eta_seconds=None,
            )
        )
        _settle(clean_root)
        tab.on_download_state(
            DownloadStateChangedEvent(job_id="tiny", state="paused")
        )
        _settle(clean_root)

        row = tab._model_rows["tiny"]
        assert row._status_label.cget("text") == "paused"

    def test_should_mark_row_installed_on_download_completed_event(self, clean_root):
        frame = ctk.CTkFrame(clean_root)
        frame.pack(fill="both", expand=True)
        tab = _make_tab(frame, installed={})
        _settle(clean_root)

        tab.on_download_completed(DownloadCompletedEvent(job_id="tiny"))
        _settle(clean_root)

        row = tab._model_rows["tiny"]
        assert row._radio.cget("state") == "normal"

    def test_should_display_error_on_download_failed_event(self, clean_root):
        frame = ctk.CTkFrame(clean_root)
        frame.pack(fill="both", expand=True)
        tab = _make_tab(frame, installed={})
        _settle(clean_root)

        tab.on_download_failed(
            DownloadFailedEvent(job_id="tiny", error="network unreachable")
        )
        _settle(clean_root)

        row = tab._model_rows["tiny"]
        assert "network unreachable" in row._status_label.cget("text")


class TestTranscriptionTabDispose:
    def test_should_not_raise_on_dispose(self, clean_root):
        frame = ctk.CTkFrame(clean_root)
        frame.pack(fill="both", expand=True)
        tab = _make_tab(frame)
        _settle(clean_root)

        tab.dispose()
