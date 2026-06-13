"""Tests for :class:`TkModelList`.

The plain controller behind the transcription page's scrollable model
list. It owns the row widgets, the per-name download snapshots, and the
selection, replacing Qt's ``QAbstractListModel`` + ``setIndexWidget``
plumbing with direct calls onto :class:`TkModelRow` widgets.
"""

from unittest.mock import MagicMock

import pytest

from voicetotext.settings.download_progress import DownloadProgressSnapshot
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.tk_model_list import (
    TkModelList,
    _scroll_region_height,
    _should_show_scrollbar,
)
from voicetotext.settings.transcription_mode import TranscriptionMode


@pytest.fixture
def root(tk_root):
    return tk_root


def _view(name: str, installed: bool = False) -> ModelView:
    return ModelView(
        name=name,
        display_size="74 MB",
        description=f"{name} model",
        installed=installed,
    )


def _service(views: list[ModelView], selected: str = "base") -> MagicMock:
    svc = MagicMock()
    svc.list_models.return_value = views
    svc.selected_model.return_value = selected
    return svc


def _snapshot(state: str = "downloading") -> DownloadProgressSnapshot:
    return DownloadProgressSnapshot(
        state=state,
        bytes_done=50,
        bytes_total=100,
        speed_bps=10.0,
        eta_seconds=5.0,
        error_message=None,
    )


@pytest.fixture
def fixture(root):
    views = [_view("tiny", installed=True), _view("base"), _view("small")]
    service = _service(views, selected="base")
    widget = TkModelList(root, service)
    yield widget, service, views
    widget.destroy()


class TestTkModelListBuild:
    def test_should_create_one_row_per_service_model(self, fixture):
        model_list, _, _ = fixture

        assert set(model_list.row_names()) == {"tiny", "base", "small"}

    def test_should_default_selection_to_service_selected_model(self, fixture):
        model_list, _, _ = fixture

        assert model_list.selected_model_name() == "base"


class TestTkModelListSelection:
    def test_set_selected_should_update_selection(self, fixture):
        model_list, _, _ = fixture

        model_list.set_selected("small")

        assert model_list.selected_model_name() == "small"

    def test_select_model_should_select_and_notify_for_installed_model(self, fixture):
        model_list, _, _ = fixture
        received: list[str] = []
        model_list.selection_changed.connect(received.append)

        model_list.select_model("tiny")  # installed

        assert model_list.selected_model_name() == "tiny"
        assert received == ["tiny"]

    def test_select_model_should_ignore_uninstalled_model(self, fixture):
        model_list, _, _ = fixture
        received: list[str] = []
        model_list.selection_changed.connect(received.append)

        model_list.select_model("small")  # not installed

        assert model_list.selected_model_name() == "base"
        assert received == []


class TestTkModelListDownload:
    def test_should_store_snapshot_and_show_row_progress(self, fixture):
        model_list, _, _ = fixture
        snapshot = _snapshot()

        model_list.set_download_progress("base", snapshot)

        assert model_list.download_snapshot("base") == snapshot
        assert model_list.row_for("base").progress_visible() is True

    def test_should_report_none_snapshot_when_no_download(self, fixture):
        model_list, _, _ = fixture

        assert model_list.download_snapshot("base") is None

    def test_should_clear_snapshot_and_hide_row_progress(self, fixture):
        model_list, _, _ = fixture
        model_list.set_download_progress("base", _snapshot())

        model_list.clear_download("base")

        assert model_list.download_snapshot("base") is None
        assert model_list.row_for("base").progress_visible() is False


class TestTkModelListInstallState:
    def test_update_install_state_should_refresh_row_to_installed(self, fixture):
        model_list, service, _ = fixture
        service.list_models.return_value = [
            _view("tiny", installed=True),
            _view("base", installed=True),
            _view("small"),
        ]

        model_list.update_install_state("base")

        assert model_list.row_for("base").action_button().cget("text") == "Delete"


class TestTkModelListMode:
    def test_set_mode_should_rebuild_rows_from_service(self, fixture):
        model_list, service, _ = fixture
        service.list_models.return_value = [_view("parakeet-tdt-0.6b-v3")]

        model_list.set_mode(TranscriptionMode.STREAMING)

        assert model_list.row_names() == ["parakeet-tdt-0.6b-v3"]


class TestScrollHelpers:
    def test_should_show_scrollbar_when_content_overflows(self):
        assert _should_show_scrollbar(200, 100) is True

    def test_should_hide_scrollbar_when_content_fits(self):
        assert _should_show_scrollbar(80, 100) is False
        assert _should_show_scrollbar(100, 100) is False

    def test_scroll_region_is_never_shorter_than_viewport(self):
        assert _scroll_region_height(80, 100) == 100
        assert _scroll_region_height(200, 100) == 200


class TestTkModelListSelectionStyle:
    def test_selected_row_uses_the_filled_selection_style(self, fixture):
        model_list, _, _ = fixture

        model_list.select_model("tiny")

        assert model_list.row_for("tiny").cget("style") == "Selected.TFrame"
        assert model_list.row_for("small").cget("style") == "TFrame"


class TestTkModelListActions:
    def test_clicking_download_button_should_call_injected_callback(self, root):
        views = [_view("base")]
        service = _service(views, selected="base")
        received: list[str] = []
        model_list = TkModelList(root, service, on_download=received.append)

        model_list.row_for("base").action_button().invoke()

        assert received == ["base"]

        model_list.destroy()
