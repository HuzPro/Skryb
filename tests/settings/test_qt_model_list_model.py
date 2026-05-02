from unittest.mock import MagicMock

import pytest

from voicetotext.settings.model_view import ModelView
from voicetotext.settings.qt_model_list_model import (
    DownloadProgressSnapshot,
    QtModelListModel,
)


def _view(name: str, installed: bool = False) -> ModelView:
    return ModelView(
        name=name,
        display_size="74 MB",
        description=f"{name} model",
        installed=installed,
    )


def _fake_service(views: list[ModelView], selected: str = "base"):
    svc = MagicMock()
    svc.list_models.return_value = views
    svc.selected_model.return_value = selected
    return svc


@pytest.fixture
def model_fixture(qtbot):
    views = [_view("tiny", installed=True), _view("base"), _view("small")]
    service = _fake_service(views, selected="base")
    list_model = QtModelListModel(service)
    return list_model, service, views


class TestQtModelListModelRowCount:
    def test_should_report_row_count_matching_service_list(self, model_fixture):
        list_model, _, views = model_fixture

        assert list_model.rowCount() == len(views)

    def test_should_report_zero_for_valid_parent_index(self, model_fixture):
        list_model, *_ = model_fixture
        parent = list_model.index(0)

        assert list_model.rowCount(parent) == 0


class TestQtModelListModelData:
    def test_should_return_model_view_under_model_role(self, model_fixture):
        list_model, _, views = model_fixture

        idx = list_model.index(1)
        value = list_model.data(idx, QtModelListModel.MODEL_ROLE)

        assert value == views[1]

    def test_should_return_true_under_selected_role_for_selected_model(
        self, model_fixture
    ):
        list_model, _, views = model_fixture

        base_idx = list_model.index(1)  # base
        tiny_idx = list_model.index(0)

        assert list_model.data(base_idx, QtModelListModel.SELECTED_ROLE) is True
        assert list_model.data(tiny_idx, QtModelListModel.SELECTED_ROLE) is False

    def test_should_return_none_under_download_role_when_no_download(
        self, model_fixture
    ):
        list_model, *_ = model_fixture
        idx = list_model.index(0)

        assert list_model.data(idx, QtModelListModel.DOWNLOAD_ROLE) is None

    def test_should_return_none_for_invalid_index(self, model_fixture):
        list_model, *_ = model_fixture
        invalid = list_model.index(999)

        assert list_model.data(invalid, QtModelListModel.MODEL_ROLE) is None

    def test_should_return_empty_string_for_display_role(self, model_fixture):
        # Custom row widget renders the row; default DisplayRole text would
        # paint behind the widget and overlap it.
        from PySide6.QtCore import Qt

        list_model, *_ = model_fixture
        idx = list_model.index(0)

        assert list_model.data(idx, Qt.DisplayRole) == ""

    def test_should_return_size_hint_tall_enough_for_row_widget(self, model_fixture):
        from PySide6.QtCore import QSize, Qt

        list_model, *_ = model_fixture
        idx = list_model.index(0)

        size = list_model.data(idx, Qt.SizeHintRole)

        assert isinstance(size, QSize)
        assert size.height() >= 60  # accommodates name + description rows


class TestQtModelListModelSelection:
    def test_should_update_selected_row_when_set_selected_called(self, model_fixture):
        list_model, *_ = model_fixture

        list_model.set_selected("small")

        small_idx = list_model.index(2)
        base_idx = list_model.index(1)
        assert list_model.data(small_idx, QtModelListModel.SELECTED_ROLE) is True
        assert list_model.data(base_idx, QtModelListModel.SELECTED_ROLE) is False

    def test_should_emit_data_changed_for_previous_and_new_rows(self, model_fixture):
        list_model, *_ = model_fixture
        emissions: list = []
        list_model.dataChanged.connect(
            lambda top, bot, roles: emissions.append((top.row(), bot.row(), list(roles)))
        )

        list_model.set_selected("small")

        changed_rows = sorted(r for r, _, _ in emissions)
        assert 1 in changed_rows  # old base
        assert 2 in changed_rows  # new small


class TestQtModelListModelDownloadState:
    def test_should_store_and_return_download_snapshot(self, model_fixture):
        list_model, *_ = model_fixture
        snapshot = DownloadProgressSnapshot(
            state="downloading",
            bytes_done=50,
            bytes_total=100,
            speed_bps=10.0,
            eta_seconds=5.0,
            error_message=None,
        )

        list_model.set_download_progress("base", snapshot)

        base_idx = list_model.index(1)
        assert list_model.data(base_idx, QtModelListModel.DOWNLOAD_ROLE) == snapshot

    def test_should_emit_data_changed_when_download_progress_updated(self, model_fixture):
        list_model, *_ = model_fixture
        emissions: list = []
        list_model.dataChanged.connect(
            lambda top, bot, roles: emissions.append((top.row(), list(roles)))
        )
        snapshot = DownloadProgressSnapshot(
            state="downloading", bytes_done=1, bytes_total=2,
            speed_bps=0.0, eta_seconds=0.0, error_message=None,
        )

        list_model.set_download_progress("base", snapshot)

        assert any(row == 1 for row, _ in emissions)

    def test_should_clear_download_snapshot(self, model_fixture):
        list_model, *_ = model_fixture
        list_model.set_download_progress(
            "base",
            DownloadProgressSnapshot(
                state="downloading", bytes_done=1, bytes_total=2,
                speed_bps=0.0, eta_seconds=0.0, error_message=None,
            ),
        )

        list_model.clear_download("base")

        base_idx = list_model.index(1)
        assert list_model.data(base_idx, QtModelListModel.DOWNLOAD_ROLE) is None


class TestQtModelListModelIndexLookup:
    def test_should_return_valid_index_for_known_model(self, model_fixture):
        list_model, _, views = model_fixture

        idx = list_model.index_for_name("base")

        assert idx.isValid()
        assert idx.row() == 1

    def test_should_return_invalid_index_for_unknown_model(self, model_fixture):
        list_model, *_ = model_fixture

        idx = list_model.index_for_name("unknown")

        assert idx.isValid() is False


class TestQtModelListModelRefresh:
    def test_should_re_read_service_and_update_install_state_for_row(
        self, model_fixture
    ):
        list_model, service, _ = model_fixture
        service.list_models.return_value = [
            _view("tiny", installed=True),
            _view("base", installed=True),
            _view("small"),
        ]

        list_model.update_install_state("base")

        base_idx = list_model.index(1)
        view = list_model.data(base_idx, QtModelListModel.MODEL_ROLE)
        assert view.installed is True

    def test_should_emit_data_changed_when_install_state_updates(self, model_fixture):
        list_model, service, _ = model_fixture
        service.list_models.return_value = [
            _view("tiny", installed=True),
            _view("base", installed=True),
            _view("small"),
        ]
        emissions: list = []
        list_model.dataChanged.connect(
            lambda top, bot, roles: emissions.append((top.row(), list(roles)))
        )

        list_model.update_install_state("base")

        assert any(row == 1 for row, _ in emissions)
