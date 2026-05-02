"""Tests for :class:`ModelRowWidget`.

The widget is the per-row UI inside :class:`QListView`'s index widgets.
It owns a single action button that flips between Download/Cancel/Delete
depending on install + download state, and a progress bar that's only
visible while a download is active.
"""

import pytest

from voicetotext.settings.model_view import ModelView
from voicetotext.settings.qt_model_list_model import DownloadProgressSnapshot
from voicetotext.settings.qt_model_row import ModelRowWidget


def _view(name: str = "base", installed: bool = False) -> ModelView:
    return ModelView(
        name=name,
        display_size="290 MB",
        description=f"{name} description",
        installed=installed,
    )


def _snapshot(
    state: str = "downloading",
    bytes_done: int = 0,
    bytes_total: int = 304087040,  # ~290 MB
    speed_bps: float = 0.0,
    eta_seconds: float = 0.0,
    error_message: str | None = None,
) -> DownloadProgressSnapshot:
    return DownloadProgressSnapshot(
        state=state,
        bytes_done=bytes_done,
        bytes_total=bytes_total,
        speed_bps=speed_bps,
        eta_seconds=eta_seconds,
        error_message=error_message,
    )


@pytest.fixture
def widget(qtbot):
    w = ModelRowWidget(_view())
    qtbot.addWidget(w)
    return w


class TestModelRowWidgetRendering:
    def test_should_show_model_name(self, widget):
        assert widget.name_label().text() == "base"

    def test_should_show_display_size(self, widget):
        assert "290 MB" in widget.size_label().text()

    def test_should_show_description(self, widget):
        assert "base description" in widget.description_label().text()

    def test_progress_bar_should_be_hidden_initially(self, widget):
        assert widget.progress_bar().isHidden() is True

    def test_action_button_should_say_download_when_not_installed(self, widget):
        assert widget.action_button().text() == "Download"

    def test_action_button_should_say_delete_when_installed(self, qtbot):
        w = ModelRowWidget(_view(installed=True))
        qtbot.addWidget(w)

        assert w.action_button().text() == "Delete"


class TestModelRowWidgetUpdateView:
    def test_should_switch_to_delete_button_when_install_state_flips(self, widget):
        widget.update_view(_view(installed=True))

        assert widget.action_button().text() == "Delete"

    def test_should_switch_back_to_download_when_uninstalled(self, qtbot):
        w = ModelRowWidget(_view(installed=True))
        qtbot.addWidget(w)

        w.update_view(_view(installed=False))

        assert w.action_button().text() == "Download"


class TestModelRowWidgetUpdateDownload:
    def test_should_show_progress_bar_when_downloading(self, widget):
        widget.update_download(_snapshot(state="downloading", bytes_done=10_000_000))

        assert widget.progress_bar().isHidden() is False

    def test_should_set_progress_bar_value_from_bytes_done_and_total(self, widget):
        widget.update_download(_snapshot(
            state="downloading", bytes_done=50_000_000, bytes_total=100_000_000,
        ))

        assert widget.progress_bar().value() == 50

    def test_should_change_action_button_to_cancel_while_downloading(self, widget):
        widget.update_download(_snapshot(state="downloading"))

        assert widget.action_button().text() == "Cancel"

    def test_should_hide_progress_bar_when_snapshot_cleared(self, widget):
        widget.update_download(_snapshot(state="downloading"))

        widget.update_download(None)

        assert widget.progress_bar().isHidden() is True

    def test_should_show_error_text_when_download_failed(self, widget):
        widget.update_download(_snapshot(state="failed", error_message="bad gateway"))

        assert "bad gateway" in widget.description_label().text()


class TestModelRowWidgetSignals:
    def test_should_emit_download_requested_when_button_clicked_uninstalled(self, qtbot, widget):
        with qtbot.waitSignal(widget.download_requested, timeout=500) as sig:
            widget.action_button().click()

        assert sig.args == ["base"]

    def test_should_emit_delete_requested_when_button_clicked_installed(self, qtbot):
        w = ModelRowWidget(_view(installed=True))
        qtbot.addWidget(w)

        with qtbot.waitSignal(w.delete_requested, timeout=500) as sig:
            w.action_button().click()

        assert sig.args == ["base"]

    def test_should_emit_cancel_requested_when_button_clicked_downloading(self, qtbot, widget):
        widget.update_download(_snapshot(state="downloading"))

        with qtbot.waitSignal(widget.cancel_requested, timeout=500) as sig:
            widget.action_button().click()

        assert sig.args == ["base"]
