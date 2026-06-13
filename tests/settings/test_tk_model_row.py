"""Tests for :class:`TkModelRow`.

The per-row widget inside the transcription page's scrollable model list.
It owns a single action button that flips between Download/Cancel/Delete
depending on install + download state, and a progress bar that's only
shown while a download is active. Qt signals are replaced by the
callback-list :class:`Signal`.
"""

import pytest

from voicetotext.settings.download_progress import DownloadProgressSnapshot
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.tk_model_row import TkModelRow


@pytest.fixture
def root(tk_root):
    return tk_root


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
def row(root):
    widget = TkModelRow(root, _view())
    yield widget
    widget.destroy()


class TestTkModelRowRendering:
    def test_should_show_model_name(self, row):
        assert row.name_label().cget("text") == "base"

    def test_should_show_display_size(self, row):
        assert "290 MB" in row.size_label().cget("text")

    def test_should_show_description(self, row):
        assert "base description" in row.description_label().cget("text")

    def test_progress_bar_should_be_hidden_initially(self, row):
        assert row.progress_visible() is False

    def test_action_button_should_say_download_when_not_installed(self, row):
        assert row.action_button().cget("text") == "Download"

    def test_action_button_should_say_delete_when_installed(self, root):
        widget = TkModelRow(root, _view(installed=True))

        assert widget.action_button().cget("text") == "Delete"

        widget.destroy()


class TestTkModelRowUpdateView:
    def test_should_switch_to_delete_button_when_install_state_flips(self, row):
        row.update_view(_view(installed=True))

        assert row.action_button().cget("text") == "Delete"

    def test_should_switch_back_to_download_when_uninstalled(self, root):
        widget = TkModelRow(root, _view(installed=True))

        widget.update_view(_view(installed=False))

        assert widget.action_button().cget("text") == "Download"

        widget.destroy()


class TestTkModelRowUpdateDownload:
    def test_should_show_progress_bar_when_downloading(self, row):
        row.update_download(_snapshot(state="downloading", bytes_done=10_000_000))

        assert row.progress_visible() is True

    def test_should_set_progress_bar_value_from_bytes_done_and_total(self, row):
        row.update_download(_snapshot(
            state="downloading", bytes_done=50_000_000, bytes_total=100_000_000,
        ))

        assert row.progress_bar().cget("value") == 50

    def test_should_change_action_button_to_cancel_while_downloading(self, row):
        row.update_download(_snapshot(state="downloading"))

        assert row.action_button().cget("text") == "Cancel"

    def test_should_hide_progress_bar_when_snapshot_cleared(self, row):
        row.update_download(_snapshot(state="downloading"))

        row.update_download(None)

        assert row.progress_visible() is False

    def test_should_show_error_text_when_download_failed(self, row):
        row.update_download(_snapshot(state="failed", error_message="bad gateway"))

        assert "bad gateway" in row.description_label().cget("text")


class TestTkModelRowSelection:
    def test_should_apply_selected_styles_when_selected(self, row):
        row.set_selected(True)

        assert row.cget("style") == "Selected.TFrame"
        assert row.name_label().cget("style") == "Selected.TLabel"

    def test_should_revert_to_default_styles_when_deselected(self, row):
        row.set_selected(True)
        row.set_selected(False)

        assert row.cget("style") == "TFrame"
        assert row.name_label().cget("style") == "TLabel"


class TestTkModelRowCallbacks:
    def test_should_emit_download_requested_when_clicked_uninstalled(self, row):
        received: list[str] = []
        row.download_requested.connect(received.append)

        row.action_button().invoke()

        assert received == ["base"]

    def test_should_emit_delete_requested_when_clicked_installed(self, root):
        widget = TkModelRow(root, _view(installed=True))
        received: list[str] = []
        widget.delete_requested.connect(received.append)

        widget.action_button().invoke()

        assert received == ["base"]

        widget.destroy()

    def test_should_emit_cancel_requested_when_clicked_downloading(self, row):
        received: list[str] = []
        row.cancel_requested.connect(received.append)
        row.update_download(_snapshot(state="downloading"))

        row.action_button().invoke()

        assert received == ["base"]
