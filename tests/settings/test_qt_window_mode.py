from unittest.mock import MagicMock

import pytest

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.qt_model_list_model import QtModelListModel
from voicetotext.settings.qt_window import QtSettingsWindow
from voicetotext.settings.settings_draft import SettingsDraft
from voicetotext.settings.transcription_mode import TranscriptionMode

_PARAKEET = "parakeet-tdt-0.6b-v3"


def _view(name: str) -> ModelView:
    return ModelView(
        name=name, display_size="74 MB", description=name, installed=True,
    )


def _draft(mode: TranscriptionMode = TranscriptionMode.BATCH) -> SettingsDraft:
    return SettingsDraft(
        model_size="base", language="en", device="auto", engine="auto",
        hotkey_combination="ctrl+alt+m", hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE, transcription_mode=mode,
    )


@pytest.fixture
def service():
    svc = MagicMock()

    def list_models(mode=None):
        if mode is TranscriptionMode.STREAMING:
            return [_view(_PARAKEET)]
        return [_view("tiny"), _view("base"), _view(_PARAKEET)]

    svc.list_models.side_effect = list_models
    svc.selected_model.return_value = "base"
    svc.current_draft.return_value = _draft()
    svc.any_model_installed.return_value = True
    svc.is_installed.return_value = True
    svc.can_apply_now.return_value = True
    svc.is_engine_swapping.return_value = False
    svc.available_devices.return_value = ["auto", "cuda", "cpu"]
    svc.allows_device_selection.side_effect = lambda n: n in {"tiny", "base"}
    svc.model_supports_streaming.side_effect = lambda n: n == _PARAKEET
    svc.model_supports_batch.return_value = True
    svc.supports_streaming_hardware.return_value = True
    svc.engine_candidates_for.return_value = ()
    return svc


@pytest.fixture
def window(qtbot, service):
    w = QtSettingsWindow(service)
    qtbot.addWidget(w)
    return w


def _model_names(window: QtSettingsWindow) -> list[str]:
    model = window.transcription_page().list_model()
    return [
        model.data(model.index(row), QtModelListModel.MODEL_ROLE).name
        for row in range(model.rowCount())
    ]


class TestModeSelection:
    """Mode lives on the Transcription page, not in a separate sidebar tab."""

    def test_should_not_show_a_mode_tab_in_the_sidebar(self, window):
        labels = [
            window._sidebar.item(i).text()
            for i in range(window._sidebar.count())
        ]

        assert labels == ["Transcription", "Hotkeys", "Output"]

    def test_should_include_mode_in_collected_draft(self, window):
        window.transcription_page().apply_mode(TranscriptionMode.STREAMING)

        assert window.collect_draft().transcription_mode is (
            TranscriptionMode.STREAMING
        )

    def test_should_set_mode_selector_from_initial_draft(self, qtbot, service):
        service.current_draft.return_value = _draft(TranscriptionMode.STREAMING)
        service.selected_model.return_value = _PARAKEET

        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        assert window.transcription_page().current_mode() is (
            TranscriptionMode.STREAMING
        )

    def test_should_refilter_model_list_when_user_toggles_streaming(self, window):
        selector = window.transcription_page().mode_selector()
        selector._buttons[TranscriptionMode.STREAMING].setChecked(True)

        assert _model_names(window) == [_PARAKEET]

    def test_should_announce_auto_switched_model_in_the_status_bar(self, window):
        selector = window.transcription_page().mode_selector()
        selector._buttons[TranscriptionMode.STREAMING].setChecked(True)

        status = window.status_label().text()
        assert _PARAKEET in status
        assert "base" in status
