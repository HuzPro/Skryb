from unittest.mock import MagicMock

import pytest

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.qt_model_list_model import QtModelListModel
from voicetotext.settings.qt_pages.transcription_page import TranscriptionPage
from voicetotext.settings.settings_draft import SettingsDraft
from voicetotext.settings.transcription_mode import TranscriptionMode

_PARAKEET = "parakeet-tdt-0.6b-v3"


def _view(name: str) -> ModelView:
    return ModelView(
        name=name, display_size="74 MB", description=f"{name} desc", installed=True,
    )


def _draft() -> SettingsDraft:
    return SettingsDraft(
        model_size="base", language="en", device="auto", engine="auto",
        hotkey_combination="ctrl+alt+m", hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE,
        transcription_mode=TranscriptionMode.BATCH,
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
    svc.available_devices.return_value = ["auto", "cuda", "cpu"]
    svc.allows_device_selection.side_effect = lambda n: n in {"tiny", "base"}
    svc.model_supports_streaming.side_effect = lambda n: n == _PARAKEET
    svc.engine_candidates_for.return_value = ()
    return svc


def _model_names(page: TranscriptionPage) -> list[str]:
    model = page.list_model()
    return [
        model.data(model.index(row), QtModelListModel.MODEL_ROLE).name
        for row in range(model.rowCount())
    ]


class TestApplyMode:
    def test_should_filter_list_to_streaming_models(self, qtbot, service):
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        page.apply_mode(TranscriptionMode.STREAMING)

        assert _model_names(page) == [_PARAKEET]

    def test_should_autoselect_parakeet_when_current_cannot_stream(
        self, qtbot, service,
    ):
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        page.apply_mode(TranscriptionMode.STREAMING)

        assert page.current_model() == _PARAKEET

    def test_should_lock_device_to_gpu_in_streaming(self, qtbot, service):
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        page.apply_mode(TranscriptionMode.STREAMING)

        assert page._device_combo.isEnabled() is False
        assert page.current_device() == "cuda"

    def test_should_restore_full_list_and_device_when_back_to_batch(
        self, qtbot, service,
    ):
        page = TranscriptionPage(service)
        qtbot.addWidget(page)
        page.apply_mode(TranscriptionMode.STREAMING)

        page.apply_mode(TranscriptionMode.BATCH)
        page.set_model("base")

        assert len(_model_names(page)) == 3
        assert page._device_combo.isEnabled() is True
