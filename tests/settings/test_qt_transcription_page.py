from unittest.mock import MagicMock

import pytest

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.qt_pages.transcription_page import TranscriptionPage
from voicetotext.settings.settings_draft import SettingsDraft


def _view(name: str, installed: bool = False) -> ModelView:
    return ModelView(
        name=name,
        display_size="74 MB",
        description=f"{name} description",
        installed=installed,
    )


def _draft(model_size: str = "base") -> SettingsDraft:
    return SettingsDraft(
        model_size=model_size,
        language="en",
        device="cuda",
        hotkey_combination="ctrl+alt+m",
        hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE,
    )


@pytest.fixture
def service():
    svc = MagicMock()
    svc.list_models.return_value = [
        _view("tiny", installed=True),
        _view("base", installed=True),
        _view("small"),
    ]
    svc.selected_model.return_value = "base"
    svc.current_draft.return_value = _draft()
    return svc


class TestTranscriptionPageInitialSelection:
    def test_should_highlight_currently_loaded_model_on_open(self, qtbot, service):
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        current = page.list_view().currentIndex()

        assert current.isValid()
        assert current.row() == 1  # "base"

    def test_should_move_view_selection_when_set_model_is_called(self, qtbot, service):
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        page.set_model("tiny")

        current = page.list_view().currentIndex()
        assert current.isValid()
        assert current.row() == 0
