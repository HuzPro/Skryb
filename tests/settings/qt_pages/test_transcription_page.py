from unittest.mock import MagicMock

import pytest

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


def _draft(
    model: str = "base", language: str = "en", device: str = "auto",
    engine: str = "auto",
) -> SettingsDraft:
    from voicetotext.output.output_mode import OutputMode

    return SettingsDraft(
        model_size=model,
        language=language,
        device=device,
        engine=engine,
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
    svc.available_devices.return_value = ["auto", "cuda", "cpu"]
    svc.engine_candidates_for.return_value = ()
    return svc


@pytest.fixture
def page(qtbot, service):
    p = TranscriptionPage(service)
    qtbot.addWidget(p)
    return p


class TestTranscriptionPageModel:
    def test_should_return_currently_selected_model(self, page):
        assert page.current_model() == "base"

    def test_should_update_selected_model_when_set(self, page):
        page.set_model("tiny")

        assert page.current_model() == "tiny"


class TestTranscriptionPageLanguage:
    def test_should_return_initial_language(self, page):
        assert page.current_language() == "en"

    def test_should_reflect_set_language(self, page):
        page.set_language("fr")

        assert page.current_language() == "fr"


class TestTranscriptionPageDevice:
    def test_should_return_initial_device(self, page):
        assert page.current_device() == "auto"

    def test_should_reflect_set_device(self, page):
        page.set_device("cuda")

        assert page.current_device() == "cuda"

    def test_should_populate_devices_from_service(self, qtbot, service):
        service.available_devices.return_value = ["auto", "cpu"]
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        assert page._device_combo.count() == 2
        items = [page._device_combo.itemText(i) for i in range(2)]
        assert items == ["auto", "cpu"]


class TestTranscriptionPageEngine:
    def test_should_default_to_auto_when_no_candidates(self, page):
        assert page.current_engine() == "auto"

    def test_should_offer_each_candidate_engine(self, qtbot, service):
        from voicetotext.transcription.engine_types import EngineCandidate

        service.engine_candidates_for.return_value = (
            EngineCandidate(
                engine_id="faster_whisper", display_name="faster-whisper",
                score=100, is_recommended=True, reason="for ROCm",
            ),
            EngineCandidate(
                engine_id="transformers", display_name="Transformers",
                score=50, is_recommended=False, reason="",
            ),
        )
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        engine_ids = [
            page._engine_combo.itemData(i)
            for i in range(page._engine_combo.count())
        ]
        assert "auto" in engine_ids
        assert "faster_whisper" in engine_ids
        assert "transformers" in engine_ids

    def test_should_reflect_set_engine(self, page):
        page.set_engine("auto")

        assert page.current_engine() == "auto"


class TestTranscriptionPageDownloadActions:
    def test_should_invoke_service_start_download_when_download_requested(
        self, page, service
    ):
        page.request_download("small")

        service.start_download.assert_called_once_with("small")

    def test_should_invoke_service_pause_download(self, page, service):
        page.request_pause("small")

        service.pause_download.assert_called_once_with("small")

    def test_should_invoke_service_resume_download(self, page, service):
        page.request_resume("small")

        service.resume_download.assert_called_once_with("small")

    def test_should_invoke_service_cancel_download(self, page, service):
        page.request_cancel("small")

        service.cancel_download.assert_called_once_with("small")

    def test_should_invoke_service_delete_model(self, page, service):
        page.request_delete("tiny")

        service.delete_model.assert_called_once_with("tiny")

    def test_should_swallow_delete_valueerror_to_avoid_ui_crash(self, page, service):
        service.delete_model.side_effect = ValueError("active")

        page.request_delete("base")

    def test_should_swallow_pause_lookuperror(self, page, service):
        service.pause_download.side_effect = LookupError("missing")

        page.request_pause("small")
