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
    svc.allows_device_selection.side_effect = lambda n: n in {"tiny", "base"}
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

    def test_should_display_gpu_label_for_cuda_device(self, page):
        idx = page._device_combo.findData("cuda")

        assert idx != -1
        assert page._device_combo.itemText(idx) == "GPU"

    def test_should_keep_cuda_as_internal_value_when_gpu_selected(self, page):
        page._device_combo.setCurrentIndex(page._device_combo.findData("cuda"))

        assert page.current_device() == "cuda"


class TestTranscriptionPageDeviceLock:
    def test_should_enable_device_combo_for_base(self, page):
        assert page._device_combo.isEnabled() is True

    def test_should_enable_device_combo_for_tiny(self, qtbot, service):
        service.current_draft.return_value = _draft(model="tiny")
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        assert page._device_combo.isEnabled() is True

    def test_should_disable_device_combo_for_large_model(self, qtbot, service):
        service.current_draft.return_value = _draft(model="large-v3-turbo")
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        assert page._device_combo.isEnabled() is False

    def test_should_lock_device_to_auto_for_large_model(self, qtbot, service):
        service.current_draft.return_value = _draft(
            model="large-v3-turbo", device="cuda",
        )
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        assert page.current_device() == "auto"

    def test_should_disable_when_switching_to_large_model(self, page):
        page.set_model("large-v3-turbo")

        assert page._device_combo.isEnabled() is False

    def test_should_reenable_when_switching_back_to_small(self, page):
        page.set_model("large-v3-turbo")
        page.set_model("tiny")

        assert page._device_combo.isEnabled() is True


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

    def test_should_emit_delete_error_when_delete_model_raises_valueerror(
        self, qtbot, page, service,
    ):
        service.delete_model.side_effect = ValueError(
            "cannot delete active model 'base'"
        )

        with qtbot.waitSignal(page.delete_error, timeout=500) as sig:
            page.request_delete("base")

        assert sig.args == ["base", "cannot delete active model 'base'"]

    def test_should_swallow_pause_lookuperror(self, page, service):
        service.pause_download.side_effect = LookupError("missing")

        page.request_pause("small")

    def test_should_emit_download_error_when_start_download_raises(
        self, qtbot, page, service,
    ):
        service.start_download.side_effect = ConnectionError("hf unreachable")

        with qtbot.waitSignal(page.download_error, timeout=500) as sig:
            page.request_download("small")

        assert sig.args == ["small", "hf unreachable"]


class TestTranscriptionPageRowWidgets:
    def test_should_attach_a_row_widget_per_model(self, page, service):
        from voicetotext.settings.qt_model_row import ModelRowWidget

        for view in service.list_models.return_value:
            index = page.list_model().index_for_name(view.name)
            widget = page.list_view().indexWidget(index)
            assert isinstance(widget, ModelRowWidget)
            assert widget.model_name() == view.name

    def test_row_widget_download_button_should_call_service_start_download(
        self, page, service,
    ):
        index = page.list_model().index_for_name("small")
        widget = page.list_view().indexWidget(index)

        widget.action_button().click()

        service.start_download.assert_called_once_with("small")

    def test_row_widget_delete_button_should_call_service_delete_model(
        self, page, service,
    ):
        index = page.list_model().index_for_name("tiny")
        widget = page.list_view().indexWidget(index)

        widget.action_button().click()

        service.delete_model.assert_called_once_with("tiny")

    def test_row_widget_should_refresh_when_install_state_changes(
        self, page, service,
    ):
        # Start: "small" is not installed → button says "Download"
        index = page.list_model().index_for_name("small")
        widget = page.list_view().indexWidget(index)
        assert widget.action_button().text() == "Download"

        # Service now reports "small" as installed
        service.list_models.return_value = [
            _view("tiny", installed=True),
            _view("base", installed=True),
            _view("small", installed=True),
        ]
        page.list_model().update_install_state("small")

        widget = page.list_view().indexWidget(index)
        assert widget.action_button().text() == "Delete"

    def test_row_widget_should_show_progress_bar_when_download_progress_set(
        self, page,
    ):
        from voicetotext.settings.qt_model_list_model import (
            DownloadProgressSnapshot,
        )

        index = page.list_model().index_for_name("small")
        widget = page.list_view().indexWidget(index)

        page.list_model().set_download_progress(
            "small",
            DownloadProgressSnapshot(
                state="downloading",
                bytes_done=10_000_000,
                bytes_total=100_000_000,
                speed_bps=0.0,
                eta_seconds=0.0,
                error_message=None,
            ),
        )

        assert widget.progress_bar().isHidden() is False
        assert widget.progress_bar().value() == 10
