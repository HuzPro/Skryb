from unittest.mock import MagicMock

import pytest

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.settings_draft import SettingsDraft
from voicetotext.settings.tk_pages.transcription_page import TkTranscriptionPage
from voicetotext.settings.transcription_mode import TranscriptionMode

_PARAKEET = "parakeet-tdt-0.6b-v3"
_NEMOTRON = "nemotron-speech-streaming-en-0.6b"
_STREAMING_CAPABLE = {_PARAKEET, _NEMOTRON}
_BATCH_CAPABLE = {"tiny", "base", _PARAKEET}


def _view(name: str, installed: bool = False) -> ModelView:
    return ModelView(
        name=name,
        display_size="74 MB",
        description=f"{name} description",
        installed=installed,
    )


def _draft(
    model: str = "base",
    language: str = "en",
    device: str = "auto",
    engine: str = "auto",
    mode: TranscriptionMode = TranscriptionMode.BATCH,
) -> SettingsDraft:
    return SettingsDraft(
        model_size=model,
        language=language,
        device=device,
        engine=engine,
        hotkey_combination="ctrl+alt+m",
        hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE,
        transcription_mode=mode,
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
    svc.supports_streaming_hardware.return_value = True
    return svc


@pytest.fixture
def streaming_service():
    svc = MagicMock()

    def list_models(mode=None):
        if mode is TranscriptionMode.STREAMING:
            return [_view(_PARAKEET, installed=True), _view(_NEMOTRON, installed=True)]
        return [
            _view("tiny", installed=True),
            _view("base", installed=True),
            _view(_PARAKEET, installed=True),
        ]

    svc.list_models.side_effect = list_models
    svc.selected_model.return_value = "base"
    svc.current_draft.return_value = _draft()
    svc.available_devices.return_value = ["auto", "cuda", "cpu"]
    svc.allows_device_selection.side_effect = lambda n: n in {"tiny", "base"}
    svc.model_supports_streaming.side_effect = lambda n: n in _STREAMING_CAPABLE
    svc.model_supports_batch.side_effect = lambda n: n in _BATCH_CAPABLE
    svc.supports_streaming_hardware.return_value = True
    svc.engine_candidates_for.return_value = ()
    return svc


@pytest.fixture
def page(tk_root, service):
    widget = TkTranscriptionPage(tk_root, service)
    yield widget
    widget.destroy()


@pytest.fixture
def streaming_page(tk_root, streaming_service):
    widget = TkTranscriptionPage(tk_root, streaming_service)
    yield widget
    widget.destroy()


class TestModel:
    def test_should_return_currently_selected_model(self, page):
        assert page.current_model() == "base"

    def test_should_update_selected_model_when_set(self, page):
        page.set_model("tiny")

        assert page.current_model() == "tiny"


class TestModelSelection:
    def test_should_select_installed_model_when_row_clicked(self, page):
        page.model_list().select_model("tiny")  # installed

        assert page.current_model() == "tiny"

    def test_should_ignore_click_on_uninstalled_model(self, page):
        page.model_list().select_model("small")  # not installed

        assert page.current_model() == "base"


class TestLanguage:
    def test_should_return_initial_language(self, page):
        assert page.current_language() == "en"

    def test_should_reflect_set_language(self, page):
        page.set_language("fr")

        assert page.current_language() == "fr"


class TestDevice:
    def test_should_return_initial_device(self, page):
        assert page.current_device() == "auto"

    def test_should_reflect_set_device(self, page):
        page.set_device("cuda")

        assert page.current_device() == "cuda"

    def test_should_populate_devices_from_service(self, tk_root, service):
        service.available_devices.return_value = ["auto", "cpu"]
        page = TkTranscriptionPage(tk_root, service)

        assert page.device_labels() == ["auto", "cpu"]

        page.destroy()

    def test_should_display_gpu_label_for_cuda_device(self, page):
        assert "GPU" in page.device_labels()

    def test_should_keep_cuda_as_internal_value_when_gpu_selected(self, page):
        page.set_device("cuda")

        assert page.current_device() == "cuda"


class TestDeviceLock:
    def test_should_enable_device_for_base(self, page):
        assert page.device_enabled() is True

    def test_should_disable_device_for_large_model(self, tk_root, service):
        service.current_draft.return_value = _draft(model="large-v3-turbo")
        page = TkTranscriptionPage(tk_root, service)

        assert page.device_enabled() is False

        page.destroy()

    def test_should_lock_device_to_auto_for_large_model(self, tk_root, service):
        service.current_draft.return_value = _draft(
            model="large-v3-turbo", device="cuda",
        )
        page = TkTranscriptionPage(tk_root, service)

        assert page.current_device() == "auto"

        page.destroy()

    def test_should_disable_when_switching_to_large_model(self, page):
        page.set_model("large-v3-turbo")

        assert page.device_enabled() is False

    def test_should_reenable_when_switching_back_to_small(self, page):
        page.set_model("large-v3-turbo")
        page.set_model("tiny")

        assert page.device_enabled() is True


class TestEngine:
    def test_should_default_to_auto_when_no_candidates(self, page):
        assert page.current_engine() == "auto"

    def test_should_offer_each_candidate_engine(self, tk_root, service):
        from voicetotext.transcription.engine_types import EngineCandidate

        service.engine_candidates_for.return_value = (
            EngineCandidate(
                engine_id="transformers", display_name="Transformers",
                score=100, is_recommended=True, reason="for ROCm",
            ),
            EngineCandidate(
                engine_id="nano_parakeet", display_name="Parakeet",
                score=50, is_recommended=False, reason="",
            ),
        )
        page = TkTranscriptionPage(tk_root, service)

        assert set(page.engine_ids()) >= {"auto", "transformers", "nano_parakeet"}

        page.destroy()

    def test_should_reflect_set_engine(self, page):
        page.set_engine("auto")

        assert page.current_engine() == "auto"


class TestDownloadActions:
    def test_should_invoke_service_start_download(self, page, service):
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

    def test_should_emit_delete_error_when_delete_raises_valueerror(
        self, page, service,
    ):
        service.delete_model.side_effect = ValueError("cannot delete active 'base'")
        received: list[tuple[str, str]] = []
        page.delete_error.connect(lambda name, err: received.append((name, err)))

        page.request_delete("base")

        assert received == [("base", "cannot delete active 'base'")]

    def test_should_swallow_pause_lookuperror(self, page, service):
        service.pause_download.side_effect = LookupError("missing")

        page.request_pause("small")

    def test_should_emit_download_error_when_start_raises(self, page, service):
        service.start_download.side_effect = ConnectionError("hf unreachable")
        received: list[tuple[str, str]] = []
        page.download_error.connect(lambda name, err: received.append((name, err)))

        page.request_download("small")

        assert received == [("small", "hf unreachable")]


class TestRowWidgets:
    def test_should_attach_a_row_per_model(self, page, service):
        for view in service.list_models.return_value:
            row = page.model_list().row_for(view.name)
            assert row is not None
            assert row.model_name() == view.name

    def test_row_download_button_should_call_service_start_download(
        self, page, service,
    ):
        page.model_list().row_for("small").action_button().invoke()

        service.start_download.assert_called_once_with("small")

    def test_row_delete_button_should_call_service_delete_model(self, page, service):
        page.model_list().row_for("tiny").action_button().invoke()

        service.delete_model.assert_called_once_with("tiny")

    def test_row_should_refresh_when_install_state_changes(self, page, service):
        row = page.model_list().row_for("small")
        assert row.action_button().cget("text") == "Download"

        service.list_models.return_value = [
            _view("tiny", installed=True),
            _view("base", installed=True),
            _view("small", installed=True),
        ]
        page.model_list().update_install_state("small")

        assert row.action_button().cget("text") == "Delete"

    def test_row_should_show_progress_when_download_progress_set(self, page):
        from voicetotext.settings.download_progress import DownloadProgressSnapshot

        page.model_list().set_download_progress(
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

        row = page.model_list().row_for("small")
        assert row.progress_visible() is True
        assert row.progress_bar().cget("value") == 10


class TestApplyMode:
    def test_should_filter_list_to_streaming_models(self, streaming_page):
        streaming_page.apply_mode(TranscriptionMode.STREAMING)

        assert streaming_page.model_list().row_names() == [_PARAKEET, _NEMOTRON]

    def test_should_autoselect_streaming_model_when_current_cannot_stream(
        self, streaming_page,
    ):
        streaming_page.apply_mode(TranscriptionMode.STREAMING)

        assert streaming_page.current_model() == _PARAKEET

    def test_should_lock_device_to_gpu_in_streaming(self, streaming_page):
        streaming_page.apply_mode(TranscriptionMode.STREAMING)

        assert streaming_page.device_enabled() is False
        assert streaming_page.current_device() == "cuda"

    def test_should_restore_full_list_and_device_when_back_to_batch(
        self, streaming_page,
    ):
        streaming_page.apply_mode(TranscriptionMode.STREAMING)

        streaming_page.apply_mode(TranscriptionMode.BATCH)
        streaming_page.set_model("base")

        assert len(streaming_page.model_list().row_names()) == 3
        assert streaming_page.device_enabled() is True

    def test_should_keep_model_that_fits_both_modes(self, tk_root, streaming_service):
        streaming_service.current_draft.return_value = _draft(model=_PARAKEET)
        streaming_service.selected_model.return_value = _PARAKEET
        page = TkTranscriptionPage(tk_root, streaming_service)

        page.apply_mode(TranscriptionMode.STREAMING)

        assert page.current_model() == _PARAKEET

        page.destroy()


class TestModeSelectorOnPage:
    def test_should_default_selector_to_draft_mode(self, streaming_page):
        assert streaming_page.current_mode() is TranscriptionMode.BATCH

    def test_user_toggle_should_refilter_the_list(self, streaming_page):
        streaming_page.mode_selector().radio_button(
            TranscriptionMode.STREAMING
        ).invoke()

        assert streaming_page.model_list().row_names() == [_PARAKEET, _NEMOTRON]

    def test_apply_mode_should_sync_the_selector(self, streaming_page):
        streaming_page.apply_mode(TranscriptionMode.STREAMING)

        assert (
            streaming_page.mode_selector().current_mode()
            is TranscriptionMode.STREAMING
        )


class TestModelAutoSwitchSignal:
    def test_should_announce_switch_to_streaming_model(self, streaming_page):
        received: list[tuple[str, str]] = []
        streaming_page.model_auto_switched.connect(
            lambda old, new: received.append((old, new))
        )

        streaming_page.apply_mode(TranscriptionMode.STREAMING)

        assert received == [("base", _PARAKEET)]


class TestStreamingHardwareGate:
    def test_should_keep_streaming_enabled_when_supported(self, streaming_page):
        assert streaming_page.mode_selector().is_streaming_enabled() is True

    def test_should_disable_streaming_when_hardware_unsupported(
        self, tk_root, streaming_service,
    ):
        streaming_service.supports_streaming_hardware.return_value = False
        page = TkTranscriptionPage(tk_root, streaming_service)

        assert page.mode_selector().is_streaming_enabled() is False

        page.destroy()
