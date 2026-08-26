from unittest.mock import MagicMock

import pytest

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.download_progress import DownloadProgressSnapshot
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.settings_draft import SettingsDraft
from voicetotext.settings.tk_pages.transcription_page import TkTranscriptionPage
from voicetotext.settings.transcription_mode import TranscriptionMode
from voicetotext.transcription.engine_types import EngineCandidate


_PARAKEET = "parakeet-tdt-0.6b-v3"
_NEMOTRON = "nemotron-speech-streaming-en-0.6b"
_BATCH_ONLY = "batch-only"


def _view(name: str, installed: bool = False) -> ModelView:
    return ModelView(
        name=name, display_size="2.5 GB", description=f"{name} description",
        installed=installed,
    )


def _draft(
    model: str = _PARAKEET,
    language: str = "en",
    engine: str = "auto",
    mode: TranscriptionMode = TranscriptionMode.BATCH,
) -> SettingsDraft:
    return SettingsDraft(
        model_size=model, language=language, device="auto", engine=engine,
        hotkey_combination="ctrl+alt+m", hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE, transcription_mode=mode,
    )


@pytest.fixture
def service() -> MagicMock:
    result = MagicMock()
    result.list_models.return_value = [
        _view(_PARAKEET, installed=True), _view(_NEMOTRON),
    ]
    result.selected_model.return_value = _PARAKEET
    result.current_draft.return_value = _draft()
    result.engine_candidates_for.return_value = ()
    result.supports_streaming_hardware.return_value = True
    return result


@pytest.fixture
def streaming_service() -> MagicMock:
    result = MagicMock()

    def list_models(
        mode: TranscriptionMode | None = None,
    ) -> list[ModelView]:
        if mode is TranscriptionMode.STREAMING:
            return [_view(_PARAKEET, True), _view(_NEMOTRON, True)]
        return [_view(_BATCH_ONLY, True), _view(_PARAKEET, True)]

    result.list_models.side_effect = list_models
    result.selected_model.return_value = _BATCH_ONLY
    result.current_draft.return_value = _draft(model=_BATCH_ONLY)
    result.model_supports_streaming.side_effect = lambda name: name in {
        _PARAKEET, _NEMOTRON,
    }
    result.model_supports_batch.side_effect = lambda name: name in {
        _BATCH_ONLY, _PARAKEET,
    }
    result.supports_streaming_hardware.return_value = True
    result.engine_candidates_for.return_value = ()
    return result


@pytest.fixture
def page(tk_root, service: MagicMock):
    widget = TkTranscriptionPage(tk_root, service)
    yield widget
    widget.destroy()


@pytest.fixture
def streaming_page(tk_root, streaming_service: MagicMock):
    widget = TkTranscriptionPage(tk_root, streaming_service)
    yield widget
    widget.destroy()


class TestModelAndLanguage:
    def test_should_return_and_update_model(self, page: TkTranscriptionPage) -> None:
        assert page.current_model() == _PARAKEET
        page.set_model(_NEMOTRON)
        assert page.current_model() == _NEMOTRON

    def test_should_ignore_uninstalled_model_click(
        self, page: TkTranscriptionPage,
    ) -> None:
        page.model_list().select_model(_NEMOTRON)
        assert page.current_model() == _PARAKEET

    def test_should_return_and_update_language(
        self, page: TkTranscriptionPage,
    ) -> None:
        assert page.current_language() == "en"
        page.set_language("fr")
        assert page.current_language() == "fr"


class TestEngine:
    def test_should_default_to_auto(self, page: TkTranscriptionPage) -> None:
        assert page.current_engine() == "auto"

    def test_should_offer_surviving_candidates(
        self, tk_root, service: MagicMock,
    ) -> None:
        service.engine_candidates_for.return_value = (
            EngineCandidate(
                engine_id="nano_parakeet", display_name="Parakeet", score=100,
                is_recommended=True, reason="for ROCm",
            ),
            EngineCandidate(
                engine_id="nemotron_streaming", display_name="Nemotron", score=50,
                is_recommended=False, reason="",
            ),
        )
        page = TkTranscriptionPage(tk_root, service)

        assert set(page.engine_ids()) == {
            "auto", "nano_parakeet", "nemotron_streaming",
        }

        page.destroy()


class TestDownloadActions:
    def test_should_delegate_download_lifecycle(
        self, page: TkTranscriptionPage, service: MagicMock,
    ) -> None:
        page.request_download(_NEMOTRON)
        page.request_pause(_NEMOTRON)
        page.request_resume(_NEMOTRON)
        page.request_cancel(_NEMOTRON)

        service.start_download.assert_called_once_with(_NEMOTRON)
        service.pause_download.assert_called_once_with(_NEMOTRON)
        service.resume_download.assert_called_once_with(_NEMOTRON)
        service.cancel_download.assert_called_once_with(_NEMOTRON)

    def test_should_delegate_delete(
        self, page: TkTranscriptionPage, service: MagicMock,
    ) -> None:
        page.request_delete(_PARAKEET)
        service.delete_model.assert_called_once_with(_PARAKEET)

    def test_should_emit_errors(
        self, page: TkTranscriptionPage, service: MagicMock,
    ) -> None:
        service.start_download.side_effect = ConnectionError("unreachable")
        received: list[tuple[str, str]] = []
        page.download_error.connect(
            lambda name, error: received.append((name, error)),
        )

        page.request_download(_NEMOTRON)

        assert received == [(_NEMOTRON, "unreachable")]


class TestRows:
    def test_should_refresh_install_state(
        self, page: TkTranscriptionPage, service: MagicMock,
    ) -> None:
        model_list = page.model_list()
        assert model_list.row_for(_NEMOTRON).action_button().cget("text") == (
            "Download"
        )
        service.list_models.return_value = [
            _view(_PARAKEET, True), _view(_NEMOTRON, True),
        ]

        model_list.update_install_state(_NEMOTRON)

        assert model_list.row_for(_NEMOTRON).action_button().cget("text") == (
            "Delete"
        )

    def test_should_show_progress(self, page: TkTranscriptionPage) -> None:
        page.model_list().set_download_progress(
            _NEMOTRON,
            DownloadProgressSnapshot(
                state="downloading", bytes_done=10, bytes_total=100,
                speed_bps=0.0, eta_seconds=0.0, error_message=None,
            ),
        )

        assert page.model_list().row_for(_NEMOTRON).progress_visible() is True


class TestModeSwitching:
    def test_should_filter_and_auto_select_compatible_model(
        self, streaming_page: TkTranscriptionPage,
    ) -> None:
        streaming_page.apply_mode(TranscriptionMode.STREAMING)

        assert streaming_page.model_list().row_names() == [_PARAKEET, _NEMOTRON]
        assert streaming_page.current_model() == _PARAKEET

    def test_should_announce_auto_switch(
        self, streaming_page: TkTranscriptionPage,
    ) -> None:
        received: list[tuple[str, str]] = []
        streaming_page.model_auto_switched.connect(
            lambda old, new: received.append((old, new)),
        )

        streaming_page.apply_mode(TranscriptionMode.STREAMING)

        assert received == [(_BATCH_ONLY, _PARAKEET)]

    def test_should_disable_streaming_when_hardware_is_unsupported(
        self, tk_root, streaming_service: MagicMock,
    ) -> None:
        streaming_service.supports_streaming_hardware.return_value = False
        page = TkTranscriptionPage(tk_root, streaming_service)

        assert page.mode_selector().is_streaming_enabled() is False

        page.destroy()
