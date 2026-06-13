from unittest.mock import MagicMock

import pytest

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.qt_model_list_model import QtModelListModel
from voicetotext.settings.qt_pages.transcription_page import TranscriptionPage
from voicetotext.settings.settings_draft import SettingsDraft
from voicetotext.settings.transcription_mode import TranscriptionMode

_PARAKEET = "parakeet-tdt-0.6b-v3"
_NEMOTRON = "nemotron-speech-streaming-en-0.6b"
_STREAMING_CAPABLE = {_PARAKEET, _NEMOTRON}
_BATCH_CAPABLE = {"tiny", "base", _PARAKEET}


def _view(name: str) -> ModelView:
    return ModelView(
        name=name, display_size="74 MB", description=f"{name} desc", installed=True,
    )


def _draft(
    model: str = "base",
    mode: TranscriptionMode = TranscriptionMode.BATCH,
) -> SettingsDraft:
    return SettingsDraft(
        model_size=model, language="en", device="auto", engine="auto",
        hotkey_combination="ctrl+alt+m", hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE,
        transcription_mode=mode,
    )


@pytest.fixture
def service():
    svc = MagicMock()

    def list_models(mode=None):
        if mode is TranscriptionMode.STREAMING:
            return [_view(_PARAKEET), _view(_NEMOTRON)]
        return [_view("tiny"), _view("base"), _view(_PARAKEET)]

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
def page(qtbot, service):
    p = TranscriptionPage(service)
    qtbot.addWidget(p)
    return p


def _model_names(page: TranscriptionPage) -> list[str]:
    model = page.list_model()
    return [
        model.data(model.index(row), QtModelListModel.MODEL_ROLE).name
        for row in range(model.rowCount())
    ]


class TestApplyMode:
    def test_should_filter_list_to_streaming_models(self, page):
        page.apply_mode(TranscriptionMode.STREAMING)

        assert _model_names(page) == [_PARAKEET, _NEMOTRON]

    def test_should_autoselect_streaming_model_when_current_cannot_stream(
        self, page,
    ):
        page.apply_mode(TranscriptionMode.STREAMING)

        assert page.current_model() == _PARAKEET

    def test_should_lock_device_to_gpu_in_streaming(self, page):
        page.apply_mode(TranscriptionMode.STREAMING)

        assert page._device_combo.isEnabled() is False
        assert page.current_device() == "cuda"

    def test_should_restore_full_list_and_device_when_back_to_batch(self, page):
        page.apply_mode(TranscriptionMode.STREAMING)

        page.apply_mode(TranscriptionMode.BATCH)
        page.set_model("base")

        assert len(_model_names(page)) == 3
        assert page._device_combo.isEnabled() is True

    def test_should_autoselect_batch_model_when_current_is_streaming_only(
        self, qtbot, service,
    ):
        service.current_draft.return_value = _draft(
            model=_NEMOTRON, mode=TranscriptionMode.STREAMING,
        )
        service.selected_model.return_value = _NEMOTRON
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        page.apply_mode(TranscriptionMode.BATCH)

        assert page.current_model() == "tiny"  # first installed batch model

    def test_should_keep_model_that_fits_both_modes(self, qtbot, service):
        service.current_draft.return_value = _draft(model=_PARAKEET)
        service.selected_model.return_value = _PARAKEET
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        page.apply_mode(TranscriptionMode.STREAMING)

        assert page.current_model() == _PARAKEET


class TestModeSelectorOnPage:
    """The mode toggle lives directly above the model list it filters."""

    def test_should_default_selector_to_draft_mode(self, page):
        assert page.current_mode() is TranscriptionMode.BATCH

    def test_should_initialize_selector_and_list_from_streaming_draft(
        self, qtbot, service,
    ):
        service.current_draft.return_value = _draft(
            model=_PARAKEET, mode=TranscriptionMode.STREAMING,
        )
        service.selected_model.return_value = _PARAKEET
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        assert page.current_mode() is TranscriptionMode.STREAMING
        assert _model_names(page) == [_PARAKEET, _NEMOTRON]

    def test_user_toggle_should_refilter_the_list(self, page):
        button = page.mode_selector()._buttons[TranscriptionMode.STREAMING]
        button.setChecked(True)

        assert _model_names(page) == [_PARAKEET, _NEMOTRON]

    def test_user_toggle_should_update_current_mode(self, page):
        button = page.mode_selector()._buttons[TranscriptionMode.STREAMING]
        button.setChecked(True)

        assert page.current_mode() is TranscriptionMode.STREAMING

    def test_apply_mode_should_sync_the_selector(self, page):
        page.apply_mode(TranscriptionMode.STREAMING)

        assert page.mode_selector().current_mode() is TranscriptionMode.STREAMING


class TestModelAutoSwitchSignal:
    """Switching modes announces a forced model change so the user sees why."""

    def test_should_announce_switch_to_streaming_model(self, qtbot, page):
        with qtbot.waitSignal(page.model_auto_switched, timeout=500) as signal:
            page.apply_mode(TranscriptionMode.STREAMING)

        assert signal.args == ["base", _PARAKEET]

    def test_should_not_announce_when_model_fits_new_mode(
        self, qtbot, service,
    ):
        service.current_draft.return_value = _draft(model=_PARAKEET)
        service.selected_model.return_value = _PARAKEET
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        with qtbot.assertNotEmitted(page.model_auto_switched):
            page.apply_mode(TranscriptionMode.STREAMING)


class TestStreamingHardwareGate:
    """The Streaming toggle is disabled when the box has no compatible GPU."""

    def test_should_keep_streaming_enabled_when_hardware_supports_it(self, page):
        streaming = page.mode_selector()._buttons[TranscriptionMode.STREAMING]

        assert streaming.isEnabled() is True

    def test_should_disable_streaming_toggle_when_hardware_unsupported(
        self, qtbot, service,
    ):
        service.supports_streaming_hardware.return_value = False
        page = TranscriptionPage(service)
        qtbot.addWidget(page)

        streaming = page.mode_selector()._buttons[TranscriptionMode.STREAMING]
        assert streaming.isEnabled() is False
        assert streaming.toolTip() != ""
