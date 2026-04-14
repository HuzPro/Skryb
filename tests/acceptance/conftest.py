import pytest

from tests.acceptance.app_fixture import VoiceToTextApp


@pytest.fixture()
def voice_to_text_app() -> VoiceToTextApp:
    return VoiceToTextApp()
