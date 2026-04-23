from collections.abc import Iterator

import pytest

from tests.acceptance.app_fixture import VoiceToTextApp


@pytest.fixture()
def voice_to_text_app() -> Iterator[VoiceToTextApp]:
    app = VoiceToTextApp()
    try:
        yield app
    finally:
        app.dispose()
