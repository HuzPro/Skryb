from collections.abc import Iterator

import pytest

from tests.acceptance.app_fixture import SkrybApp


@pytest.fixture()
def voice_to_text_app() -> Iterator[SkrybApp]:
    app = SkrybApp()
    try:
        yield app
    finally:
        app.dispose()
