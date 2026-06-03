from unittest.mock import MagicMock

import pytest

from voicetotext.core.exceptions import EngineSwapError
from voicetotext.transcription.engine_install import make_engine_installer


def _set_engine(use_case, engine):
    use_case.set_engine(engine)


class TestMakeEngineInstaller:
    def test_should_install_via_the_given_step(self):
        engine = MagicMock()
        use_case = MagicMock()

        make_engine_installer(engine, _set_engine)(use_case)

        use_case.set_engine.assert_called_once_with(engine)

    def test_should_dispose_engine_when_install_is_rejected(self):
        engine = MagicMock()
        use_case = MagicMock()
        use_case.set_engine.side_effect = EngineSwapError("busy")

        with pytest.raises(EngineSwapError):
            make_engine_installer(engine, _set_engine)(use_case)

        engine.dispose.assert_called_once_with()

    def test_should_not_dispose_engine_on_success(self):
        engine = MagicMock()

        make_engine_installer(engine, _set_engine)(MagicMock())

        engine.dispose.assert_not_called()
