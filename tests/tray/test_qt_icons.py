import pytest
from PySide6.QtGui import QIcon

from voicetotext.core.state_machine import AppState
from voicetotext.tray.qt_icons import create_qicon


@pytest.fixture(autouse=True)
def _qapp(qtbot):
    # QIcon requires a QApplication; qtbot provides one.
    return qtbot


class TestCreateQIconReturnsIcon:
    @pytest.mark.parametrize("state", list(AppState))
    def test_should_return_non_null_icon_for_every_state(self, state):
        icon = create_qicon(state)

        assert isinstance(icon, QIcon)
        assert not icon.isNull()

    def test_should_return_different_pixmap_bytes_for_different_states(self):
        idle = create_qicon(AppState.IDLE).pixmap(64, 64).toImage()
        recording = create_qicon(AppState.RECORDING).pixmap(64, 64).toImage()

        assert idle != recording
