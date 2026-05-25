import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from voicetotext.core.state_machine import AppState
from voicetotext.tray.qt_icons import create_qicon, render_qicon_for_scheme


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


class TestRenderQIconForScheme:
    @pytest.mark.parametrize("state", list(AppState))
    def test_should_render_different_colours_for_dark_vs_light(self, state):
        # Fixed hex palettes designed for one theme sink into the
        # other — grey on a light taskbar, dark blue on a dark
        # taskbar. Picking per-scheme palettes is the minimum a
        # tray icon owes its user.
        light = render_qicon_for_scheme(
            state, Qt.ColorScheme.Light,
        ).pixmap(64, 64).toImage()
        dark = render_qicon_for_scheme(
            state, Qt.ColorScheme.Dark,
        ).pixmap(64, 64).toImage()

        assert light != dark

    def test_should_treat_unknown_scheme_as_light(self):
        # Platforms that don't expose a preference (older Linux DEs,
        # some test harnesses) report ``Unknown``; we render the
        # light palette so the icon is visible on the default
        # Windows / macOS taskbars.
        unknown = render_qicon_for_scheme(
            AppState.RECORDING, Qt.ColorScheme.Unknown,
        ).pixmap(64, 64).toImage()
        light = render_qicon_for_scheme(
            AppState.RECORDING, Qt.ColorScheme.Light,
        ).pixmap(64, 64).toImage()

        assert unknown == light
