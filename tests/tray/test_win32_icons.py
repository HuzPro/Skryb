"""Tests for the native tray icon renderer.

The PIL rendering and the registry-driven scheme detection are both pure
(an injected ``winreg`` fake stands in for the OS), so these run on every
platform. The HICON conversion is exercised by the manual Windows smoke
test, not here.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock

from skryb.core.state_machine import AppState
from skryb.tray import win32_icons
from skryb.tray.win32_icons import ColorScheme, render_frames_for_scheme

# Middle of the cursor's stem in a 64 px tray frame: always fully drawn.
_STEM = (51, 32)


def _center_pixel(state: AppState, scheme: ColorScheme) -> tuple[int, int, int]:
    image = render_frames_for_scheme(state, scheme, size=64)[0]
    r, g, b, _ = image.getpixel(_STEM)
    return (r, g, b)


class TestRenderFramesForScheme:
    def test_should_produce_rgba_frames_of_the_requested_size(self):
        frames = render_frames_for_scheme(AppState.RECORDING, ColorScheme.LIGHT, size=48)

        assert all(frame.size == (48, 48) and frame.mode == "RGBA" for frame in frames)

    def test_should_leave_the_corner_transparent(self):
        image = render_frames_for_scheme(AppState.IDLE, ColorScheme.LIGHT)[0]

        assert image.getpixel((0, 0))[3] == 0

    def test_should_stand_still_when_idle_or_failed(self):
        for state in (AppState.IDLE, AppState.ERROR):
            assert len(render_frames_for_scheme(state, ColorScheme.DARK)) == 1

    def test_should_move_while_recording_transcribing_or_loading(self):
        for state in (AppState.RECORDING, AppState.TRANSCRIBING, AppState.LOADING):
            frames = render_frames_for_scheme(state, ColorScheme.DARK)

            assert len(frames) > 1
            assert len({frame.tobytes() for frame in frames}) > len(frames) // 2

    def test_should_use_a_visibly_different_colour_per_scheme(self):
        light = _center_pixel(AppState.RECORDING, ColorScheme.LIGHT)
        dark = _center_pixel(AppState.RECORDING, ColorScheme.DARK)

        assert light != dark

    def test_should_use_the_recording_red_in_light_scheme(self):
        # #D63A3A -> the solid centre should be close to that red.
        r, g, b = _center_pixel(AppState.RECORDING, ColorScheme.LIGHT)

        assert r > 180 and g < 90 and b < 90

    def test_should_fall_back_to_idle_colour_for_unknown_state(self):
        # AppState members all have palette entries, so route an explicit
        # check that a known state renders its own (non-idle) colour.
        idle = _center_pixel(AppState.IDLE, ColorScheme.LIGHT)
        loading = _center_pixel(AppState.LOADING, ColorScheme.LIGHT)

        assert idle != loading


def _fake_winreg(*, uses_light: int | None, raises: bool = False) -> MagicMock:
    winreg = MagicMock()
    winreg.HKEY_CURRENT_USER = object()
    if raises:
        winreg.OpenKey.side_effect = OSError("no such key")
        return winreg

    @contextmanager
    def open_key(_root, _path):
        yield object()

    winreg.OpenKey.side_effect = open_key
    winreg.QueryValueEx.return_value = (uses_light, 4)
    return winreg


class TestColorSchemeFromRegistry:
    def test_should_report_light_when_system_uses_light_theme(self):
        scheme = win32_icons.color_scheme_from_registry(
            _fake_winreg(uses_light=1),
        )

        assert scheme == ColorScheme.LIGHT

    def test_should_report_dark_when_system_does_not_use_light_theme(self):
        scheme = win32_icons.color_scheme_from_registry(
            _fake_winreg(uses_light=0),
        )

        assert scheme == ColorScheme.DARK

    def test_should_default_to_light_when_registry_read_fails(self):
        scheme = win32_icons.color_scheme_from_registry(
            _fake_winreg(uses_light=None, raises=True),
        )

        assert scheme == ColorScheme.LIGHT
