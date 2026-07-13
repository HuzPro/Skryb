"""Tests for :class:`Win32ModifierKeyStateReader`.

Same fake-Win32 strategy as the other win32 adapter tests: the
``GetAsyncKeyState`` call is monkeypatched so the tests script which
virtual keys are physically held.
"""

import pytest

from voicetotext.output.modifier_key_state_reader import ModifierKeyStateReader
from voicetotext.output import win32_modifier_key_state_reader as module
from voicetotext.output.win32_modifier_key_state_reader import (
    Win32ModifierKeyStateReader,
)


_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_MENU = 0x12
_VK_LWIN = 0x5B
_VK_RWIN = 0x5C

_KEY_DOWN_FLAG = 0x8000


@pytest.fixture()
def held_keys(monkeypatch) -> set[int]:
    """Fake keyboard: virtual keys in this set report as physically down."""
    held: set[int] = set()

    def fake_get_async_key_state(virtual_key: int) -> int:
        return _KEY_DOWN_FLAG if virtual_key in held else 0

    monkeypatch.setattr(
        module._user32, "GetAsyncKeyState", fake_get_async_key_state,
    )
    return held


class TestWin32ModifierKeyStateReaderAnyModifierDown:
    def test_should_report_released_when_no_modifier_is_held(self, held_keys):
        assert Win32ModifierKeyStateReader().any_modifier_down() is False

    @pytest.mark.parametrize(
        "virtual_key",
        [_VK_SHIFT, _VK_CONTROL, _VK_MENU, _VK_LWIN, _VK_RWIN],
    )
    def test_should_report_held_when_a_modifier_is_down(
        self, held_keys, virtual_key,
    ):
        held_keys.add(virtual_key)

        assert Win32ModifierKeyStateReader().any_modifier_down() is True

    def test_should_ignore_non_modifier_keys(self, held_keys):
        held_keys.add(0x4D)  # VK 'M' — the typical hotkey trigger key

        assert Win32ModifierKeyStateReader().any_modifier_down() is False

    def test_should_ignore_the_least_significant_pressed_since_last_call_bit(
        self, held_keys, monkeypatch,
    ):
        # GetAsyncKeyState's bit 0 means "pressed since last call", not
        # "currently down" — it must not keep the paste waiting.
        monkeypatch.setattr(
            module._user32, "GetAsyncKeyState", lambda vk: 0x0001,
        )

        assert Win32ModifierKeyStateReader().any_modifier_down() is False


class TestWin32ModifierKeyStateReaderIsAModifierKeyStateReader:
    def test_should_implement_modifier_key_state_reader_interface(self):
        assert isinstance(Win32ModifierKeyStateReader(), ModifierKeyStateReader)
