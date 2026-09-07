from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from voicetotext.edit_capture import observer as observer_module
from voicetotext.edit_capture.observer import Win32EditObserver
from voicetotext.edit_capture.recorder import EditRecorder
from voicetotext.edit_capture.window import ForegroundWindow
from voicetotext.output.win32_key_sender import VOICETOTEXT_INPUT_TAG

pytestmark = pytest.mark.unit

_CHAT = ForegroundWindow(hwnd=11, pid=100, process="claude.exe", title="Claude")
_VK_CONTROL = 0xA2
_VK_SHIFT = 0xA0
_VK_V = 0x56
_VK_RETURN = 0x0D
_UNTAGGED = 0
_LLKHF_INJECTED = 0x10


def _fake_character(
    vk: int, shift: bool, ctrl: bool, alt: bool, *, caps_lock: bool = False,
) -> str:
    if ctrl or alt or not chr(vk).isalpha():
        return ""
    return chr(vk) if shift != caps_lock else chr(vk).lower()


class _Capture:
    """Drives a real observer and recorder with the Win32 calls stubbed out.

    ``stop`` drains the queue before it joins, so every event enqueued
    beforehand has been handled by the time the assertions run.
    """

    def __init__(self, *, caps_lock: bool = False) -> None:
        self.events: list[dict[str, object]] = []
        recorder = EditRecorder(
            self.events.append, lambda: datetime(2026, 9, 7, tzinfo=UTC),
        )
        with patch.object(observer_module, "caps_lock_enabled", return_value=caps_lock):
            self.observer = Win32EditObserver(recorder)
        self._patches = patch.multiple(
            observer_module,
            foreground_window=lambda: _CHAT,
            foreground_hwnd=lambda: _CHAT.hwnd,
            key_character=_fake_character,
            clicked_since_last_check=lambda: False,
        )

    def __enter__(self) -> "_Capture":
        self._patches.start()
        self.observer.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.observer.stop()
        self._patches.stop()

    def paste(self, text: str) -> None:
        self.observer.observe_delivery(text, "batch", "direct_paste")
        self.observer.observe_key(True, _VK_V, 0, VOICETOTEXT_INPUT_TAG)

    def press(self, vk: int, down: bool = True) -> None:
        self.observer.observe_key(down, vk, 0, _UNTAGGED)

    def press_injected(self, vk: int) -> None:
        self.observer.observe_key(True, vk, _LLKHF_INJECTED, _UNTAGGED)

    def type_text(self, text: str) -> None:
        for char in text:
            self.press(ord(char.upper()))

    def of_kind(self, event: str) -> list[dict[str, object]]:
        return [item for item in self.events if item["event"] == event]


class TestWin32EditObserver:
    def test_should_separate_pastes_after_a_click_in_the_same_window(self) -> None:
        with _Capture() as capture:
            with patch.object(
                observer_module, "clicked_since_last_check", side_effect=[False, True],
            ):
                capture.paste("first field")
                capture.paste("second field")
                capture.observer.stop()

        sent = capture.of_kind("sent")
        assert [event["text"] for event in sent] == ["first field", "second field"]
        assert sent[0]["reliable"] is False

    def test_should_preserve_caps_lock_enabled_before_capture_starts(self) -> None:
        with _Capture(caps_lock=True) as capture:
            capture.paste("hello ")
            capture.type_text("a")
            capture.press(_VK_SHIFT)
            capture.type_text("b")
            capture.press(_VK_SHIFT, down=False)
            capture.press(_VK_RETURN)

        assert capture.of_kind("sent")[0]["text"] == "hello Ab"

    def test_should_toggle_caps_lock_once_per_press(self) -> None:
        with _Capture() as capture:
            capture.paste("hello ")
            capture.press(observer_module._VK_CAPITAL)
            capture.press(observer_module._VK_CAPITAL)
            capture.type_text("a")
            capture.press(observer_module._VK_CAPITAL, down=False)
            capture.press(observer_module._VK_CAPITAL)
            capture.type_text("b")
            capture.press(_VK_RETURN)

        assert capture.of_kind("sent")[0]["text"] == "hello Ab"

    def test_should_stop_when_typing_moves_to_another_window(self) -> None:
        with _Capture() as capture:
            with patch.object(observer_module, "foreground_hwnd", return_value=22):
                capture.paste("hello")
                capture.type_text("elsewhere")
                capture.observer.stop()

        assert capture.of_kind("key") == []
        assert capture.of_kind("sent")[0]["flags"] == ["focus_changed"]

    def test_should_stop_before_recording_a_key_after_a_mouse_click(self) -> None:
        with _Capture() as capture:
            with patch.object(
                observer_module, "clicked_since_last_check", side_effect=[False, True],
            ):
                capture.paste("hello")
                capture.type_text("private")
                capture.press(_VK_RETURN)
                capture.observer.stop()

        assert capture.of_kind("key") == []
        assert capture.of_kind("sent")[0]["reason"] == "focus_left"
        assert capture.of_kind("sent")[0]["reliable"] is False

    def test_should_stop_after_foreign_injected_input(self) -> None:
        with _Capture() as capture:
            capture.paste("hello")
            capture.press_injected(ord("A"))
            capture.type_text("private")
            capture.press(_VK_RETURN)

        assert capture.of_kind("key") == []
        assert capture.of_kind("sent")[0]["reliable"] is False

    def test_should_open_a_block_on_our_own_injected_paste(self):
        with _Capture() as capture:
            capture.paste("hello")

        assert capture.of_kind("paste")[0]["process"] == "claude.exe"

    def test_should_record_typing_that_follows_the_paste(self):
        with _Capture() as capture:
            capture.paste("helo")
            capture.type_text("x")

        assert capture.of_kind("key")[0]["char"] == "x"

    def test_should_apply_the_held_shift_to_the_typed_character(self):
        with _Capture() as capture:
            capture.paste("hi")
            capture.press(_VK_SHIFT)
            capture.type_text("a")

        assert capture.of_kind("key")[0]["char"] == "A"

    def test_should_forget_a_modifier_once_it_is_released(self):
        with _Capture() as capture:
            capture.paste("hi")
            capture.press(_VK_SHIFT)
            capture.press(_VK_SHIFT, down=False)
            capture.type_text("a")

        assert capture.of_kind("key")[0]["char"] == "a"

    def test_should_not_let_our_injected_control_key_leak_into_the_user_state(self):
        with _Capture() as capture:
            capture.observer.observe_key(
                True, _VK_CONTROL, 0, VOICETOTEXT_INPUT_TAG,
            )
            capture.paste("hi")
            capture.type_text("a")

        assert capture.of_kind("key")[0]["ctrl"] is False

    def test_should_ignore_keystrokes_injected_by_another_program(self):
        # Password-manager auto-type and macro tools inject keys the user
        # never typed. Recording them would write a password to the log.
        with _Capture() as capture:
            capture.paste("hi")
            capture.press_injected(ord("A"))

        assert capture.of_kind("key") == []

    def test_should_ignore_keys_typed_while_no_dictation_is_open(self):
        with _Capture() as capture:
            capture.type_text("private note")

        assert capture.events == []

    def test_should_report_the_edited_text_when_enter_is_pressed(self):
        with _Capture() as capture:
            capture.paste("hell")
            capture.type_text("o")
            capture.press(_VK_RETURN)

        sent = capture.of_kind("sent")[0]
        assert sent["text"] == "hello"
        assert sent["reason"] == "enter"

    def test_should_flush_an_unsent_dictation_when_the_app_stops(self):
        with _Capture() as capture:
            capture.paste("draft")

        assert capture.of_kind("sent")[0]["reason"] == "shutdown"
