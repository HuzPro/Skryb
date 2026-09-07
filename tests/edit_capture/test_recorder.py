from datetime import UTC, datetime, timedelta

import pytest

from voicetotext.edit_capture.recorder import (
    DELIVERY_TIMEOUT,
    IDLE_TIMEOUT,
    EditRecorder,
)
from voicetotext.edit_capture.replay import KeyStroke
from voicetotext.edit_capture.window import ForegroundWindow

pytestmark = pytest.mark.unit

_VK_RETURN = 0x0D
_VK_BACK = 0x08
_CHAT = ForegroundWindow(hwnd=11, pid=100, process="claude.exe", title="Claude")
_OTHER = ForegroundWindow(hwnd=22, pid=200, process="discord.exe", title="Discord")
_TERMINAL = ForegroundWindow(
    hwnd=33, pid=300, process="windowsterminal.exe", title="pwsh",
)


class _Clock:
    def __init__(self) -> None:
        self._at = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self._at

    def advance(self, amount: timedelta) -> None:
        self._at += amount


def _typing(text: str) -> list[KeyStroke]:
    return [KeyStroke(vk=ord(char.upper()), char=char) for char in text]


class _Recording:
    def __init__(self) -> None:
        self.clock = _Clock()
        self.events: list[dict[str, object]] = []
        self.recorder = EditRecorder(self.events.append, self.clock)

    def of_kind(self, event: str) -> list[dict[str, object]]:
        return [item for item in self.events if item["event"] == event]

    def paste(self, text: str, window: ForegroundWindow = _CHAT) -> None:
        self.recorder.delivered(text, "batch", "direct_paste")
        self.recorder.pasted(window)

    def type_text(self, text: str, window: ForegroundWindow = _CHAT) -> None:
        for stroke in _typing(text):
            self.recorder.keyed(stroke, window.hwnd)

    def press(self, vk: int, window: ForegroundWindow = _CHAT) -> None:
        self.recorder.keyed(KeyStroke(vk=vk, char=""), window.hwnd)


class TestOpeningABlock:
    def test_should_record_the_delivered_text_before_the_paste(self):
        recording = _Recording()

        recording.paste("hello there")

        assert recording.of_kind("delivered")[0]["text"] == "hello there"
        assert recording.of_kind("paste")[0]["process"] == "claude.exe"

    def test_should_watch_the_window_the_paste_landed_in(self):
        recording = _Recording()

        recording.paste("hello")

        assert recording.recorder.watching_hwnd == _CHAT.hwnd

    def test_should_ignore_a_paste_with_no_delivery_behind_it(self):
        recording = _Recording()

        recording.recorder.pasted(_CHAT)

        assert recording.events == []
        assert recording.recorder.watching_hwnd == 0

    def test_should_drop_a_delivery_that_never_became_a_paste(self):
        recording = _Recording()
        recording.recorder.delivered("stale", "batch", "clipboard_only")
        recording.clock.advance(DELIVERY_TIMEOUT * 2)

        recording.recorder.pasted(_CHAT)

        assert recording.of_kind("paste") == []
        assert recording.recorder.watching_hwnd == 0

    def test_should_refuse_to_watch_a_denied_process(self):
        recording = _Recording()

        recording.paste("secret", window=_TERMINAL)

        assert recording.of_kind("paste") == []
        assert recording.of_kind("skipped")[0]["process"] == "windowsterminal.exe"
        assert recording.recorder.watching_hwnd == 0


class TestRecordingEdits:
    def test_should_ignore_keys_typed_in_another_window(self):
        recording = _Recording()
        recording.paste("hello")

        recording.type_text("xyz", window=_OTHER)

        assert recording.of_kind("key") == []

    def test_should_report_the_edited_text_when_enter_is_pressed(self):
        recording = _Recording()
        recording.paste("helo")
        recording.press(_VK_BACK)
        recording.type_text("lo")

        recording.press(_VK_RETURN)

        sent = recording.of_kind("sent")[0]
        assert sent["text"] == "hello"
        assert sent["reason"] == "enter"
        assert sent["spans"] == [
            {"seq": 1, "pasted_text": "helo", "final_text": "hello"},
        ]

    def test_should_report_an_untouched_dictation_unchanged(self):
        recording = _Recording()
        recording.paste("no edits needed")

        recording.press(_VK_RETURN)

        assert recording.of_kind("sent")[0]["text"] == "no edits needed"

    def test_should_stop_watching_once_the_text_is_sent(self):
        recording = _Recording()
        recording.paste("hello")

        recording.press(_VK_RETURN)

        assert recording.recorder.watching_hwnd == 0

    def test_should_attribute_each_dictation_of_a_block_to_its_own_span(self):
        recording = _Recording()
        recording.paste("first sentence. ")
        recording.paste("second sentence.")

        recording.press(_VK_RETURN)

        spans = recording.of_kind("sent")[0]["spans"]
        assert [span["seq"] for span in spans] == [1, 2]
        assert spans[0]["pasted_text"] == "first sentence. "
        assert spans[1]["pasted_text"] == "second sentence."

    def test_should_mark_a_block_unreliable_after_a_mouse_click(self):
        recording = _Recording()
        recording.paste("hello")

        recording.recorder.keyed(
            KeyStroke(vk=ord("X"), char="x", click_since_last=True), _CHAT.hwnd,
        )
        recording.press(_VK_RETURN)

        sent = recording.of_kind("sent")[0]
        assert sent["reliable"] is False
        assert "mouse_click" in sent["flags"]


class TestClosingABlock:
    def test_should_close_the_previous_block_when_a_paste_lands_elsewhere(self):
        recording = _Recording()
        recording.paste("first")

        recording.paste("second", window=_OTHER)

        sent = recording.of_kind("sent")[0]
        assert sent["reason"] == "paste_elsewhere"
        assert sent["text"] == "first"
        assert recording.recorder.watching_hwnd == _OTHER.hwnd

    def test_should_close_a_block_left_untouched_for_too_long(self):
        recording = _Recording()
        recording.paste("abandoned")
        recording.clock.advance(IDLE_TIMEOUT * 2)

        recording.recorder.expire_idle_block()

        assert recording.of_kind("sent")[0]["reason"] == "idle"

    def test_should_keep_a_block_that_is_still_recent(self):
        recording = _Recording()
        recording.paste("still going")

        recording.recorder.expire_idle_block()

        assert recording.of_kind("sent") == []

    def test_should_flush_an_open_block_on_shutdown(self):
        recording = _Recording()
        recording.paste("unsent draft")

        recording.recorder.dispose()

        assert recording.of_kind("sent")[0]["reason"] == "shutdown"

    def test_should_write_nothing_on_shutdown_without_a_block(self):
        recording = _Recording()

        recording.recorder.dispose()

        assert recording.events == []
