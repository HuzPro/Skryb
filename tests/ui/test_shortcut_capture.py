from collections.abc import Iterator

from voicetotext.settings.signal import Signal
from voicetotext.ui.api_handlers import ShortcutCapture


class FakeBridge:
    def __init__(self) -> None:
        self.key_pressed = Signal()

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def press(self, *names: str) -> None:
        for name in names:
            self.key_pressed.emit(name)


def _recorder(answers: Iterator[bool]) -> tuple[FakeBridge, ShortcutCapture, list[str]]:
    bridge = FakeBridge()
    recorded: list[str] = []
    capture = ShortcutCapture(
        bridge,
        lambda event: recorded.append(str(event["keys"])),
        run_in_background=lambda action: action(),
        copilot_acts_as_right_ctrl=lambda: next(answers),
    )
    return bridge, capture, recorded


# What the recorder's hook receives for the Copilot key: Windows only swaps in
# Right Ctrl when the keys get through, and the recorder swallows them.
_COPILOT = ("cmd", "shift", "f23")


class TestRecordingTheCopilotKey:
    def test_with_the_key_acting_as_right_ctrl_copilot_then_quote_records_right_ctrl_quote(self):
        bridge, capture, recorded = _recorder(iter([True]))
        capture.start()

        bridge.press(*_COPILOT, *["f23"] * 30, "quote")

        assert recorded == ["cmd", "cmd+shift", "ctrl_r", "ctrl_r+quote"]

    def test_without_the_swap_copilot_then_quote_records_the_keys_as_sent(self):
        bridge, capture, recorded = _recorder(iter([False]))
        capture.start()

        bridge.press(*_COPILOT, "quote")

        assert recorded[-1] == "cmd+shift+f23+quote"

    def test_win_shift_and_another_key_is_not_taken_for_the_copilot_key(self):
        bridge, capture, recorded = _recorder(iter([True]))
        capture.start()

        bridge.press("cmd", "shift", "s")

        assert recorded[-1] == "cmd+shift+s"

    def test_the_windows_setting_is_read_again_for_each_recording(self):
        bridge, capture, recorded = _recorder(iter([False, True]))

        capture.start()
        bridge.press(*_COPILOT, "quote")
        capture.stop()
        capture.start()
        bridge.press(*_COPILOT, "quote")

        assert [recorded[3], recorded[-1]] == ["cmd+shift+f23+quote", "ctrl_r+quote"]

    def test_a_held_key_repeating_is_recorded_once(self):
        bridge, capture, recorded = _recorder(iter([False]))
        capture.start()

        bridge.press("ctrl", "ctrl", "ctrl", "shift", "shift", "m", "m")

        assert recorded == ["ctrl", "ctrl+shift", "ctrl+shift+m"]
