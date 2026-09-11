from voicetotext.output.caps_lock_uppercase_decorator import (
    CapsLockUppercaseDecorator,
)
from voicetotext.output.handler import TextOutputHandler


class _RecordingInner(TextOutputHandler):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def output(self, text: str) -> None:
        self.calls.append(text)


class TestCapsLockUppercaseDecorator:
    def test_uppercases_when_caps_lock_is_on(self) -> None:
        inner = _RecordingInner()
        handler = CapsLockUppercaseDecorator(inner, lambda: True)

        handler.output("Hello, world.")

        assert inner.calls == ["HELLO, WORLD."]

    def test_passes_text_through_when_caps_lock_is_off(self) -> None:
        inner = _RecordingInner()
        handler = CapsLockUppercaseDecorator(inner, lambda: False)

        handler.output("Hello, world.")

        assert inner.calls == ["Hello, world."]

    def test_reads_caps_lock_at_output_time(self) -> None:
        inner = _RecordingInner()
        caps_lock = [False]
        handler = CapsLockUppercaseDecorator(inner, lambda: caps_lock[0])

        handler.output("one")
        caps_lock[0] = True
        handler.output("two")

        assert inner.calls == ["one", "TWO"]
