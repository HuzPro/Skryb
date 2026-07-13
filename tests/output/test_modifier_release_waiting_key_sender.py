"""Tests for :class:`ModifierReleaseWaitingKeySender`.

The decorator guards the synthetic Ctrl+V against the user's physically
held hotkey modifiers (e.g. Ctrl+Shift+Alt+M): injecting Ctrl+V while
Shift/Alt are still down makes the target application see
Ctrl+Shift+Alt+V and silently drop the paste.
"""

import logging

from voicetotext.output.key_sender import KeySender
from voicetotext.output.modifier_release_waiting_key_sender import (
    ModifierReleaseWaitingKeySender,
)

from tests.fakes.output import InstantSleeper


_MODULE = "voicetotext.output.modifier_release_waiting_key_sender"


class _FakeKeySender(KeySender):
    """Records each paste request so tests can assert count and ordering."""

    def __init__(self, journal: list[str] | None = None) -> None:
        self.journal = journal if journal is not None else []

    def send_ctrl_v(self) -> None:
        self.journal.append("paste")


class _FakeModifierKeyStateReader:
    """Replays a scripted sequence of "any modifier down?" answers.

    The last answer repeats once the script is exhausted, and every
    poll is recorded in the shared journal to allow ordering asserts.
    """

    def __init__(
        self, answers: list[bool], journal: list[str] | None = None,
    ) -> None:
        self._answers = list(answers)
        self.journal = journal if journal is not None else []

    def any_modifier_down(self) -> bool:
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        self.journal.append(f"poll:{'down' if answer else 'up'}")
        return answer


def _build_sender(
    answers: list[bool],
) -> tuple[ModifierReleaseWaitingKeySender, _FakeKeySender, InstantSleeper, list[str]]:
    journal: list[str] = []
    inner = _FakeKeySender(journal)
    reader = _FakeModifierKeyStateReader(answers, journal)
    sleeper = InstantSleeper()
    sender = ModifierReleaseWaitingKeySender(
        inner=inner, modifier_state=reader, sleeper=sleeper,
    )
    return sender, inner, sleeper, journal


class TestModifierReleaseWaitingKeySenderSendCtrlV:
    def test_should_paste_immediately_when_no_modifiers_are_held(self):
        sender, inner, sleeper, journal = _build_sender(answers=[False])

        sender.send_ctrl_v()

        assert journal == ["poll:up", "paste"]
        assert sleeper.requested_durations == []

    def test_should_wait_until_modifiers_are_released_before_pasting(self):
        sender, inner, sleeper, journal = _build_sender(
            answers=[True, True, False],
        )

        sender.send_ctrl_v()

        assert journal == ["poll:down", "poll:down", "poll:up", "paste"]
        assert len(sleeper.requested_durations) == 2

    def test_should_paste_anyway_when_modifiers_never_release(self):
        sender, inner, _, journal = _build_sender(answers=[True])

        sender.send_ctrl_v()

        assert journal.count("paste") == 1
        assert journal[-1] == "paste"

    def test_should_bound_the_wait_when_modifiers_never_release(self):
        sender, _, sleeper, _ = _build_sender(answers=[True])

        sender.send_ctrl_v()

        # Summing 100 x 0.01 drifts a few ULP above 1.0 on Python < 3.12
        # (3.12's sum() compensates); allow that float noise, not more.
        assert 0 < sleeper.total_requested <= 1.0 + 1e-9

    def test_should_log_warning_when_wait_times_out(self, caplog):
        sender, *_ = _build_sender(answers=[True])
        logger = logging.getLogger(_MODULE)
        logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.DEBUG)
        try:
            sender.send_ctrl_v()
        finally:
            logger.removeHandler(caplog.handler)

        assert "modifier" in caplog.text.lower()

    def test_should_not_log_when_modifiers_release_in_time(self, caplog):
        sender, *_ = _build_sender(answers=[True, False])
        logger = logging.getLogger(_MODULE)
        logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.WARNING)
        try:
            sender.send_ctrl_v()
        finally:
            logger.removeHandler(caplog.handler)

        assert caplog.text == ""


class TestModifierReleaseWaitingKeySenderIsAKeySender:
    def test_should_implement_key_sender_interface(self):
        sender, *_ = _build_sender(answers=[False])

        assert isinstance(sender, KeySender)
