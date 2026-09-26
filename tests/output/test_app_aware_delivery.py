from collections.abc import Callable

import pytest

from tests.fakes import FakeTextOutputHandler
from voicetotext.output.app_aware_delivery import (
    AppAwareDelivery,
    DeliveryFailure,
    paste_landed,
)
from voicetotext.output.caret_context import NO_CONTEXT, CaretContext
from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.output.last_dictation import LastDictation, LastDictationDecorator


class InlineGui(GuiThreadDispatcher):
    def __init__(self) -> None:
        self.dispatched = 0

    def dispatch(self, action: Callable[[], None]) -> None:
        self.dispatched += 1
        action()


class Harness:
    """An app-aware delivery wired to fakes, running every step inline."""

    def __init__(
        self,
        *,
        text_field: bool = True,
        contexts: list[CaretContext] | None = None,
        foregrounds: list[int] | None = None,
    ) -> None:
        self.pasted = FakeTextOutputHandler()
        self.copied = FakeTextOutputHandler()
        self.failures: list[tuple[DeliveryFailure, str]] = []
        self.slept: list[float] = []
        self._contexts = list(contexts or [NO_CONTEXT])
        self._foregrounds = list(foregrounds or [0x1234])
        self.delivery = AppAwareDelivery(
            self.pasted,
            self.copied,
            InlineGui(),
            lambda failure, text: self.failures.append((failure, text)),
            text_field_focused=lambda: text_field,
            read_context=self._next_context,
            foreground=self._next_foreground,
            start_thread=lambda work: work(),
            sleep=self.slept.append,
        )

    def _next_context(self) -> CaretContext:
        return self._contexts.pop(0) if len(self._contexts) > 1 else self._contexts[0]

    def _next_foreground(self) -> int:
        return self._foregrounds.pop(0) if len(self._foregrounds) > 1 else self._foregrounds[0]


def test_a_target_without_a_text_field_gets_the_clipboard_and_no_paste() -> None:
    harness = Harness(text_field=False)

    harness.delivery.output("Meeting moved to Thursday.")

    assert harness.pasted.outputs == []
    assert harness.copied.outputs == ["Meeting moved to Thursday."]
    assert harness.failures == [
        (DeliveryFailure.NO_TEXT_FIELD, "Meeting moved to Thursday."),
    ]


def test_a_paste_that_landed_is_left_alone() -> None:
    harness = Harness(contexts=[CaretContext("Hi team, Meeting moved to Thursday. ", "", "")])

    harness.delivery.output("Meeting moved to Thursday. ")

    assert harness.pasted.outputs == ["Meeting moved to Thursday. "]
    assert harness.failures == []
    assert harness.slept == [0.3]


def test_a_paste_that_never_showed_up_is_reported_once_after_a_second_look() -> None:
    before = CaretContext("Dear Sam,", "", "")
    harness = Harness(contexts=[before, before])

    harness.delivery.output("Thanks for the notes.")

    assert harness.pasted.outputs == ["Thanks for the notes."]
    assert harness.failures == [(DeliveryFailure.NOT_LANDED, "Thanks for the notes.")]
    assert harness.slept == [0.3, 0.7]


def test_a_slow_app_that_shows_the_paste_on_the_second_look_is_not_reported() -> None:
    harness = Harness(contexts=[
        CaretContext("Dear Sam,", "", ""),
        CaretContext("Dear Sam, thanks for the notes.", "", ""),
    ])

    harness.delivery.output("thanks for the notes.")

    assert harness.failures == []


def test_an_unreadable_field_is_never_second_guessed() -> None:
    harness = Harness(contexts=[NO_CONTEXT])

    harness.delivery.output("git status")

    assert harness.pasted.outputs == ["git status"]
    assert harness.failures == []


def test_switching_windows_before_the_check_skips_it() -> None:
    harness = Harness(
        contexts=[CaretContext("something else entirely", "", "")],
        foregrounds=[0x1234, 0x9999],
    )

    harness.delivery.output("Paste me.")

    assert harness.failures == []


@pytest.mark.parametrize(
    ("pasted", "before_caret", "selected"),
    [
        ("Let's meet at 10.  See you then.\n", "…ok. Let's meet at 10. See you\r\nthen.", ""),
        ("TYPED WITH CAPS LOCK", "typed with caps lock", ""),
        ("She said \"it's fine\".", "She said “it’s fine”.", ""),
        ("Grüße aus 東京 🎙️", "prefix Grüße aus 東京 🎙️", ""),
        ("selected after paste", "some text ", "selected after paste"),
        ("...", "anything at all", ""),
    ],
)
def test_the_check_accepts_a_paste_that_landed_with_different_spacing_or_marks(
    pasted: str, before_caret: str, selected: str,
) -> None:
    assert paste_landed(pasted, CaretContext(before_caret[-40:], selected, ""))


@pytest.mark.parametrize(
    ("pasted", "before_caret"),
    [
        ("Thanks for the notes.", "Dear Sam,"),
        ("Send the invoice today", "Send the invoice"),
        ("東京で会いましょう", "大阪で会いましょう"),
    ],
)
def test_the_check_rejects_a_paste_that_did_not_land(pasted: str, before_caret: str) -> None:
    assert not paste_landed(pasted, CaretContext(before_caret, "", ""))


def test_paste_last_resends_the_final_text_through_the_chain_without_the_pipeline(
) -> None:
    last = LastDictation()
    delivered = FakeTextOutputHandler()
    copied = FakeTextOutputHandler()
    last.route(paste=delivered, copy=copied)
    chain = LastDictationDecorator(delivered, last)

    chain.output("First note. ")
    chain.output("Second — naïve café. ")
    chain.output("   ")

    assert last.paste() is True
    assert delivered.outputs[-1] == "Second — naïve café. "
    assert copied.outputs == []
    assert last.copy() is True
    assert copied.outputs == ["Second — naïve café."]


def test_nothing_is_repeated_before_the_first_dictation() -> None:
    last = LastDictation()
    delivered = FakeTextOutputHandler()
    last.route(paste=delivered, copy=delivered)

    assert last.paste() is False
    assert last.copy() is False
    assert delivered.outputs == []


def test_a_repeat_follows_the_output_chain_configured_now() -> None:
    last = LastDictation()
    old_chain = FakeTextOutputHandler()
    new_chain = FakeTextOutputHandler()
    last.route(paste=old_chain, copy=old_chain)
    LastDictationDecorator(old_chain, last).output("dictated before Save & Apply")

    last.route(paste=new_chain, copy=new_chain)
    last.paste()

    assert new_chain.outputs == ["dictated before Save & Apply"]
