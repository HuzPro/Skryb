from collections.abc import Callable

from tests.fakes import FakeTextOutputHandler
from skryb.output.app_aware_delivery import AppAwareDelivery, DeliveryFailure
from skryb.output.gui_thread_dispatcher import GuiThreadDispatcher
from skryb.output.last_dictation import LastDictation, LastDictationDecorator


class InlineGui(GuiThreadDispatcher):
    def __init__(self) -> None:
        self.dispatched = 0

    def dispatch(self, action: Callable[[], None]) -> None:
        self.dispatched += 1
        action()


class Harness:
    """An app-aware delivery wired to fakes, running every step inline."""

    def __init__(self, *, text_field: bool = True) -> None:
        self.pasted = FakeTextOutputHandler()
        self.copied = FakeTextOutputHandler()
        self.failures: list[tuple[DeliveryFailure, str]] = []
        self.delivery = AppAwareDelivery(
            self.pasted,
            self.copied,
            InlineGui(),
            lambda failure, text: self.failures.append((failure, text)),
            text_field_focused=lambda: text_field,
        )


def test_a_target_without_a_text_field_gets_the_clipboard_and_no_paste() -> None:
    harness = Harness(text_field=False)

    harness.delivery.output("Meeting moved to Thursday.")

    assert harness.pasted.outputs == []
    assert harness.copied.outputs == ["Meeting moved to Thursday."]
    assert harness.failures == [
        (DeliveryFailure.NO_TEXT_FIELD, "Meeting moved to Thursday."),
    ]


def test_a_paste_into_a_text_field_is_pasted_once_and_never_reported_as_lost() -> None:
    harness = Harness()

    for text in ("Zoë said ✓ ", "", "   ", "Grüße aus 東京 🎙️ " * 200):
        harness.delivery.output(text)

    assert harness.pasted.outputs == ["Zoë said ✓ ", "", "   ", "Grüße aus 東京 🎙️ " * 200]
    assert harness.copied.outputs == []
    assert harness.failures == []

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
