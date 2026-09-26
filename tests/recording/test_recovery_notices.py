from collections.abc import Callable, Sequence

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    DeliveryFailedEvent,
    RecordingCancelledEvent,
    TranscriptionEmptyEvent,
    TranscriptionFailedEvent,
)
from voicetotext.recording.notice import NoticeAction
from voicetotext.recording.recovery_notices import RecoveryNotices


class FakeNotice:
    def __init__(self) -> None:
        self.shown: list[tuple[str, list[NoticeAction]]] = []

    def show(self, message: str, buttons: Sequence[NoticeAction] = ()) -> None:
        self.shown.append((message, list(buttons)))

    def labels(self) -> list[str]:
        return [label for label, _action in self.shown[-1][1]]

    def press(self, label: str) -> None:
        dict(self.shown[-1][1])[label]()


class Setup:
    def __init__(self) -> None:
        self.bus = SimpleEventBus()
        self.notice = FakeNotice()
        self.calls: list[str] = []
        self.scheduled: list[Callable[[], None]] = []
        RecoveryNotices(
            self.bus,
            self.notice,
            self.scheduled.append,
            retry=lambda: self.calls.append("retry"),
            open_history=lambda: self.calls.append("history"),
            paste_again=lambda: self.calls.append("paste"),
            copy_again=lambda: self.calls.append("copy"),
        )

    def publish_on_worker(self, event) -> None:
        self.bus.publish(event)
        assert self.notice.shown == [] or self.scheduled == []
        for action in self.scheduled:
            action()
        self.scheduled.clear()


def test_a_failed_transcription_offers_retry_and_history() -> None:
    setup = Setup()

    setup.publish_on_worker(TranscriptionFailedEvent(error="CUDA OOM", recoverable=True))
    setup.notice.press("Retry")

    assert setup.notice.shown[0][0] == "Transcription failed."
    assert setup.notice.labels() == ["Retry", "Open history"]
    assert setup.calls == ["retry"]


def test_a_failure_with_nothing_to_retry_only_offers_history() -> None:
    setup = Setup()

    setup.publish_on_worker(TranscriptionFailedEvent(error="Streaming failed"))

    assert setup.notice.labels() == ["Open history"]


def test_nothing_to_paste_into_says_where_the_text_went() -> None:
    setup = Setup()

    setup.publish_on_worker(DeliveryFailedEvent(reason="no_text_field", text="Hi — ✓"))
    setup.notice.press("Open history")

    assert setup.notice.shown[0][0] == "Nothing to paste into. The text is on your clipboard."
    assert setup.calls == ["history"]


def test_a_paste_that_did_not_land_offers_paste_again_and_copy() -> None:
    setup = Setup()

    setup.publish_on_worker(DeliveryFailedEvent(reason="not_landed", text="Ship it."))
    setup.notice.press("Paste again")
    setup.notice.press("Copy")

    assert setup.notice.shown[0][0] == "The paste did not land."
    assert setup.calls == ["paste", "copy"]


def test_a_cancel_offers_transcribe_anyway_only_when_there_is_audio() -> None:
    setup = Setup()

    setup.publish_on_worker(RecordingCancelledEvent(recoverable=False))
    assert setup.notice.shown == []
    setup.publish_on_worker(RecordingCancelledEvent(recoverable=True))
    setup.notice.press("Transcribe anyway")

    assert setup.notice.shown[0][0] == "Recording cancelled."
    assert setup.calls == ["retry"]


def test_an_empty_transcription_shows_no_notice() -> None:
    setup = Setup()

    setup.publish_on_worker(TranscriptionEmptyEvent(transcription_seconds=0.2))

    assert setup.notice.shown == []


def test_notices_are_shown_on_the_gui_thread_not_the_publisher() -> None:
    setup = Setup()

    setup.bus.publish(TranscriptionFailedEvent(error="boom", recoverable=True))

    assert setup.notice.shown == []
    assert len(setup.scheduled) == 1
