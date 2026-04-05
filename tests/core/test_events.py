from voicetotext.core.events import (
    DomainEvent,
    ErrorOccurredEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    StateChangedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)


class TestDomainEventTimestamp:
    def test_should_auto_assign_timestamp(self):
        event = RecordingStartedEvent()

        assert event.timestamp > 0

    def test_should_assign_unique_timestamps(self):
        first = RecordingStartedEvent()
        second = RecordingStoppedEvent()

        assert first.timestamp <= second.timestamp


class TestStateChangedEvent:
    def test_should_carry_old_and_new_state(self):
        event = StateChangedEvent(old_state="IDLE", new_state="RECORDING")

        assert event.old_state == "IDLE"
        assert event.new_state == "RECORDING"


class TestTranscriptionCompletedEvent:
    def test_should_carry_text_and_duration(self):
        event = TranscriptionCompletedEvent(text="hello", duration_seconds=1.5)

        assert event.text == "hello"
        assert event.duration_seconds == 1.5


class TestTranscriptionFailedEvent:
    def test_should_carry_error_message(self):
        event = TranscriptionFailedEvent(error="Model not loaded")

        assert event.error == "Model not loaded"


class TestErrorOccurredEvent:
    def test_should_carry_error_details(self):
        event = ErrorOccurredEvent(error="Microphone unavailable")

        assert event.error == "Microphone unavailable"


class TestEventImmutability:
    def test_events_should_be_immutable(self):
        event = RecordingStartedEvent()

        try:
            event.timestamp = 0
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestEventInheritance:
    def test_all_events_should_be_domain_events(self):
        events = [
            RecordingStartedEvent(),
            RecordingStoppedEvent(),
            StateChangedEvent(old_state="IDLE", new_state="RECORDING"),
            TranscriptionCompletedEvent(text="hi", duration_seconds=1.0),
            TranscriptionFailedEvent(error="fail"),
            ErrorOccurredEvent(error="oops"),
        ]

        for event in events:
            assert isinstance(event, DomainEvent)
