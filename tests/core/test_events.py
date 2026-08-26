from voicetotext.core.events import (
    DomainEvent,
    EngineLoadFailedEvent,
    ErrorOccurredEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    StateChangedEvent,
    TranscriptionCompletedEvent,
    TranscriptionEmptyEvent,
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
        event = TranscriptionCompletedEvent(text="hello", transcription_seconds=1.5)

        assert event.text == "hello"
        assert event.transcription_seconds == 1.5


class TestTranscriptionFailedEvent:
    def test_should_carry_error_message(self):
        event = TranscriptionFailedEvent(error="Model not loaded")

        assert event.error == "Model not loaded"


class TestTranscriptionEmptyEvent:
    """Surface engine-returned-no-speech as an explicit event so the
    tray can notify the user instead of swallowing the empty clipboard
    silently.
    """

    def test_should_carry_transcription_seconds(self):
        event = TranscriptionEmptyEvent(transcription_seconds=0.42)

        assert event.transcription_seconds == 0.42

    def test_should_default_transcription_seconds_to_zero(self):
        event = TranscriptionEmptyEvent()

        assert event.transcription_seconds == 0.0


class TestEngineLoadFailedEvent:
    """Surface engine-load failures (missing backend, missing model
    file) to the tray so the user sees the actionable reason rather
    than a stuck LOADING icon.
    """

    def test_should_carry_model_name_and_error(self):
        event = EngineLoadFailedEvent(
            model_name="nemotron-speech-streaming-en-0.6b", error="nano-parakeet not installed",
        )

        assert event.model_name == "nemotron-speech-streaming-en-0.6b"
        assert event.error == "nano-parakeet not installed"


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
            TranscriptionCompletedEvent(text="hi", transcription_seconds=1.0),
            TranscriptionFailedEvent(error="fail"),
            TranscriptionEmptyEvent(transcription_seconds=0.5),
            EngineLoadFailedEvent(model_name="parakeet-tdt-0.6b-v3", error="missing"),
            ErrorOccurredEvent(error="oops"),
        ]

        for event in events:
            assert isinstance(event, DomainEvent)
