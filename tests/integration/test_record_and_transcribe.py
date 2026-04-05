from unittest.mock import MagicMock

import numpy as np

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    RecordingStartedEvent,
    RecordingStoppedEvent,
    StateChangedEvent,
    TranscriptionCompletedEvent,
)
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


def _make_audio() -> AudioSegment:
    return AudioSegment(data=np.zeros(16000, dtype=np.float32), sample_rate=16000)


class TestFullTogglePipeline:
    def test_should_record_transcribe_and_output_text(self):
        event_bus = SimpleEventBus()
        state_machine = AppStateMachine(event_bus)

        recorder = MagicMock()
        recorder.stop_recording.return_value = _make_audio()

        engine = MagicMock()
        engine.transcribe.return_value = TranscriptionResult(
            text="integration test works",
            language="en",
            language_probability=0.99,
            duration_seconds=0.3,
        )

        output = MagicMock()

        events_received = {
            "state_changes": [],
            "recording_started": [],
            "recording_stopped": [],
            "transcription_completed": [],
        }
        event_bus.subscribe(
            StateChangedEvent, events_received["state_changes"].append
        )
        event_bus.subscribe(
            RecordingStartedEvent, events_received["recording_started"].append
        )
        event_bus.subscribe(
            RecordingStoppedEvent, events_received["recording_stopped"].append
        )
        event_bus.subscribe(
            TranscriptionCompletedEvent,
            events_received["transcription_completed"].append,
        )

        use_case = RecordAndTranscribeUseCase(
            recorder=recorder,
            engine=engine,
            output=output,
            state_machine=state_machine,
            event_bus=event_bus,
        )

        # First press: start recording
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        assert state_machine.state == AppState.RECORDING
        recorder.start_recording.assert_called_once()

        # Second press: stop recording + transcribe
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        # Verify full pipeline
        recorder.stop_recording.assert_called_once()
        engine.transcribe.assert_called_once()
        output.output.assert_called_once_with("integration test works")
        assert state_machine.state == AppState.IDLE

        # Verify event sequence
        assert len(events_received["recording_started"]) == 1
        assert len(events_received["recording_stopped"]) == 1
        assert len(events_received["transcription_completed"]) == 1
        assert events_received["transcription_completed"][0].text == "integration test works"

        # State transitions: IDLE->RECORDING, RECORDING->TRANSCRIBING, TRANSCRIBING->IDLE
        state_changes = events_received["state_changes"]
        assert len(state_changes) == 3
        assert state_changes[0].new_state == "recording"
        assert state_changes[1].new_state == "transcribing"
        assert state_changes[2].new_state == "idle"

        use_case.dispose()


class TestFullPushToTalkPipeline:
    def test_should_record_on_activate_and_transcribe_on_deactivate(self):
        event_bus = SimpleEventBus()
        state_machine = AppStateMachine(event_bus)

        recorder = MagicMock()
        recorder.stop_recording.return_value = _make_audio()

        engine = MagicMock()
        engine.transcribe.return_value = TranscriptionResult(
            text="push to talk works",
            language="en",
            language_probability=0.98,
            duration_seconds=0.2,
        )

        output = MagicMock()

        use_case = RecordAndTranscribeUseCase(
            recorder=recorder,
            engine=engine,
            output=output,
            state_machine=state_machine,
            event_bus=event_bus,
        )

        # Key press: start recording
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        assert state_machine.state == AppState.RECORDING

        # Key release: stop + transcribe
        use_case.on_hotkey_deactivated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        output.output.assert_called_once_with("push to talk works")
        assert state_machine.state == AppState.IDLE

        use_case.dispose()
