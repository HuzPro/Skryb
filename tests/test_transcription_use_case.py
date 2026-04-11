import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    ErrorOccurredEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.exceptions import AudioRecorderError
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(data=np.zeros(samples, dtype=np.float32), sample_rate=16000)


def _make_result(text: str = "hello world") -> TranscriptionResult:
    return TranscriptionResult(
        text=text, language="en", language_probability=0.95, duration_seconds=0.5
    )


def _build_use_case():
    event_bus = SimpleEventBus()
    state_machine = AppStateMachine(event_bus)
    recorder = MagicMock()
    engine = MagicMock()
    output = MagicMock()
    use_case = RecordAndTranscribeUseCase(
        recorder=recorder,
        engine=engine,
        output=output,
        state_machine=state_machine,
        event_bus=event_bus,
    )
    return use_case, recorder, engine, output, event_bus, state_machine


@pytest.fixture()
def _synchronous_threads(monkeypatch):
    class _SyncThread:
        def __init__(self, *, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr("threading.Thread", _SyncThread)


class TestActivateFromIdle:
    def test_should_start_recording_on_activate_when_idle(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case()

        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        recorder.start_recording.assert_called_once()
        assert state_machine.state == AppState.RECORDING

    def test_should_publish_recording_started_event(self):
        use_case, _, _, _, event_bus, _ = _build_use_case()
        received = []
        event_bus.subscribe(RecordingStartedEvent, received.append)

        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        assert len(received) == 1


class TestActivateFromRecording:
    def test_should_stop_recording_and_begin_transcription(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        audio = _make_audio()
        recorder.stop_recording.return_value = audio
        engine.transcribe.return_value = _make_result("hello")

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        recorder.stop_recording.assert_called_once()
        engine.transcribe.assert_called_once_with(audio)

    def test_should_send_text_to_output_handler(self):
        use_case, recorder, engine, output, _, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        output.output.assert_called_once_with("hello")

    def test_should_not_output_empty_transcription(self):
        use_case, recorder, engine, output, _, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("")

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        output.output.assert_not_called()

    def test_should_return_to_idle_after_transcription(self):
        use_case, recorder, engine, _, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        assert state_machine.state == AppState.IDLE

    def test_should_publish_recording_stopped_event(self):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")
        received = []
        event_bus.subscribe(RecordingStoppedEvent, received.append)

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        assert len(received) == 1

    def test_should_publish_transcription_completed_event(self):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")
        received = []
        event_bus.subscribe(TranscriptionCompletedEvent, received.append)

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        assert len(received) == 1
        assert received[0].text == "hello"


class TestActivateFromTranscribing:
    def test_should_ignore_activate_while_transcribing(self):
        use_case, recorder, engine, _, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        blocker = threading.Event()
        engine.transcribe.side_effect = lambda a: (
            blocker.wait(),
            _make_result("hi"),
        )[1]

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        assert state_machine.state == AppState.TRANSCRIBING
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        assert state_machine.state == AppState.TRANSCRIBING

        blocker.set()
        use_case._wait_for_transcription()


class TestDeactivateForPushToTalk:
    def test_should_stop_and_transcribe_on_deactivate(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_deactivated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        recorder.stop_recording.assert_called_once()
        engine.transcribe.assert_called_once()
        output.output.assert_called_once_with("hello")
        assert state_machine.state == AppState.IDLE

    def test_should_ignore_deactivate_when_not_recording(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case()

        use_case.on_hotkey_deactivated()
        use_case._wait_for_action()

        recorder.stop_recording.assert_not_called()
        assert state_machine.state == AppState.IDLE


class TestTranscriptionError:
    def test_should_return_to_idle_on_transcription_failure(self):
        use_case, recorder, engine, _, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.side_effect = RuntimeError("model crashed")

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        assert state_machine.state == AppState.IDLE

    def test_should_publish_transcription_failed_event_on_error(self):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.side_effect = RuntimeError("model crashed")
        received = []
        event_bus.subscribe(TranscriptionFailedEvent, received.append)

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        assert len(received) == 1
        assert "model crashed" in received[0].error


class TestHotSwapEngine:
    def test_should_swap_engine_when_idle(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        new_engine = MagicMock()
        new_engine.transcribe.return_value = _make_result("from new engine")
        recorder.stop_recording.return_value = _make_audio()

        use_case.set_engine(new_engine)
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        new_engine.transcribe.assert_called_once()
        engine.transcribe.assert_not_called()

    def test_should_reject_engine_swap_when_not_idle(self):
        use_case, recorder, _, _, _, _ = _build_use_case()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        with pytest.raises(RuntimeError, match="idle"):
            use_case.set_engine(MagicMock())

    def test_should_dispose_old_engine_when_swapped_out(self):
        use_case, _, old_engine, _, _, _ = _build_use_case()
        new_engine = MagicMock()

        use_case.set_engine(new_engine)

        old_engine.dispose.assert_called_once()

    def test_should_not_dispose_new_engine_when_swap_rejected(self):
        use_case, recorder, old_engine, _, _, _ = _build_use_case()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        new_engine = MagicMock()

        with pytest.raises(RuntimeError):
            use_case.set_engine(new_engine)

        old_engine.dispose.assert_not_called()
        new_engine.dispose.assert_not_called()


class TestHotSwapOutput:
    def test_should_swap_output_handler(self):
        use_case, recorder, engine, _, _, _ = _build_use_case()
        new_output = MagicMock()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")

        use_case.set_output(new_output)
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case._wait_for_transcription()

        new_output.output.assert_called_once_with("hello")
        use_case.dispose()


class TestStartRecordingFailure:
    def test_should_recover_to_idle_when_start_recording_fails(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case()
        recorder.start_recording.side_effect = AudioRecorderError("mic unavailable")

        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        assert state_machine.state == AppState.IDLE

    @pytest.mark.usefixtures("_synchronous_threads")
    def test_should_publish_error_event_when_start_recording_fails(self):
        use_case, recorder, _, _, event_bus, _ = _build_use_case()
        recorder.start_recording.side_effect = AudioRecorderError("mic unavailable")
        received = []
        event_bus.subscribe(ErrorOccurredEvent, received.append)

        use_case.on_hotkey_activated()

        assert len(received) == 1
        assert "mic unavailable" in received[0].error


class TestStopRecordingFailure:
    def test_should_recover_to_idle_when_stop_recording_fails(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case()
        recorder.stop_recording.side_effect = AudioRecorderError("stream error")

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        assert state_machine.state == AppState.IDLE

    def test_should_publish_error_event_when_stop_recording_fails(self):
        use_case, recorder, _, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.side_effect = AudioRecorderError("stream error")
        received = []
        event_bus.subscribe(ErrorOccurredEvent, received.append)

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        assert len(received) == 1
        assert "stream error" in received[0].error


class TestShortRecording:
    def test_should_skip_transcription_for_audio_shorter_than_minimum(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        short_audio = AudioSegment(
            data=np.zeros(800, dtype=np.float32),
            sample_rate=16000,
        )
        recorder.stop_recording.return_value = short_audio

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        engine.transcribe.assert_not_called()
        output.output.assert_not_called()
        assert state_machine.state == AppState.IDLE


class TestEmptyRecording:
    def test_should_skip_transcription_when_recording_is_empty(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = None

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        engine.transcribe.assert_not_called()
        output.output.assert_not_called()
        assert state_machine.state == AppState.IDLE

    def test_should_publish_recording_stopped_event_even_when_empty(self):
        use_case, recorder, _, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = None
        received = []
        event_bus.subscribe(RecordingStoppedEvent, received.append)

        use_case.on_hotkey_activated()
        use_case._wait_for_action()
        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        assert len(received) == 1


class TestNonBlockingCallbacks:
    def test_should_not_block_caller_on_activate(self):
        use_case, recorder, _, _, _, _ = _build_use_case()
        blocker = threading.Event()
        recorder.start_recording.side_effect = lambda: blocker.wait(timeout=5)

        start = time.monotonic()
        use_case.on_hotkey_activated()
        elapsed = time.monotonic() - start

        blocker.set()
        use_case._wait_for_action()
        assert elapsed < 0.1

    def test_should_not_block_caller_on_deactivate(self):
        use_case, recorder, engine, _, _, _ = _build_use_case()

        use_case.on_hotkey_activated()
        use_case._wait_for_action()

        blocker = threading.Event()
        original_stop = recorder.stop_recording.return_value

        def blocking_stop():
            blocker.wait(timeout=5)
            return _make_audio()

        recorder.stop_recording.side_effect = blocking_stop

        start = time.monotonic()
        use_case.on_hotkey_deactivated()
        elapsed = time.monotonic() - start

        blocker.set()
        use_case._wait_for_action()
        assert elapsed < 0.1


class TestDispose:
    def test_should_shutdown_executor_on_dispose(self):
        use_case, _, _, _, _, _ = _build_use_case()

        use_case.dispose()
