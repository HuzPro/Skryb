"""Acceptance-test DSL harness for VoiceToText.

``VoiceToTextApp`` composes the real ``RecordAndTranscribeUseCase``,
``AppStateMachine``, ``ClipboardPreservingDecorator``, and
``EngineHotSwapCoordinator`` with hand-rolled fakes at every port
boundary.  Its methods read as business actions; its properties as
observable outcomes.  Acceptance tests should never reach past it
into the components it composes - everything they need is exposed
as a first-class DSL step.
"""

from pathlib import Path
from unittest.mock import MagicMock

from tests.builders import a_transcription_result, an_audio_segment
from tests.fakes import (
    FakeAudioRecorder,
    FakeClipboardPreserver,
    FakeTextOutputHandler,
    FakeTranscriptionEngine,
    InlineActionDispatcher,
    InstantSleeper,
    RecordingEventBus,
)
from voicetotext.core.events import (
    DomainEvent,
    EngineLoadCompletedEvent,
)
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.diagnostics.session import SessionDiagnostics
from voicetotext.output.clipboard_preserving_decorator import (
    ClipboardPreservingDecorator,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.settings.engine_hot_swap import EngineHotSwapCoordinator
from voicetotext.transcription.engine_install import make_engine_installer
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


class VoiceToTextApp:
    """DSL harness for acceptance-testing the voice-to-text flow.

    All components are wired synchronously via an
    :class:`InlineActionDispatcher`, so every call completes before
    control returns to the test.  Scenarios read top-to-bottom as
    Given/When/Then.
    """

    def __init__(self) -> None:
        self._bus = RecordingEventBus()
        self._state_machine = AppStateMachine(self._bus)
        self._recorder = FakeAudioRecorder()
        self._engine = FakeTranscriptionEngine()
        self._preserver = FakeClipboardPreserver()
        self._sleeper = InstantSleeper()
        self._raw_output = FakeTextOutputHandler()
        self._wrapped_output = ClipboardPreservingDecorator(
            inner=self._raw_output,
            preserver=self._preserver,
            sleeper=self._sleeper,
        )
        self._dispatcher = InlineActionDispatcher()
        self._use_case = RecordAndTranscribeUseCase(
            recorder=self._recorder,
            engine=self._engine,
            output=self._wrapped_output,
            state_machine=self._state_machine,
            event_bus=self._bus,
            action_dispatcher=self._dispatcher,
            diagnostics=MagicMock(spec=SessionDiagnostics),
        )
        self._next_engine: FakeTranscriptionEngine | None = None
        self._hot_swap = EngineHotSwapCoordinator(
            use_case=self._use_case,
            event_bus=self._bus,
            loader=self._loader,
            action_dispatcher=self._dispatcher,
        )
        self._bus.subscribe(
            EngineLoadCompletedEvent, self._on_engine_load_completed
        )

        # Sensible defaults so the happy path works with no setup:
        # a one-second silent recording transcribed as "hello world".
        self._recorder.will_return(
            an_audio_segment().silent().with_duration(seconds=1.0).build()
        )
        self._engine.will_return(
            a_transcription_result().with_text("hello world").build()
        )

    # ------------------------------------------------------------------
    # Given - programme the world the user sees
    # ------------------------------------------------------------------

    def user_speaks(self, text: str) -> None:
        self._engine.will_return(
            a_transcription_result().with_text(text).build()
        )

    def microphone_captures(self, audio: AudioSegment) -> None:
        self._recorder.will_return(audio)

    def microphone_captures_from_file(self, path: Path) -> None:
        from tests.helpers.audio_file_loader import load_audio_file

        self._recorder.will_return(load_audio_file(path))

    def microphone_captures_silence_of(self, seconds: float) -> None:
        self._recorder.will_return(
            an_audio_segment().silent().with_duration(seconds=seconds).build()
        )

    def microphone_captures_nothing(self) -> None:
        self._recorder.will_return(None)

    def microphone_fails_to_start_with(self, error: Exception) -> None:
        self._recorder.will_fail_start(error)

    def microphone_fails_to_stop_with(self, error: Exception) -> None:
        self._recorder.will_fail_stop(error)

    def engine_fails_with(self, error: Exception) -> None:
        self._engine.will_fail(error)

    def clipboard_save_fails_with(self, error: Exception) -> None:
        self._preserver.will_fail_save(error)

    def clipboard_restore_fails_with(self, error: Exception) -> None:
        self._preserver.will_fail_restore(error)

    def output_fails_with(self, error: Exception) -> None:
        self._raw_output.will_fail(error)

    def next_engine_will_be(self, engine: FakeTranscriptionEngine) -> None:
        self._next_engine = engine

    # ------------------------------------------------------------------
    # When - the user's actions
    # ------------------------------------------------------------------

    def press_hotkey(self) -> None:
        self._use_case.on_hotkey_activated()

    def release_hotkey(self) -> None:
        self._use_case.on_hotkey_deactivated()

    def press_escape(self) -> None:
        self._use_case.on_cancel()

    def trigger_engine_swap(
        self, model_name: str = "parakeet-tdt-0.6b-v3",
    ) -> None:
        self._hot_swap.start_swap(model_name)

    # ------------------------------------------------------------------
    # Then - observable outcomes
    # ------------------------------------------------------------------

    @property
    def typed_text(self) -> str | None:
        return self._raw_output.outputs[-1] if self._raw_output.outputs else None

    @property
    def all_typed_text(self) -> list[str]:
        return list(self._raw_output.outputs)

    @property
    def events(self) -> list[DomainEvent]:
        return list(self._bus.history)

    def events_of(self, event_type: type[DomainEvent]) -> list:
        return self._bus.events_of(event_type)

    @property
    def state(self) -> AppState:
        return self._state_machine.state

    @property
    def clipboard_was_preserved(self) -> bool:
        return self._preserver.saved_count > 0 and self._preserver.restored_count > 0

    @property
    def clipboard_saved_count(self) -> int:
        return self._preserver.saved_count

    @property
    def clipboard_restored_count(self) -> int:
        return self._preserver.restored_count

    @property
    def current_engine(self) -> FakeTranscriptionEngine:
        return self._engine

    @property
    def engine_transcribe_calls(self) -> int:
        return len(self._engine.transcribe_calls)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        """Release any resources the app is holding.

        Safe to call even when the engine is a pure fake; the adapter
        contract guarantees ``dispose`` is idempotent.
        """
        self._engine.dispose()

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _loader(self):
        engine = self._next_engine or FakeTranscriptionEngine()
        return make_engine_installer(
            engine, lambda use_case, loaded: use_case.set_engine(loaded),
        )

    def _on_engine_load_completed(
        self, event: EngineLoadCompletedEvent
    ) -> None:
        if self._next_engine is not None:
            self._engine = self._next_engine
            self._next_engine = None
