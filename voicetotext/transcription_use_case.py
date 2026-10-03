import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from voicetotext.core.action_dispatcher import ActionDispatcher
from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    DomainEvent,
    ErrorOccurredEvent,
    NoLastDictationEvent,
    RecordingCancelledEvent,
    RecordingCapWarningEvent,
    RecordingCappedEvent,
    RecordingPausedEvent,
    RecordingResumedEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    RetryStartedEvent,
    StateChangedEvent,
    TranscriptionCompletedEvent,
    TranscriptionEmptyEvent,
    TranscriptionFailedEvent,
    TranscriptionStartedEvent,
    WritingNoticeEvent,
)
from voicetotext.core.exceptions import EngineSwapError
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.diagnostics.session import SessionDiagnostics
from voicetotext.output.clipboard_write_error import ClipboardWriteError
from voicetotext.output.dictation_trace import begin_trace, end_trace
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.last_dictation import LastDictation
from voicetotext.recording.audio_segment import AudioSegment, load_wav
from voicetotext.recording.recorder import AudioRecorder
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.placeholder_engine import PlaceholderEngine
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.live_streaming_controller import (
    LiveStreamingController,
)
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionEngine,
)
from voicetotext.writing.voice_edit import Selection, VoiceEditError, VoiceEditor

_logger = logging.getLogger(__name__)
_MIN_DURATION_SECONDS = 0.1
_DEFAULT_MAX_RECORDING_SECONDS = 600.0
_EDIT_CAPTION = "Editing selection"


def _no_writing_time(_audio_seconds: float) -> float:
    return 0.0


def _default_cap() -> float:
    return _DEFAULT_MAX_RECORDING_SECONDS


def _no_warning() -> float:
    return 0.0


def _never_wake() -> float:
    return 0.0


class RecordAndTranscribeUseCase:

    def __init__(
        self,
        recorder: AudioRecorder,
        engine: TranscriptionEngine,
        output: TextOutputHandler,
        state_machine: AppStateMachine,
        event_bus: EventBus,
        action_dispatcher: ActionDispatcher,
        diagnostics: SessionDiagnostics,
        max_recording_seconds: Callable[[], float] = _default_cap,
        streaming_controller: LiveStreamingController | None = None,
        is_streaming_mode: Callable[[], bool] | None = None,
        prepare_audio: Callable[[AudioSegment], AudioSegment] | None = None,
        voice_edit: VoiceEditor | None = None,
        writing_seconds: Callable[[float], float] = _no_writing_time,
        warn_before_cap_seconds: Callable[[], float] = _no_warning,
        last_dictation: LastDictation | None = None,
        wake_after_idle_seconds: Callable[[], float] = _never_wake,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._recorder = recorder
        self._engine = engine
        self._output = output
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._dispatcher = action_dispatcher
        self._diagnostics = diagnostics
        # Both limits are asked at every start, so a saved change applies to
        # the next recording.
        self._max_recording_seconds = max_recording_seconds
        self._warn_before_cap_seconds = warn_before_cap_seconds
        self._streaming = streaming_controller
        self._is_streaming_mode = is_streaming_mode or (lambda: False)
        # Runs on the audio the engine hears only; history keeps the original.
        self._prepare_audio = prepare_audio or (lambda audio: audio)
        self._streaming_active = False
        self._stream_requested_at: datetime | None = None
        self._action_lock = threading.Lock()
        # A retry from history runs beside live dictation; one model must
        # never decode two recordings at once.
        self._engine_lock = threading.Lock()
        # Latched signal: did the user press the hotkey while the
        # engine was still loading? On ``LOADING → IDLE`` we replay
        # exactly one activation so the user's first hotkey is honoured
        # rather than swallowed. Multiple presses during LOADING
        # coalesce to a single replay - the user's intent is "start
        # the moment you're ready", not "start N times".
        self._queued_activation = False
        # ``threading.Timer`` is one-shot: scheduled at recording start
        # to enforce the duration cap, cancelled when the user stops
        # normally or replaced when the recording finishes auto-stopped.
        self._cap_timer: threading.Timer | None = None
        self._cap_warning_timer: threading.Timer | None = None
        # Paused time does not count towards the cap: the cap bounds the
        # audio, and a pause records none.
        self._paused = False
        self._cap_seconds = _DEFAULT_MAX_RECORDING_SECONDS
        self._warn_seconds = 0.0
        self._cap_warned = False
        self._cap_remaining = self._cap_seconds
        self._cap_started_at = 0.0
        self._voice_edit = voice_edit
        # How long the writing model will add, from the audio length; it
        # stretches the pill's estimate so the bar does not stall at full.
        self._writing_seconds = writing_seconds
        # The selection a voice edit will replace, from its start until the
        # recording is transcribed or dropped.
        self._editing: Selection | None = None
        self._last_dictation = last_dictation or LastDictation()
        # The last recording that was cancelled or failed to transcribe,
        # kept in memory so the notice beside it can still transcribe it.
        self._unfinished_audio: AudioSegment | None = None
        self._unfinished_lock = threading.Lock()
        # After a quiet spell the first decode ran several times slower in
        # the logs, most likely the GPU clocking down and the driver paging
        # the idle model out. A recording that starts after this long wakes
        # the engine while the speaker is still talking. Zero turns it off.
        self._wake_after_idle_seconds = wake_after_idle_seconds
        self._clock = clock
        self._engine_used_at = clock()
        event_bus.subscribe(StateChangedEvent, self._on_state_changed)

    def on_hotkey_activated(self) -> None:
        self._dispatcher.dispatch(self._handle_activate)

    def on_hotkey_deactivated(self) -> None:
        self._dispatcher.dispatch(self._handle_deactivate)

    def on_edit_hotkey_activated(self) -> None:
        self._dispatcher.dispatch(self._handle_edit_activate)

    def on_edit_hotkey_deactivated(self) -> None:
        self._dispatcher.dispatch(self._handle_deactivate)

    def on_pause_toggled(self) -> None:
        self._dispatcher.dispatch(self._handle_pause_toggle)

    def on_cancel(self) -> None:
        self._dispatcher.dispatch(self._handle_cancel)

    def is_recording(self) -> bool:
        return self._state_machine.state == AppState.RECORDING

    def paste_last(self) -> None:
        """Deliver the last dictation's final text again, as output is set up now."""
        self._dispatcher.dispatch(
            lambda: self._repeat_last_dictation(self._last_dictation.paste),
        )

    def copy_last(self) -> None:
        """Put the last dictation's final text on the clipboard, nothing more."""
        self._dispatcher.dispatch(
            lambda: self._repeat_last_dictation(self._last_dictation.copy),
        )

    def retry_last_recording(self) -> None:
        """Transcribe the last cancelled or failed recording and deliver it."""
        self._dispatcher.dispatch(self._transcribe_unfinished)

    def _repeat_last_dictation(self, send: Callable[[], bool]) -> None:
        if not send():
            self._event_bus.publish(NoLastDictationEvent())

    def _keep_unfinished(self, audio: AudioSegment | None) -> bool:
        if audio is None or audio.duration_seconds < _MIN_DURATION_SECONDS:
            return False
        with self._unfinished_lock:
            self._unfinished_audio = audio
        return True

    def _transcribe_unfinished(self) -> None:
        with self._unfinished_lock:
            audio, self._unfinished_audio = self._unfinished_audio, None
        if audio is None:
            return
        self._event_bus.publish(RetryStartedEvent())
        requested_at = datetime.now(UTC)
        try:
            result = self._transcribe(audio)
        except Exception as exc:  # noqa: BLE001 - offered again beside the failure
            self._report_unfinished_failure(audio, exc, requested_at)
            return
        self._deliver_transcribed_text(result)
        self._diagnostics.transcription_finished(audio, result, requested_at)
        self._event_bus.publish(
            TranscriptionCompletedEvent(
                text=result.text,
                transcription_seconds=result.transcription_seconds,
            )
        )

    def _report_unfinished_failure(
        self, audio: AudioSegment, error: Exception, requested_at: datetime,
    ) -> None:
        # No state change: a new dictation may already be under way.
        _logger.error("Transcribing the kept recording failed: %s", error)
        self._diagnostics.transcription_finished(audio, error, requested_at)
        self._event_bus.publish(
            TranscriptionFailedEvent(
                error=str(error), recoverable=self._keep_unfinished(audio),
            )
        )

    def _transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        heard = self._prepare_audio(audio)
        with self._engine_lock:
            try:
                return self._engine.transcribe(heard)
            finally:
                self._engine_used_at = self._clock()

    def _wake_engine_if_idle(self) -> None:
        wake_after = self._wake_after_idle_seconds()
        if wake_after <= 0 or self._clock() - self._engine_used_at < wake_after:
            return
        self._engine_used_at = self._clock()
        self._dispatcher.dispatch(self._wake_engine)

    def _wake_engine(self) -> None:
        # The engine lock makes the dictation's own decode wait for this
        # warmup rather than run beside it on the same GPU.
        with self._engine_lock:
            started = self._clock()
            try:
                self._engine.warmup()
            except Exception as exc:  # noqa: BLE001 - the dictation decodes anyway
                _logger.warning("Waking the engine after an idle spell failed: %s", exc)
                return
            finally:
                self._engine_used_at = self._clock()
        _logger.info(
            "Engine woken after an idle spell in %.2fs.", self._engine_used_at - started,
        )

    def retranscribe(
        self, wav_path: Path, on_done: Callable[[str | Exception], None],
    ) -> None:
        """Transcribe a saved recording again, for history; nothing is pasted."""

        def run() -> None:
            try:
                result = self._transcribe(load_wav(wav_path))
            except Exception as exc:  # noqa: BLE001 - reported to the window
                on_done(exc)
                return
            on_done(result.text)

        self._dispatcher.dispatch(run)

    def _handle_pause_toggle(self) -> None:
        with self._action_lock:
            if self._state_machine.state != AppState.RECORDING:
                return
            self._paused = not self._paused
            self._recorder.set_paused(self._paused)
            if self._paused:
                self._cap_remaining -= time.monotonic() - self._cap_started_at
                self._cancel_cap_timer()
                self._event_bus.publish(RecordingPausedEvent())
                _logger.info("Recording paused.")
            else:
                self._schedule_cap_timer()
                self._event_bus.publish(RecordingResumedEvent())
                _logger.info("Recording resumed.")

    def _handle_cancel(self) -> None:
        with self._action_lock:
            if self._state_machine.state != AppState.RECORDING:
                return
            if self._streaming_active:
                # Streaming has already typed what it heard; there is
                # nothing left to discard, so cancel just stops.
                self._stop_streaming()
                return
            self._cancel_cap_timer()
            self._paused = False
            editing, self._editing = self._editing, None
            try:
                audio = self._recorder.stop_recording()
            except Exception as exc:
                self._report_error(
                    "Failed to stop recording",
                    exc,
                    lambda message: ErrorOccurredEvent(error=message),
                )
                return
            self._state_machine.transition_to(AppState.IDLE)
            # A voice edit's audio is an instruction, not text to paste, so
            # it is never offered back as a dictation.
            self._event_bus.publish(
                RecordingCancelledEvent(
                    recoverable=editing is None and self._keep_unfinished(audio),
                ),
            )
            _logger.info("Recording cancelled.")
            self._dispatcher.dispatch(
                lambda: self._diagnostics.recording_cancelled(audio),
            )

    def _handle_activate(self) -> None:
        with self._action_lock:
            state = self._state_machine.state
            if state == AppState.LOADING:
                self._queued_activation = True
                _logger.info("Hotkey queued - engine still loading.")
                return
            if state == AppState.IDLE:
                self._begin_capture()
            elif state == AppState.RECORDING:
                self._end_capture()
            # TRANSCRIBING is intentionally ignored; no action while engine is busy.

    def _handle_edit_activate(self) -> None:
        with self._action_lock:
            state = self._state_machine.state
            if state == AppState.IDLE:
                self._begin_voice_edit()
            elif state == AppState.RECORDING:
                self._end_capture()

    def _begin_voice_edit(self) -> None:
        """Record an instruction for the selected text, or dictate when none is."""
        if self._voice_edit is None:
            self._begin_capture()
            return
        try:
            selection = self._voice_edit.capture()
        except VoiceEditError as exc:
            self._notify(str(exc))
            return
        if selection is None:
            self._begin_capture()
            return
        if self._streaming_ready():
            # Streaming parks the batch engine, which the instruction needs.
            self._notify("Voice editing works in batch mode only. The selection was left unchanged.")
            return
        self._start_recording(caption=_EDIT_CAPTION)
        if self._state_machine.state == AppState.RECORDING:
            self._editing = selection

    def _on_state_changed(self, event: StateChangedEvent) -> None:
        if event.old_state != AppState.LOADING.value:
            return
        if event.new_state != AppState.IDLE.value:
            return
        with self._action_lock:
            if not self._queued_activation:
                return
            self._queued_activation = False
        _logger.info("Engine ready - firing queued hotkey.")
        self._dispatcher.dispatch(self._handle_activate)

    def _handle_deactivate(self) -> None:
        with self._action_lock:
            if self._state_machine.state == AppState.RECORDING:
                self._end_capture()

    def dispose_current_engine(self) -> None:
        self._engine.dispose()

    def set_engine(self, engine: TranscriptionEngine) -> None:
        # Permit ``LOADING`` so bootstrap can attach the real engine to
        # the placeholder that the use case was constructed with on a
        # background thread, before the state machine flips to
        # ``IDLE``.
        self._require_swappable_state()
        # Under the engine lock, so a wake or a history retry still
        # decoding finishes before its engine is disposed.
        with self._engine_lock:
            old_engine = self._engine
            self._engine = engine
            old_engine.dispose()
            self._engine_used_at = self._clock()
        if self._streaming is not None:
            self._streaming.set_engine(None)

    def set_streaming_engine(
        self, engine: StreamingTranscriptionEngine,
    ) -> None:
        """Install the streaming engine and park the batch slot.

        The batch engine is replaced with a :class:`PlaceholderEngine` so
        its model frees its VRAM while streaming mode is active.
        """
        self._require_swappable_state()
        if self._streaming is None:
            engine.dispose()
            raise EngineSwapError("no streaming controller configured")
        self._streaming.set_engine(engine)
        with self._engine_lock:
            old_engine = self._engine
            self._engine = PlaceholderEngine()
            old_engine.dispose()

    def _require_swappable_state(self) -> None:
        if self._state_machine.state not in {AppState.IDLE, AppState.LOADING}:
            raise EngineSwapError("Can only swap engine when idle or loading")

    def set_output(self, output: TextOutputHandler) -> None:
        self._output = output
        if self._streaming is not None:
            self._streaming.set_output(output)

    def dispose(self) -> None:
        self._diagnostics.dispose()
        self.dispose_current_engine()
        if self._streaming is not None:
            self._streaming.set_engine(None)

    def _begin_capture(self) -> None:
        if self._streaming_ready():
            self._start_streaming()
        else:
            self._start_recording()

    def _end_capture(self) -> None:
        if self._streaming_active:
            self._stop_streaming()
        else:
            self._stop_and_transcribe()

    def _streaming_ready(self) -> bool:
        return (
            self._streaming is not None
            and self._streaming.has_engine()
            and self._is_streaming_mode()
        )

    def _start_recording(self, caption: str = "") -> None:
        self._paused = False
        self._reset_cap()
        try:
            self._recorder.start_recording()
        except Exception as e:
            _logger.error("Failed to start recording: %s", e)
            self._event_bus.publish(ErrorOccurredEvent(error=str(e)))
            return
        self._state_machine.transition_to(AppState.RECORDING)
        self._event_bus.publish(RecordingStartedEvent(caption=caption))
        self._schedule_cap_timer()
        _logger.info("Recording started.")
        self._wake_engine_if_idle()

    def _start_streaming(self) -> None:
        try:
            self._streaming.start(self._on_stream_complete)
        except Exception as exc:  # noqa: BLE001 - surface to the user
            _logger.error("Failed to start streaming: %s", exc)
            self._event_bus.publish(ErrorOccurredEvent(error=str(exc)))
            return
        self._paused = False
        self._reset_cap()
        self._streaming_active = True
        self._state_machine.transition_to(AppState.RECORDING)
        self._event_bus.publish(RecordingStartedEvent())
        self._schedule_cap_timer()
        _logger.info("Streaming started.")

    def _stop_streaming(self) -> None:
        self._stream_requested_at = datetime.now(UTC)
        self._finish_streaming()
        self._state_machine.transition_to(AppState.TRANSCRIBING)
        self._event_bus.publish(RecordingStoppedEvent())
        _logger.info("Streaming stopped. Finalizing transcript...")
        # Finalize (a GPU decode of the trailing window) off the action
        # lock, mirroring the batch path's dispatched transcription.
        self._dispatcher.dispatch(self._streaming.stop)

    def _finish_streaming(self) -> None:
        # Tear down the per-run streaming bookkeeping. Idempotent on
        # purpose: the user-stop path runs it, and ``_on_stream_complete``
        # runs it again so a mid-run worker failure - which never passes
        # through ``_stop_streaming`` - cannot leave ``_streaming_active``
        # latched (which would misroute the next stop) or a cap timer
        # pending (which would auto-stop the next run).
        self._cancel_cap_timer()
        self._streaming_active = False

    def _on_stream_complete(
        self,
        result: TranscriptionResult | None,
        error: Exception | None,
        audio: AudioSegment | None,
    ) -> None:
        self._finish_streaming()
        requested_at = self._stream_requested_at or datetime.now(UTC)
        self._stream_requested_at = None
        if error is not None:
            self._diagnostics.transcription_finished(
                audio, error, requested_at,
            )
            self._report_error(
                "Streaming failed",
                error,
                lambda message: TranscriptionFailedEvent(error=message),
            )
            return
        if result is None:
            self._state_machine.transition_to(AppState.IDLE)
            return
        # The committed text was already pasted live by the streaming
        # writer; this only mirrors the batch path's terminal log and
        # completion event so streaming has the same observability.
        self._diagnostics.transcription_finished(
            audio, result, requested_at,
        )
        self._announce_streaming_empty(result)
        self._state_machine.transition_to(AppState.IDLE)
        self._event_bus.publish(
            TranscriptionCompletedEvent(
                text=result.text,
                transcription_seconds=result.transcription_seconds,
            )
        )

    def _announce_streaming_empty(self, result: TranscriptionResult) -> None:
        if result.is_empty:
            self._announce_empty_transcription(result)

    def _reset_cap(self) -> None:
        self._cap_seconds = self._max_recording_seconds()
        self._warn_seconds = self._warn_before_cap_seconds()
        self._cap_warned = False
        self._cap_remaining = self._cap_seconds

    def _schedule_cap_timer(self) -> None:
        self._cap_started_at = time.monotonic()
        self._cap_timer = threading.Timer(
            max(self._cap_remaining, 0.0),
            self._on_cap_timer_fired,
        )
        self._cap_timer.daemon = True
        self._cap_timer.start()
        if self._warn_seconds > 0 and not self._cap_warned:
            self._cap_warning_timer = threading.Timer(
                max(self._cap_remaining - self._warn_seconds, 0.0),
                self._on_cap_warning_due,
            )
            self._cap_warning_timer.daemon = True
            self._cap_warning_timer.start()

    def _cancel_cap_timer(self) -> None:
        for timer in (self._cap_timer, self._cap_warning_timer):
            if timer is not None:
                timer.cancel()
        self._cap_timer = None
        self._cap_warning_timer = None

    def _on_cap_warning_due(self) -> None:
        self._dispatcher.dispatch(self._warn_of_cap)

    def _warn_of_cap(self) -> None:
        with self._action_lock:
            if (
                self._state_machine.state != AppState.RECORDING
                or self._paused
                or self._cap_warned
            ):
                return
            self._cap_warned = True
            remaining = self._cap_remaining - (time.monotonic() - self._cap_started_at)
            self._event_bus.publish(
                RecordingCapWarningEvent(remaining_seconds=max(remaining, 0.0)),
            )

    def _on_cap_timer_fired(self) -> None:
        # Timer thread → worker thread, so the auto-stop takes the
        # same action lock the hotkey path uses. Without the
        # dispatcher hop, a cap firing mid-stop would race the lock.
        self._dispatcher.dispatch(self._handle_auto_stop)

    def _handle_auto_stop(self) -> None:
        with self._action_lock:
            if self._state_machine.state != AppState.RECORDING:
                # Cap raced a normal stop; the user already released
                # the hotkey or the state moved on. Nothing to do.
                return
            _logger.info(
                "Recording auto-stopped: %.0fs cap reached.",
                self._cap_seconds,
            )
            self._event_bus.publish(
                RecordingCappedEvent(
                    max_duration_seconds=self._cap_seconds,
                )
            )
            self._end_capture()

    def _stop_and_transcribe(self) -> None:
        requested_at = datetime.now(UTC)
        self._cancel_cap_timer()
        self._paused = False
        editing, self._editing = self._editing, None
        try:
            audio = self._recorder.stop_recording()
        except Exception as exc:
            self._report_error(
                "Failed to stop recording",
                exc,
                lambda message: ErrorOccurredEvent(error=message),
            )
            return
        self._state_machine.transition_to(AppState.TRANSCRIBING)
        self._event_bus.publish(RecordingStoppedEvent())
        if self._audio_is_skippable(audio):
            self._state_machine.transition_to(AppState.IDLE)
            return
        _logger.info(
            "Recording stopped: %.1fs of audio. Transcribing...",
            audio.duration_seconds,
        )
        self._event_bus.publish(
            TranscriptionStartedEvent(
                audio_seconds=audio.duration_seconds,
                estimated_seconds=self._estimate_seconds(audio.duration_seconds),
            )
        )
        self._dispatcher.dispatch(
            lambda: self._do_transcription(audio, requested_at, editing),
        )

    def _estimate_seconds(self, audio_seconds: float) -> float | None:
        transcribing = self._diagnostics.estimate_transcription_seconds(audio_seconds)
        if transcribing is None:
            return None
        return transcribing + self._writing_seconds(audio_seconds)

    def _audio_is_skippable(self, audio: AudioSegment | None) -> bool:
        if audio is None:
            _logger.info("Recording stopped: no audio captured.")
            return True
        if audio.duration_seconds < _MIN_DURATION_SECONDS:
            _logger.info(
                "Recording stopped: too short (%.2fs).", audio.duration_seconds
            )
            return True
        return False

    def _do_transcription(
        self,
        audio: AudioSegment,
        requested_at: datetime,
        editing: Selection | None = None,
    ) -> None:
        try:
            self._transcribe_and_publish(audio, requested_at, editing)
        except Exception as exc:
            self._diagnostics.transcription_finished(
                audio, exc, requested_at,
            )
            recoverable = editing is None and self._keep_unfinished(audio)
            self._report_error(
                "Transcription failed",
                exc,
                lambda message: TranscriptionFailedEvent(
                    error=message, recoverable=recoverable,
                ),
            )

    def _transcribe_and_publish(
        self,
        audio: AudioSegment,
        requested_at: datetime,
        editing: Selection | None = None,
    ) -> None:
        result = self._transcribe(audio)
        # The output stage runs the text pipeline, writing model included,
        # on this thread; timing it apart keeps the transcription estimate
        # about transcribing alone.
        output_started = time.monotonic()
        trace = None
        if editing is None:
            begin_trace()
            try:
                self._deliver_transcribed_text(result)
            finally:
                trace = end_trace()
        else:
            self._deliver_voice_edit(result, editing)
        self._diagnostics.transcription_finished(
            audio, result, requested_at,
            kind="dictation" if editing is None else "edit",
            output_seconds=time.monotonic() - output_started,
            trace=trace,
        )
        self._state_machine.transition_to(AppState.IDLE)
        _logger.debug("post-transcribe: state -> IDLE")
        self._event_bus.publish(
            TranscriptionCompletedEvent(
                text=result.text,
                transcription_seconds=result.transcription_seconds,
            )
        )
        _logger.debug("post-transcribe: completion event published")

    def _deliver_transcribed_text(self, result: TranscriptionResult) -> None:
        if result.is_empty:
            self._announce_empty_transcription(result)
            return
        _logger.debug("output.output: calling")
        try:
            self._output.output(result.text)
        except ClipboardWriteError as exc:
            self._handle_clipboard_write_failure(exc)
        _logger.debug("output.output: returned")

    def _deliver_voice_edit(
        self, result: TranscriptionResult, editing: Selection,
    ) -> None:
        if result.is_empty:
            self._announce_empty_transcription(result)
            return
        try:
            replacement = self._voice_edit_replacement(editing, result.text)
        except VoiceEditError as exc:
            self._notify(str(exc))
            return
        try:
            self._output.output_verbatim(replacement)
        except ClipboardWriteError as exc:
            self._handle_clipboard_write_failure(exc)

    def _voice_edit_replacement(self, editing: Selection, instruction: str) -> str:
        if self._voice_edit is None:
            raise VoiceEditError("Voice editing is not available.")
        return self._voice_edit.replacement(editing, instruction)

    def _notify(self, message: str) -> None:
        _logger.info("Writing notice: %s", message)
        self._event_bus.publish(WritingNoticeEvent(message=message))

    def _announce_empty_transcription(self, result: TranscriptionResult) -> None:
        self._event_bus.publish(
            TranscriptionEmptyEvent(
                transcription_seconds=result.transcription_seconds,
            )
        )

    def _handle_clipboard_write_failure(self, error: ClipboardWriteError) -> None:
        _logger.warning("Clipboard write failed: %s", error)
        self._event_bus.publish(ErrorOccurredEvent(error=str(error)))

    def _report_error(
        self,
        operation: str,
        error: Exception,
        make_event: Callable[[str], DomainEvent],
    ) -> None:
        _logger.error("%s: %s", operation, error)
        self._state_machine.transition_to(AppState.ERROR)
        self._event_bus.publish(make_event(str(error)))
        self._state_machine.transition_to(AppState.IDLE)
