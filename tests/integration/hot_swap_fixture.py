"""Integration-test DSL harness for the engine hot-swap scenarios.

Wires the real :class:`RecordAndTranscribeUseCase`,
:class:`AppStateMachine`, :class:`SimpleEventBus`,
:class:`EngineHotSwapCoordinator`, :class:`ThreadedActionDispatcher`,
and :class:`TransformersWhisperAdapter` (tiny Whisper model) with
hand-rolled fakes at the recorder/output boundary only.  Tests reach
in through this harness and never touch the wired components directly.

Unlike the in-process ``VoiceToTextApp`` in ``tests/acceptance``, this
harness uses **real threading** (``ThreadedActionDispatcher``) and a
**real transcription engine** so that the adapter's post-dispose
behaviour - raising ``TranscriptionError("Model not loaded")`` - is
observable.  That behaviour is essential for reproducing the
record-during-hot-swap race, which fake engines silently mask.
"""

import threading
import time
from typing import Optional

from voicetotext.core.action_dispatcher import ThreadedActionDispatcher
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    StateChangedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.paths import AppPaths
from voicetotext.core.state_machine import AppStateMachine
from voicetotext.settings.engine_hot_swap import EngineHotSwapCoordinator
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.transformers_whisper_adapter import (
    TransformersWhisperAdapter,
)
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase

from tests.builders import an_audio_segment
from tests.fakes import (
    FakeAudioRecorder,
    FakeTextOutputHandler,
    FakeTranscriptionEngine,
)


class BlockingLoader:
    """Controllable loader that lets the test land actions mid-swap.

    The coordinator invokes this as its engine loader.  On call, the
    loader signals :attr:`started` and then blocks on :attr:`release`
    until the test explicitly releases it.  This gives the test a
    deterministic window in which to observe and drive intermediate
    state - no sleeps, no races, no flakiness.
    """

    def __init__(self, result: TranscriptionEngine) -> None:
        self._result = result
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self) -> TranscriptionEngine:
        self.started.set()
        if not self.release.wait(timeout=10.0):
            raise RuntimeError(
                "BlockingLoader: test did not release the loader within 10s"
            )
        return self._result


class HotSwapIntegrationApp:
    """DSL harness for integration-testing the hot-swap flow.

    Methods read as user actions (``user_presses_hotkey``,
    ``user_triggers_engine_swap_to``) or observable outcomes
    (``transcription_count``, ``transcription_errors``).  Threading is
    real, so every ``When`` step that schedules work on a background
    thread has a matching ``wait_for_*`` step that blocks the main
    thread until the observable state has caught up.
    """

    def __init__(self) -> None:
        self._bus = SimpleEventBus()
        self._state_machine = AppStateMachine(self._bus)
        self._recorder = FakeAudioRecorder()
        self._output = FakeTextOutputHandler()
        self._dispatcher = ThreadedActionDispatcher()
        self._engine: Optional[TransformersWhisperAdapter] = None
        self._use_case: Optional[RecordAndTranscribeUseCase] = None
        self._hot_swap: Optional[EngineHotSwapCoordinator] = None
        self._blocking_loader: Optional[BlockingLoader] = None

        self._transcription_count = 0
        self._transcription_errors: list[str] = []
        self._transcription_outcome = threading.Event()
        self._swap_finished = threading.Event()
        self._state_changed = threading.Event()

        self._bus.subscribe(
            StateChangedEvent, self._on_state_changed
        )
        self._bus.subscribe(
            TranscriptionCompletedEvent, self._on_transcription_completed
        )
        self._bus.subscribe(
            TranscriptionFailedEvent, self._on_transcription_failed
        )
        self._bus.subscribe(
            EngineLoadCompletedEvent, self._on_swap_finished
        )
        self._bus.subscribe(
            EngineLoadFailedEvent, self._on_swap_finished
        )

    # ------------------------------------------------------------------
    # Given - programme the world the user sees
    # ------------------------------------------------------------------

    def load_tiny_model_for_real(self) -> None:
        device = self._pick_device()
        engine = TransformersWhisperAdapter(
            model_size="tiny",
            device=device,
            beam_size=1,
            # Match the production config: a set language takes a
            # Whisper generate() code path that accepts mel features
            # shorter than 3000 frames.  language=None falls into the
            # auto-detect + long-form path which rejects short input,
            # and that's an isolated-test artefact, not a production
            # bug.
            language="en",
            download_root=str(AppPaths().models_dir),
        )
        engine.load_model()
        self._engine = engine
        self._use_case = RecordAndTranscribeUseCase(
            recorder=self._recorder,
            engine=engine,
            output=self._output,
            state_machine=self._state_machine,
            event_bus=self._bus,
            action_dispatcher=self._dispatcher,
        )
        self._hot_swap = EngineHotSwapCoordinator(
            use_case=self._use_case,
            event_bus=self._bus,
            loader=self._default_loader,
            action_dispatcher=self._dispatcher,
        )

    def microphone_will_capture_silence_of(self, seconds: float) -> None:
        self._recorder.will_return(
            an_audio_segment().silent().with_duration(seconds=seconds).build()
        )

    def next_swap_will_use_a_blocking_loader(self) -> None:
        assert self._use_case is not None, (
            "Call load_tiny_model_for_real() first"
        )
        self._blocking_loader = BlockingLoader(FakeTranscriptionEngine())
        self._hot_swap = EngineHotSwapCoordinator(
            use_case=self._use_case,
            event_bus=self._bus,
            loader=self._blocking_loader,
            action_dispatcher=self._dispatcher,
        )

    # ------------------------------------------------------------------
    # When - the user's actions
    # ------------------------------------------------------------------

    def user_presses_hotkey(self) -> None:
        assert self._use_case is not None
        self._use_case.on_hotkey_activated()
        self._wait_for_state("recording", timeout=5.0)

    def user_releases_hotkey(self) -> None:
        assert self._use_case is not None
        self._use_case.on_hotkey_deactivated()

    def user_triggers_engine_swap_to(self, model_name: str) -> None:
        assert self._hot_swap is not None
        self._swap_finished.clear()
        self._hot_swap.start_swap(model_name)

    def wait_until_swap_loader_is_blocked(self, timeout: float) -> None:
        assert self._blocking_loader is not None, (
            "Call next_swap_will_use_a_blocking_loader() first"
        )
        if not self._blocking_loader.started.wait(timeout=timeout):
            raise TimeoutError(
                f"BlockingLoader did not start within {timeout}s"
            )

    def release_swap_loader(self) -> None:
        assert self._blocking_loader is not None
        self._blocking_loader.release.set()

    def wait_for_swap_to_finish(self, timeout: float) -> None:
        if not self._swap_finished.wait(timeout=timeout):
            raise TimeoutError(
                f"Swap did not finish within {timeout}s"
            )

    def wait_for_next_transcription_outcome(self, timeout: float) -> None:
        if not self._transcription_outcome.wait(timeout=timeout):
            raise TimeoutError(
                f"No transcription outcome within {timeout}s. "
                f"Completed={self._transcription_count}, "
                f"errors={self._transcription_errors}"
            )
        self._transcription_outcome.clear()

    # ------------------------------------------------------------------
    # Then - observable outcomes
    # ------------------------------------------------------------------

    @property
    def transcription_count(self) -> int:
        return self._transcription_count

    @property
    def transcription_errors(self) -> list[str]:
        return list(self._transcription_errors)

    @property
    def typed_text(self) -> list[str]:
        return list(self._output.outputs)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        # Order matters: release the blocking loader first so the
        # swap thread unwinds cleanly, then dispose the real engine
        # so torch/transformers objects are released while Python is
        # still alive, then force garbage collection to drop any
        # lingering references.  Skipping this leaves sentencepiece
        # state alive into interpreter finalization and segfaults
        # on shutdown.
        if self._blocking_loader is not None:
            self._blocking_loader.release.set()
        if self._use_case is not None:
            try:
                self._use_case.dispose()
            except Exception:
                pass
        self._engine = None
        self._use_case = None
        self._hot_swap = None
        self._blocking_loader = None
        import gc
        gc.collect()
        # Catch OSError too - on Windows, re-importing torch during
        # shutdown can fail with WinError 1455 ("paging file too small")
        # if the host is low on swap, which would otherwise mask the
        # test's own exit code with a harness traceback.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, OSError):
            pass

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _default_loader(self) -> TranscriptionEngine:
        return FakeTranscriptionEngine()

    def _pick_device(self) -> str:
        # CPU tiny takes ~60s per call which is too slow for an
        # integration test.  Use CUDA when available (ROCm on Windows
        # is exposed as CUDA via HIP translation) so each transcribe()
        # call completes in under a second.
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _on_state_changed(self, event: StateChangedEvent) -> None:
        self._state_changed.set()

    def _on_transcription_completed(
        self, event: TranscriptionCompletedEvent
    ) -> None:
        self._transcription_count += 1
        self._transcription_outcome.set()

    def _on_transcription_failed(
        self, event: TranscriptionFailedEvent
    ) -> None:
        self._transcription_errors.append(event.error)
        self._transcription_outcome.set()

    def _on_swap_finished(self, event: object) -> None:
        self._swap_finished.set()

    def _wait_for_state(self, state_name: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._state_machine.state.value == state_name:
                return
            self._state_changed.wait(timeout=0.05)
            self._state_changed.clear()
        raise TimeoutError(
            f"State did not reach '{state_name}' within {timeout}s "
            f"(current: {self._state_machine.state.value})"
        )
