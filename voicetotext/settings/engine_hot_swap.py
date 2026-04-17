import concurrent.futures
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from voicetotext.core.action_dispatcher import ActionDispatcher
from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.core.exceptions import EngineSwapError
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_diagnostics import (
    count_live_adapters,
    vram_snapshot,
)
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


_logger = logging.getLogger(__name__)


EngineLoader = Callable[[], TranscriptionEngine]


class EngineHotSwap(ABC):
    """Port for an engine hot-swap strategy.

    Two implementations exist: :class:`EngineHotSwapCoordinator` for
    normal runtime and :class:`NullEngineHotSwap` for first-run setup,
    where no engine is loaded yet and swapping is meaningless.
    """

    @abstractmethod
    def start_swap(self, model_name: str) -> None: ...

    @abstractmethod
    def is_busy(self) -> bool: ...


class NullEngineHotSwap(EngineHotSwap):
    """Null Object (GoF) for first-run setup.

    During first-run the app has no engine to swap; the first engine
    load happens in :func:`voicetotext.bootstrap.build_app` after the
    setup window closes. This null object lets the facade still receive
    a valid collaborator without introducing ``None`` into its API.
    Calling :meth:`start_swap` is a programming error and raises.
    """

    def is_busy(self) -> bool:
        return False

    def start_swap(self, model_name: str) -> None:
        raise EngineSwapError(
            "engine hot-swap is unavailable during first-run setup"
        )


class EngineHotSwapCoordinator(EngineHotSwap):
    """Runs a new engine load on a background thread and swaps it in.

    The existing synchronous hot-swap in bootstrap froze the caller
    thread (tray or settings window daemon) for 5 to 30 seconds with no
    feedback. This coordinator offloads the load and publishes
    :class:`EngineLoadStartedEvent`, :class:`EngineLoadCompletedEvent`,
    or :class:`EngineLoadFailedEvent` so the UI can render a loading
    state.

    Only one swap may run at a time. Starting a second swap while the
    first is still in-flight raises :class:`EngineSwapError`; callers
    that expose this via UI should disable the relevant control while
    :meth:`is_busy` is true.
    """

    def __init__(
        self,
        use_case: RecordAndTranscribeUseCase,
        event_bus: EventBus,
        loader: EngineLoader,
        action_dispatcher: ActionDispatcher,
        timeout_seconds: float | None = None,
    ) -> None:
        self._use_case = use_case
        self._event_bus = event_bus
        self._loader = loader
        self._dispatcher = action_dispatcher
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._busy = False

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def start_swap(self, model_name: str) -> None:
        with self._lock:
            if self._busy:
                raise EngineSwapError("engine swap already in progress")
            self._busy = True
        self._dispatcher.dispatch(lambda: self._run_swap(model_name))

    def _run_swap(self, model_name: str) -> None:
        self._announce_swap_started(model_name)
        failure = self._try_swap(model_name)
        self._release_busy_flag()
        self._publish_swap_outcome(model_name, failure)

    def _try_swap(self, model_name: str) -> Exception | None:
        try:
            self._perform_swap()
            return None
        except Exception as exc:
            _logger.exception("[swap] failed for '%s'", model_name)
            return exc

    def _perform_swap(self) -> None:
        new_engine = self._load_new_engine()
        self._install_or_discard(new_engine)

    def _load_new_engine(self) -> TranscriptionEngine:
        started_at = time.monotonic()
        new_engine = self._invoke_loader()
        self._log_phase(f"loader done in {time.monotonic() - started_at:.2f}s")
        return new_engine

    def _invoke_loader(self) -> TranscriptionEngine:
        if self._timeout_seconds is None:
            return self._loader()
        return self._invoke_loader_with_timeout()

    def _invoke_loader_with_timeout(self) -> TranscriptionEngine:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engine-load"
        )
        try:
            future = executor.submit(self._loader)
            try:
                return future.result(timeout=self._timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"engine load timed out after {self._timeout_seconds:.0f}s"
                ) from None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _install_or_discard(self, new_engine: TranscriptionEngine) -> None:
        try:
            self._use_case.set_engine(new_engine)
        except EngineSwapError:
            new_engine.dispose()
            raise
        self._log_phase("engine installed")

    def _announce_swap_started(self, model_name: str) -> None:
        self._log_phase(f"start target={model_name}")
        self._event_bus.publish(EngineLoadStartedEvent(model_name=model_name))

    def _publish_swap_outcome(
        self, model_name: str, failure: Exception | None
    ) -> None:
        if failure is None:
            self._log_phase(f"complete '{model_name}' is live")
            self._event_bus.publish(
                EngineLoadCompletedEvent(model_name=model_name)
            )
            return
        self._event_bus.publish(
            EngineLoadFailedEvent(model_name=model_name, error=str(failure))
        )

    def _release_busy_flag(self) -> None:
        with self._lock:
            self._busy = False

    def _log_phase(self, message: str) -> None:
        _logger.info(
            "[swap] %s | adapters=%d | %s",
            message,
            count_live_adapters(),
            vram_snapshot(),
        )
