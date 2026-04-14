import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

from voicetotext.core.action_dispatcher import ActionDispatcher
from voicetotext.transcription.engine_diagnostics import count_live_adapters, vram_snapshot
from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


_logger = logging.getLogger(__name__)


EngineLoader = Callable[[], TranscriptionEngine]


class EngineHotSwap(Protocol):
    """Structural interface for an engine hot-swap strategy.

    Two implementations exist:
    :class:`EngineHotSwapCoordinator` (normal runtime) and
    :class:`NullEngineHotSwap` (first-run setup, where no engine is
    loaded yet and swapping is meaningless).
    """

    def start_swap(self, model_name: str) -> None: ...
    def is_busy(self) -> bool: ...


class NullEngineHotSwap:
    """Null Object (GoF) for first-run setup.

    During first-run the app has no engine to swap — the first engine
    load happens in :func:`voicetotext.bootstrap.build_app` after the
    setup window closes. This null object lets the facade still receive
    a valid collaborator without introducing ``None`` into its API.
    Calling :meth:`start_swap` is a programming error and raises.
    """

    def is_busy(self) -> bool:
        return False

    def start_swap(self, model_name: str) -> None:
        raise RuntimeError(
            "engine hot-swap is unavailable during first-run setup"
        )


class EngineHotSwapCoordinator:
    """Runs a new engine load on a background thread and swaps it in.

    The existing synchronous hot-swap in bootstrap freezes the caller
    thread (tray or settings window daemon) for 5–30 seconds with no
    feedback. This coordinator offloads the load and publishes
    :class:`EngineLoadStartedEvent`,
    :class:`EngineLoadCompletedEvent`, or :class:`EngineLoadFailedEvent`
    so the UI can render a loading state.

    Only one swap may run at a time. Starting a second swap while the
    first is still in-flight raises :class:`RuntimeError` — callers that
    expose this via UI should disable the relevant control while
    :meth:`is_busy` is true.
    """

    def __init__(
        self,
        use_case: RecordAndTranscribeUseCase,
        event_bus: EventBus,
        loader: EngineLoader,
        action_dispatcher: ActionDispatcher,
    ):
        self._use_case = use_case
        self._event_bus = event_bus
        self._loader = loader
        self._dispatcher = action_dispatcher
        self._lock = threading.Lock()
        self._busy = False

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def start_swap(self, model_name: str) -> None:
        with self._lock:
            if self._busy:
                raise RuntimeError("engine swap already in progress")
            self._busy = True
        self._dispatcher.dispatch(lambda: self._run_swap(model_name))

    def _run_swap(self, model_name: str) -> None:
        _logger.info(
            "[swap] start target=%s | adapters=%d | %s",
            model_name,
            count_live_adapters(),
            vram_snapshot(),
        )
        self._event_bus.publish(
            EngineLoadStartedEvent(model_name=model_name)
        )
        try:
            self._use_case.dispose_current_engine()
            _logger.info(
                "[swap] old engine disposed | adapters=%d | %s",
                count_live_adapters(),
                vram_snapshot(),
            )

            t0 = time.monotonic()
            new_engine = self._loader()
            _logger.info(
                "[swap] loader done in %.2fs | adapters=%d | %s",
                time.monotonic() - t0,
                count_live_adapters(),
                vram_snapshot(),
            )

            self._use_case.set_engine(new_engine)
            _logger.info(
                "[swap] engine installed | adapters=%d | %s",
                count_live_adapters(),
                vram_snapshot(),
            )
        except Exception as exc:
            _logger.exception("[swap] failed for '%s'", model_name)
            with self._lock:
                self._busy = False
            self._event_bus.publish(
                EngineLoadFailedEvent(
                    model_name=model_name, error=str(exc)
                )
            )
            return

        _logger.info(
            "[swap] complete '%s' is live | adapters=%d | %s",
            model_name,
            count_live_adapters(),
            vram_snapshot(),
        )
        with self._lock:
            self._busy = False
        self._event_bus.publish(
            EngineLoadCompletedEvent(model_name=model_name)
        )
