"""Load the speech model from the moment startup knows which one it is.

The tray used to be built first and the model loaded after it, so a launch
waited for both in turn. Loading starts here instead, while the main thread
is still building the tray, and the loader the tray starts picks the result
up (``bootstrap._take_preloaded``).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable

from voicetotext.core.startup_log import milestone
from voicetotext.transcription.engine import TranscriptionEngine

_logger = logging.getLogger(__name__)

# Loading the model takes the GIL in thousands of short bursts. With the
# default 5 ms switch interval each burst waits out the main thread's
# imports, which took the tokenizer from 0.1 s to 3.3 s; at 0.5 ms it
# loads at idle speed. Restored as soon as the model is warm.
_LOADING_SWITCH_INTERVAL = 0.0005


class EnginePreload:
    """One engine, built, loaded and warmed on its own thread.

    *build* may convert the model on first use, which takes minutes, so
    it runs here and never on the thread building the tray.
    """

    def __init__(self, build: Callable[[], TranscriptionEngine]) -> None:
        self._build = build
        self._engine: TranscriptionEngine | None = None
        self._done = threading.Event()
        self._error: BaseException | None = None
        self._warmed = False

    @classmethod
    def start(cls, build: Callable[[], TranscriptionEngine]) -> EnginePreload:
        preload = cls(build)
        threading.Thread(
            target=preload._run, daemon=True, name="engine-preload",
        ).start()
        return preload

    @property
    def warmed(self) -> bool:
        """Whether the warmup ran cleanly; if not, the loader runs its own."""
        return self._warmed

    def wait(self) -> TranscriptionEngine:
        """The loaded engine, or the error building or loading it raised."""
        self._done.wait()
        if self._error is not None:
            raise self._error
        assert self._engine is not None
        return self._engine

    def _run(self) -> None:
        milestone("engine-preload-started")
        previous = sys.getswitchinterval()
        sys.setswitchinterval(_LOADING_SWITCH_INTERVAL)
        try:
            self._load_then_warm()
        except Exception as exc:  # noqa: BLE001 - the waiter classifies it
            self._error = exc
        finally:
            sys.setswitchinterval(previous)
            self._done.set()

    def _load_then_warm(self) -> None:
        engine = self._build()
        engine.load_model()
        self._engine = engine
        milestone("model-loaded")
        started_at = time.monotonic()
        try:
            engine.warmup()
        except Exception:
            # The loader repeats the warmup through ``_warmup_safely``,
            # which decides whether the failure is fatal.
            _logger.debug("Preload warmup failed; the loader retries it", exc_info=True)
            return
        self._warmed = True
        _logger.info(
            "Engine warmup complete in %.2fs", time.monotonic() - started_at,
        )
