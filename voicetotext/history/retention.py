import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import DomainEvent, StateChangedEvent
from voicetotext.core.state_machine import AppState
from voicetotext.core.timestamps import utc_now
from voicetotext.history.entries import load_history
from voicetotext.history.management import delete_entries

_logger = logging.getLogger(__name__)

RETENTION_CHOICES = (0, 7, 30, 90)
_PRUNE_INTERVAL_SECONDS = 6 * 60 * 60


def history_retention_days(config: ConfigurationStore) -> int:
    """Days to keep history for; 0, and anything unreadable, keeps it forever."""
    value = config.get("history.retention_days", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def prune_history(diagnostics_dir: Path, retention_days: int, now: datetime) -> int:
    """Delete entries older than *retention_days*; an entry exactly that old stays."""
    if retention_days <= 0:
        return 0
    cutoff = now - timedelta(days=retention_days)
    expired = [entry for entry in load_history(diagnostics_dir) if entry.at < cutoff]
    return delete_entries(diagnostics_dir, expired, now) if expired else 0


def _prune_until_now(diagnostics_dir: Path, retention_days: int) -> int:
    return prune_history(diagnostics_dir, retention_days, utc_now())


class HistoryPruner:
    """Ages history out once the app is ready, then every six hours it stays open.

    Each run happens on a daemon ``threading.Timer``, never on the startup
    path or the GUI thread. Keeping history forever stops the cycle; a new
    retention setting restarts it through :meth:`prune_now`. ``dispose``
    cancels the pending run when the app quits.
    """

    def __init__(
        self,
        diagnostics_dir: Path,
        retention_days: Callable[[], int],
        *,
        interval_seconds: float = _PRUNE_INTERVAL_SECONDS,
        prune: Callable[[Path, int], int] = _prune_until_now,
    ) -> None:
        self._diagnostics_dir = diagnostics_dir
        self._retention_days = retention_days
        self._interval_seconds = interval_seconds
        self._prune = prune
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._disposed = False

    def start_when_ready(self, event_bus: EventBus) -> None:
        """Prune for the first time once the app leaves ``LOADING``."""

        def on_state_changed(event: DomainEvent) -> None:
            if not isinstance(event, StateChangedEvent):
                return
            if event.old_state != AppState.LOADING.value:
                return
            event_bus.unsubscribe(StateChangedEvent, on_state_changed)
            self.prune_now()

        event_bus.subscribe(StateChangedEvent, on_state_changed)

    def prune_now(self) -> None:
        """Prune on a background thread now, then every interval after it."""
        with self._lock:
            self._cancel_timer()
            if self._disposed or self._retention_days() <= 0:
                return
            self._arm(0.0)

    def dispose(self) -> None:
        with self._lock:
            self._disposed = True
            self._cancel_timer()

    def _arm(self, delay_seconds: float) -> None:
        timer = threading.Timer(delay_seconds, self._run)
        timer.name = "history-prune"
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _run(self) -> None:
        days = self._retention_days()
        if days > 0:
            self._prune_quietly(days)
        with self._lock:
            # A newer ``prune_now`` or ``dispose`` has taken over the cycle.
            if self._timer is not threading.current_thread():
                return
            self._timer = None
            if not self._disposed and self._retention_days() > 0:
                self._arm(self._interval_seconds)

    def _prune_quietly(self, days: int) -> None:
        try:
            pruned = self._prune(self._diagnostics_dir, days)
        except Exception:  # a daemon thread would drop it silently
            _logger.exception("History pruning failed.")
            return
        if pruned:
            _logger.info("Pruned %d history entries older than %d days.", pruned, days)
