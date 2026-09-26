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


def prune_in_background(diagnostics_dir: Path, retention_days: int) -> None:
    if retention_days <= 0:
        return
    threading.Thread(
        target=_prune_quietly,
        args=(diagnostics_dir, retention_days),
        name="history-prune",
        daemon=True,
    ).start()


def prune_once_ready(
    event_bus: EventBus, diagnostics_dir: Path, retention_days: Callable[[], int],
) -> None:
    """Prune once, as soon as the app leaves ``LOADING``, off the startup path."""

    def on_state_changed(event: DomainEvent) -> None:
        if not isinstance(event, StateChangedEvent):
            return
        if event.old_state != AppState.LOADING.value:
            return
        event_bus.unsubscribe(StateChangedEvent, on_state_changed)
        prune_in_background(diagnostics_dir, retention_days())

    event_bus.subscribe(StateChangedEvent, on_state_changed)


def _prune_quietly(diagnostics_dir: Path, retention_days: int) -> None:
    try:
        pruned = prune_history(diagnostics_dir, retention_days, utc_now())
    except Exception:  # a daemon thread would drop it silently
        _logger.exception("History pruning failed.")
        return
    if pruned:
        _logger.info("Pruned %d history entries older than %d days.", pruned, retention_days)
