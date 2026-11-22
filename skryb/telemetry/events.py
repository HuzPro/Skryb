"""Opt-in usage data: which pages were opened, which controls were used and which settings were saved.

Never what was said. Every event is checked on its way in: its kind is one of three, and its name and
any text value must be a short lowercase token, so a sentence, a path or a person's name cannot pass.
Everything stays in one file on this computer until its owner exports it.
"""

import json
import math
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path

from skryb.core.timestamps import utc_now
from skryb.diagnostics.jsonl import JsonlWriter

Json = dict[str, object]

FILE_NAME = "telemetry.jsonl"
FORMAT = "skryb-telemetry/1"
KINDS = frozenset({"page", "click", "setting"})
# ponytail: one flat file, read whole. Past this many events the oldest go; a database if it ever needs more.
MAX_EVENTS = 100_000
MAX_BATCH = 500
# The shell sends events in batches, each saying how long ago it happened; older than this is a wrong clock.
MAX_AGO_MS = 3_600_000
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-]{0,63}")

ABOUT = (
    "Opt-in usage data from one Skryb install, exported by its owner. It holds no dictated text, audio, "
    "names or file paths. 'events' is every recorded event, oldest first: 'at' is local time with its offset from UTC; 'kind' is "
    "'page' (a page was left; 'name' is the page and 'ms' the time spent on it), 'click' (a control was "
    "used; 'name' is page.control, the control named by its purpose or its icon) or 'setting' (a setting "
    "was saved; 'value' is its new value when that is a number, a switch or a fixed choice). 'summary' "
    "counts the same events by kind:name, most used first. 'settings', 'hardware', 'engine' and "
    "'dictation' describe the install at the time of export; 'dictation' is counts of words and seconds."
)


def scrub(event: object, now: datetime) -> Json | None:
    """*event* with only what telemetry may hold, stamped with when it happened; None when it isn't one."""
    if not isinstance(event, dict):
        return None
    kind, name = event.get("kind"), _token(event.get("name"))
    if kind not in KINDS or name is None:
        return None
    clean: Json = {"at": _stamp(now, event.get("ago")), "kind": kind, "name": name}
    value = event.get("value")
    if isinstance(value, bool) or _is_number(value):
        clean["value"] = value
    elif (token := _token(value)) is not None:
        clean["value"] = token
    ms = event.get("ms")
    if _is_number(ms) and ms >= 0:
        clean["ms"] = round(ms)
    return clean


def _token(value: object) -> str | None:
    return value if isinstance(value, str) and _TOKEN.fullmatch(value) else None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stamp(now: datetime, ago: object) -> str:
    behind = ago if _is_number(ago) and 0 <= ago <= MAX_AGO_MS else 0
    return (now - timedelta(milliseconds=behind)).isoformat(timespec="seconds")


def key_of(event: Json) -> str:
    return f"{event['kind']}:{event['name']}"


class TelemetryStore:
    """The events file: appended to while telemetry is on, counted for the shell, and exported whole."""

    def __init__(
        self,
        path: Path,
        enabled: Callable[[], bool],
        clock: Callable[[], datetime] = lambda: utc_now().astimezone(),
    ) -> None:
        self._path = path
        self._enabled = enabled
        self._clock = clock
        self._writer = JsonlWriter(path)
        self._lock = threading.Lock()
        self._counts: Counter[str] | None = None

    def record(self, events: Iterable[object]) -> int:
        """Keep what passes :func:`scrub`; nothing at all while telemetry is off. Returns how many were kept."""
        if not self._enabled():
            return 0
        now = self._clock()
        batch = list(events)[:MAX_BATCH]
        clean = [event for event in (scrub(raw, now) for raw in batch) if event is not None]
        with self._lock:
            counts = self._loaded_counts()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer.append_many(clean)
            counts.update(key_of(event) for event in clean)
        return len(clean)

    def counts(self) -> dict[str, int]:
        """How often each kind:name has happened, most used first."""
        with self._lock:
            return dict(self._loaded_counts().most_common())

    def events(self) -> list[Json]:
        with self._lock:
            return self._read_events()

    def clear(self) -> None:
        with self._lock:
            self._path.unlink(missing_ok=True)
            self._counts = Counter()

    def export(self, path: Path, context: Json) -> int:
        """Write everything as one JSON file that explains itself; returns how many events it holds."""
        events = self.events()
        head = json.dumps(
            {"format": FORMAT, "about": ABOUT, "exported_at": self._clock().isoformat(timespec="seconds"),
             **context, "summary": summary(events)},
            ensure_ascii=False, indent=2,
        )
        lines = ",\n".join(
            "    " + json.dumps(event, ensure_ascii=False, separators=(",", ":")) for event in events
        )
        body = f'[\n{lines}\n  ]' if events else "[]"
        path.write_text(f'{head[:-2]},\n  "events": {body}\n}}\n', encoding="utf-8")
        return len(events)

    def _loaded_counts(self) -> Counter[str]:
        if self._counts is None:
            self._drop_oldest_beyond_limit()
            self._counts = Counter(key_of(event) for event in self._read_events())
        return self._counts

    def _drop_oldest_beyond_limit(self) -> None:
        if self._path.is_file():
            self._writer.rewrite(lambda lines: lines[-MAX_EVENTS:])

    def _read_events(self) -> list[Json]:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [event for event in map(_parse, lines) if event is not None]


def _parse(line: str) -> Json | None:
    """One stored event; None for a line cut short by a crash or edited by hand into something else."""
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) and _token(event.get("name")) and event.get("kind") in KINDS else None


def summary(events: list[Json]) -> Json:
    counts = Counter(key_of(event) for event in events)
    time_on_pages: Counter[str] = Counter()
    for event in events:
        if event["kind"] == "page" and _is_number(event.get("ms")):
            time_on_pages[str(event["name"])] += int(event["ms"])  # type: ignore[call-overload]
    stamps = sorted(str(event.get("at", "")) for event in events)
    return {
        "events": len(events),
        "first": stamps[0] if stamps else None,
        "last": stamps[-1] if stamps else None,
        "days_active": len({stamp[:10] for stamp in stamps}),
        "by_name": dict(counts.most_common()),
        "ms_on_pages": dict(time_on_pages.most_common()),
    }
