"""Learns spellings from re-dictating selected words, the way a misheard name gets fixed.

Selecting a word the engine got wrong and dictating over it is an attempt
to fix it, and doing it again over the result is another attempt at the
same spot. While such a fix is open the box is read every half second;
once the message is sent, the window changes or a few minutes pass, the
last reading says what finally sits where the word was. When that differs
from what the engine wrote, each wrong spelling and the final one go to
the learner as a correction, which teaches the pair once two separate
fixes agree. The text read from the box is only compared, never kept.

Edit capture cannot see these fixes: the selection is usually made with
the mouse, which its keystroke replay cannot follow.
"""

import itertools
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from skryb.output.caret_context import CaretContext

_logger = logging.getLogger(__name__)

_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
_MOST_WORDS = 3
_ANCHOR_CHARS = 16
# The fix itself is short; anything wider between the anchors is a rewrite.
_WIDEST_FIX_CHARS = 80
_POLL_SECONDS = 0.5
_OPEN_SECONDS = 180.0
_READ_CHARS = 8000
_MOST_OPEN_FIXES = 20

Learn = Callable[[str, str, str], None]


def _words(text: str) -> list[str]:
    return [word.casefold() for word in _WORD.findall(text)]


def _same_words(first: str, second: str) -> bool:
    return _words(first) == _words(second)


def replacement_between(text: str, before: str, after: str) -> str | None:
    """What sits between the two anchors in ``text``; ``None`` once either is gone."""
    starts = (
        [0] if not before
        else [match.end() for match in re.finditer(re.escape(before), text)]
    )
    found = []
    for start in starts:
        if not after:
            rest = text[start:]
            if len(rest) <= _WIDEST_FIX_CHARS:
                found.append(rest)
            continue
        end = text.find(after, start, start + _WIDEST_FIX_CHARS + len(after))
        if end != -1:
            found.append(text[start:end])
    # An anchor can repeat earlier in the box; the tightest pair is the fix.
    return min(found, key=len) if found else None


@dataclass
class _Fix:
    hwnd: int
    before: str
    after: str
    original: str
    attempts: list[str]
    changed_at: float
    final: str | None = None
    id: int = field(default_factory=itertools.count().__next__)

    def continues(self, hwnd: int, context: CaretContext) -> bool:
        return (
            hwnd == self.hwnd
            and context.before.endswith(self.before)
            and context.after.startswith(self.after)
            and _same_words(context.selected, self.attempts[-1])
        )


class RedictationTracker:
    """Follows fixes made by dictating over selected words until they settle."""

    def __init__(
        self,
        learn: Learn,
        foreground: Callable[[], int],
        read_text: Callable[[int], str | None],
        enabled: Callable[[], bool] = lambda: True,
        clock: Callable[[], float] = time.monotonic,
        spawn: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self._learn = learn
        self._foreground = foreground
        self._read_text = read_text
        self._enabled = enabled
        self._clock = clock
        self._spawn = spawn or _daemon_thread
        self._lock = threading.Lock()
        self._read: CaretContext | None = None
        self._fixes: list[_Fix] = []
        self._polling = False

    def caret_read(self, context: CaretContext) -> None:
        """The caret reading for the dictation about to be pasted."""
        with self._lock:
            self._read = context

    def pasting(self, text: str) -> None:
        """The dictation going in now, over the selection the last reading saw."""
        with self._lock:
            context, self._read = self._read, None
        if context is None or not self._enabled():
            return
        selected = context.selected.strip()
        if not selected or len(_words(selected)) > _MOST_WORDS or not _words(text):
            return
        hwnd = self._foreground()
        with self._lock:
            self._open_or_continue(hwnd, context, selected, text.strip())
            start = not self._polling
            self._polling = True
        if start:
            self._spawn(self._poll_until_settled)

    def poll(self) -> bool:
        """Read the box once for every open fix; whether any are still open."""
        with self._lock:
            fixes = list(self._fixes)
        if not fixes:
            return False
        foreground, now = self._foreground(), self._clock()
        text = self._read_text(_READ_CHARS) if any(f.hwnd == foreground for f in fixes) else None
        learn_from, dropped = [], []
        for fix in fixes:
            # Moving on (another window, or minutes of quiet) or sending (the
            # box empties) settles a fix on its last reading. Text rewritten
            # around it or a box that stops answering leaves no trustworthy
            # final word, so that fix is dropped unlearned.
            if fix.hwnd != foreground or now - fix.changed_at >= _OPEN_SECONDS:
                learn_from.append(fix)
                continue
            if text is None:
                dropped.append(fix)
                continue
            between = replacement_between(text, fix.before, fix.after)
            if between is not None:
                fix.final = between
            elif text.strip():
                dropped.append(fix)
            else:
                learn_from.append(fix)
        with self._lock:
            self._fixes = [fix for fix in self._fixes if fix not in learn_from and fix not in dropped]
            still_open = bool(self._fixes)
        for fix in learn_from:
            self._settle(fix)
        return still_open

    def _open_or_continue(
        self, hwnd: int, context: CaretContext, selected: str, text: str,
    ) -> None:
        for fix in self._fixes:
            if fix.continues(hwnd, context):
                fix.attempts.append(text)
                fix.changed_at = self._clock()
                return
        self._fixes.append(_Fix(
            hwnd=hwnd,
            before=context.before[-_ANCHOR_CHARS:],
            after=context.after[:_ANCHOR_CHARS],
            original=selected,
            attempts=[text],
            changed_at=self._clock(),
        ))
        del self._fixes[:-_MOST_OPEN_FIXES]

    def _settle(self, fix: _Fix) -> None:
        final = fix.final
        if final is None or not _words(final) or _same_words(final, fix.original):
            return
        wrong = dict.fromkeys(
            spelling for spelling in (fix.original, *fix.attempts)
            if not _same_words(spelling, final)
        )
        for spelling in wrong:
            self._learn(
                fix.before + spelling + fix.after,
                fix.before + final + fix.after,
                f"redictation-{fix.id}",
            )
        _logger.info("A re-dictated fix settled: %d spelling(s) to learn from.", len(wrong))

    def _poll_until_settled(self) -> None:
        try:
            while self.poll():
                time.sleep(_POLL_SECONDS)
        except Exception as exc:  # noqa: BLE001 - learning must never break dictation
            _logger.warning("Following a re-dictated fix failed: %s", exc)
        finally:
            with self._lock:
                self._polling = False
                restart = bool(self._fixes)
            if restart:
                self._spawn(self._poll_until_settled)


def _daemon_thread(target: Callable[[], None]) -> None:
    threading.Thread(target=target, name="redictation", daemon=True).start()
