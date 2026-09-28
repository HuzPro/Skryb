"""What happened to one dictation between the engine and the paste.

History keeps it next to the transcription, so a writing model that made a
dictation worse can be seen doing so. The trace lives on the thread running
the dictation: the text pipeline, the writing model included, runs there
from start to finish, and a voice edit or a history retry on another thread
never mixes its text in.
"""

import threading
from dataclasses import dataclass

APPLIED = "applied"
REJECTED = "rejected"
TOO_SLOW = "too_slow"
SKIPPED = "skipped"

# The session log's names for a trace. The outcome fields hold no dictated
# text; the text fields are what a history deletion blanks.
TYPED_FIELD = "typed_text"
ANSWER_FIELD = "cleanup_answer"
CONTEXT_FIELD = "cleanup_context_before"
WINDOW_FIELD = "cleanup_window"
TRACE_TEXT_FIELDS = (TYPED_FIELD, ANSWER_FIELD, CONTEXT_FIELD, WINDOW_FIELD)
OUTCOME_FIELD = "cleanup"
REASON_FIELD = "cleanup_reason"
SECONDS_FIELD = "cleanup_seconds"


@dataclass
class DictationTrace:
    # The text the output chain was given after every stage, ``None`` until then.
    typed: str | None = None
    # Empty while the writing model was not asked: cleanup off or nothing said.
    cleanup: str = ""
    cleanup_reason: str = ""
    # The model's own words, before any guard or stage after it touched them.
    cleanup_answer: str | None = None
    cleanup_seconds: float | None = None
    # What the model was shown about the surroundings, when the settings allowed it.
    context_before: str = ""
    window: str = ""


_local = threading.local()


def begin_trace() -> DictationTrace:
    trace = DictationTrace()
    _local.trace = trace
    return trace


def current_trace() -> DictationTrace | None:
    return getattr(_local, "trace", None)


def end_trace() -> DictationTrace | None:
    trace = current_trace()
    _local.trace = None
    return trace
