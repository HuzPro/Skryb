"""What happened to one dictation between the engine and the paste.

History keeps it next to the transcription, so what was typed can be told
apart from what the engine heard. The trace lives on the thread running the
dictation: the text pipeline runs there from start to finish, and a history
retry on another thread never mixes its text in.
"""

import threading
from dataclasses import dataclass

# The session log's name for the typed text, which a history deletion blanks.
TYPED_FIELD = "typed_text"


@dataclass
class DictationTrace:
    # The text the output chain was given after every stage, ``None`` until then.
    typed: str | None = None


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
