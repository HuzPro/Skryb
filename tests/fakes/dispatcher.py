from collections.abc import Callable

from voicetotext.core.action_dispatcher import ActionDispatcher


class InlineActionDispatcher(ActionDispatcher):
    """Runs every dispatched action immediately on the calling thread.

    Lets orchestration tests exercise the full code path synchronously -
    no sleeps, no timeouts, no _wait_for_* plumbing.  Also records every
    dispatched action for inspection.
    """

    def __init__(self) -> None:
        self.dispatched: list[Callable[[], None]] = []

    def dispatch(self, action: Callable[[], None]) -> None:
        self.dispatched.append(action)
        action()


class DeferredActionDispatcher(ActionDispatcher):
    """Queues dispatched actions without running them until drained.

    Useful for tests that need to observe intermediate state between
    the moment an action is scheduled and the moment it runs - e.g. to
    check that state has been set to ``TRANSCRIBING`` before the actual
    transcription work executes.
    """

    def __init__(self) -> None:
        self.pending: list[Callable[[], None]] = []

    def dispatch(self, action: Callable[[], None]) -> None:
        self.pending.append(action)

    def drain_next(self) -> None:
        if not self.pending:
            raise RuntimeError("drain_next called with no pending actions")
        self.pending.pop(0)()

    def drain_all(self) -> None:
        while self.pending:
            self.drain_next()
