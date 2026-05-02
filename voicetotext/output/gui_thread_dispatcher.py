from abc import ABC, abstractmethod
from collections.abc import Callable


class GuiThreadDispatcher(ABC):
    """Port for marshaling a callable onto the GUI thread.

    Win32 clipboard and ``SendInput``-style calls misbehave from
    non-UI threads — most visibly, in our case, by silently crashing
    the process after :class:`NanoParakeetAdapter`'s GPU work has
    perturbed the worker thread's HIP runtime state. The use case
    runs on a daemon worker thread (via :class:`ThreadedActionDispatcher`)
    so any output handler that touches Win32 needs an explicit hop
    back to the GUI thread. This port is the seam.
    """

    @abstractmethod
    def dispatch(self, action: Callable[[], None]) -> None:
        """Schedule ``action`` to run on the GUI thread.

        Implementations may run synchronously (test doubles) or
        asynchronously (Qt impl using a queued signal). Callers must
        not assume ``action`` has run by the time ``dispatch`` returns.
        """
