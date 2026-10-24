import time
from abc import abstractmethod
from collections.abc import Callable

from skryb.output.sleeper import Sleeper


class GuiThreadDispatcher(Sleeper):
    """Port for marshaling a callable onto the GUI thread.

    Win32 clipboard and ``SendInput``-style calls misbehave from
    non-UI threads - most visibly, in our case, by silently crashing
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

    def sleep(self, seconds: float) -> None:
        """Wait ``seconds`` inside an action running on the GUI thread.

        A plain sleep freezes every window the GUI thread owns, so a paste
        aimed at one of our own windows would only be read after the action
        returns. Implementations with an event loop keep it running here.
        """
        time.sleep(seconds)
