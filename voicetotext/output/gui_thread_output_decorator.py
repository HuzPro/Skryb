from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.output.handler import TextOutputHandler


class GuiThreadOutputDecorator(TextOutputHandler):
    """Decorator (GoF) marshaling :meth:`output` onto the GUI thread.

    The use case runs on a daemon worker thread, but Win32 clipboard
    and keyboard simulation calls require a UI thread (one that owns
    a window with a message pump). Wrapping any inner
    :class:`TextOutputHandler` with this decorator keeps the worker
    thread responsible only for inference and hands the actual paste
    to the GUI thread via the injected :class:`GuiThreadDispatcher`.

    The use case is unchanged - it still talks to a
    :class:`TextOutputHandler` and is none the wiser about threading.
    """

    def __init__(
        self,
        inner: TextOutputHandler,
        dispatcher: GuiThreadDispatcher,
    ) -> None:
        self._inner = inner
        self._dispatcher = dispatcher

    def output(self, text: str) -> None:
        self._dispatcher.dispatch(lambda: self._inner.output(text))
