import threading

from voicetotext.output.handler import TextOutputHandler


class LastDictation:
    """The latest dictation as it left the text pipeline, ready to send again.

    Thread-safe: dictations are recorded on the transcription worker while
    paste-last and copy-last arrive from the hotkey and the tray. The output
    chain in use registers where a repeat goes through :meth:`route`, so a
    repeat follows the output mode as it is configured now.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._text = ""
        self._paste: TextOutputHandler | None = None
        self._copy: TextOutputHandler | None = None

    def route(self, paste: TextOutputHandler, copy: TextOutputHandler) -> None:
        """Send repeats through *paste* (the chain minus the pipeline) and *copy*."""
        with self._lock:
            self._paste = paste
            self._copy = copy

    def remember(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            self._text = text

    def text(self) -> str:
        with self._lock:
            return self._text

    def paste(self) -> bool:
        """Deliver the last dictation again; ``False`` when there is none."""
        with self._lock:
            text, output = self._text, self._paste
        return self._send(text, output)

    def copy(self) -> bool:
        """Put the last dictation on the clipboard; ``False`` when there is none.

        Trimmed: the spaces that fitted it to the caret belong to that paste.
        """
        with self._lock:
            text, output = self._text, self._copy
        return self._send(text.strip(), output)

    @staticmethod
    def _send(text: str, output: TextOutputHandler | None) -> bool:
        if not text or output is None:
            return False
        output.output(text)
        return True


class LastDictationDecorator(TextOutputHandler):
    """Records each dictation's final text on its way to delivery."""

    def __init__(self, inner: TextOutputHandler, last: LastDictation) -> None:
        self._inner = inner
        self._last = last

    def output(self, text: str) -> None:
        self._last.remember(text)
        self._inner.output(text)
