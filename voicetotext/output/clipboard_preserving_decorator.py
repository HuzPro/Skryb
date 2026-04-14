import logging
import time

from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.handler import TextOutputHandler


_logger = logging.getLogger(__name__)

# After the wrapped adapter simulates Ctrl+V, the target application reads
# the clipboard asynchronously via its message loop.  We must wait before
# restoring the original clipboard contents.  This is a best-effort
# heuristic — there is no Windows API to detect when a target app has
# finished processing a paste.
_PASTE_CONSUME_DELAY_SECONDS = 0.05


class ClipboardPreservingDecorator(TextOutputHandler):

    def __init__(
        self,
        inner: TextOutputHandler,
        preserver: ClipboardPreserver,
    ) -> None:
        self._inner = inner
        self._preserver = preserver

    def output(self, text: str) -> None:
        if not self._try_save():
            _logger.error(
                "Clipboard snapshot failed; transcription not pasted (%d chars). "
                "Consider switching to 'Paste and copy to clipboard' mode.",
                len(text),
            )
            return
        try:
            self._inner.output(text)
            time.sleep(_PASTE_CONSUME_DELAY_SECONDS)
        finally:
            self._try_restore()

    def _try_save(self) -> bool:
        try:
            self._preserver.save()
            return True
        except Exception as exc:
            _logger.warning("Clipboard save failed (%s).", exc)
            return False

    def _try_restore(self) -> None:
        try:
            self._preserver.restore()
        except Exception as exc:
            _logger.error(
                "Clipboard restore failed (%s); original clipboard contents lost.",
                exc,
            )
