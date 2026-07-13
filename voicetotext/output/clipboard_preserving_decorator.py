import logging

from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.sleeper import Sleeper


_logger = logging.getLogger(__name__)

# After the wrapped adapter simulates Ctrl+V, the target application reads
# the clipboard asynchronously via its message loop.  We must wait before
# restoring the original clipboard contents.  This is a best-effort
# heuristic — there is no Windows API to detect when a target app has
# finished processing a paste.  50ms was observed losing the race against
# a busy target app, which then pasted the restored OLD clipboard instead
# of the transcription.  The delay only defers when the user's previous
# clipboard returns; it does not delay the paste itself.
_PASTE_CONSUME_DELAY_SECONDS = 0.25


class ClipboardPreservingDecorator(TextOutputHandler):

    def __init__(
        self,
        inner: TextOutputHandler,
        preserver: ClipboardPreserver,
        sleeper: Sleeper,
    ) -> None:
        self._inner = inner
        self._preserver = preserver
        self._sleeper = sleeper

    def output(self, text: str) -> None:
        _logger.debug("clipboard_preserving: save snapshot")
        if not self._try_save():
            _logger.error(
                "Clipboard snapshot failed; transcription not pasted (%d chars). "
                "Consider switching to 'Paste and copy to clipboard' mode.",
                len(text),
            )
            return
        try:
            _logger.debug("clipboard_preserving: inner.output")
            self._inner.output(text)
            _logger.debug("clipboard_preserving: consume sleep")
            self._sleeper.sleep(_PASTE_CONSUME_DELAY_SECONDS)
        finally:
            _logger.debug("clipboard_preserving: restore snapshot")
            self._try_restore()
            _logger.debug("clipboard_preserving: done")

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
