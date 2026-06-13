import logging

from voicetotext.output.clipboard_writer import ClipboardWriter
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.key_sender import KeySender
from voicetotext.output.sleeper import Sleeper


_logger = logging.getLogger(__name__)
_CLIPBOARD_SETTLE_DELAY_SECONDS = 0.1


class PasteAndClipboardAdapter(TextOutputHandler):

    def __init__(
        self, sleeper: Sleeper, writer: ClipboardWriter, key_sender: KeySender,
    ) -> None:
        self._sleeper = sleeper
        self._writer = writer
        self._key_sender = key_sender

    def output(self, text: str) -> None:
        _logger.debug("paste_and_clipboard: writer.set_text")
        self._writer.set_text(text)
        _logger.debug("paste_and_clipboard: settle sleep")
        self._sleeper.sleep(_CLIPBOARD_SETTLE_DELAY_SECONDS)
        self._paste()
        _logger.debug("paste_and_clipboard: done")

    def _paste(self) -> None:
        _logger.debug("paste_and_clipboard: send Ctrl+V")
        try:
            self._key_sender.send_ctrl_v()
        except Exception as e:
            _logger.warning("Auto-paste failed (%s), text is in clipboard.", e)
