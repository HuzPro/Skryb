import logging
from collections.abc import Callable

from voicetotext.output.clipboard_writer import ClipboardWriter
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.key_sender import KeySender
from voicetotext.output.paste_method import PasteMethod
from voicetotext.output.sleeper import Sleeper


_logger = logging.getLogger(__name__)
_CLIPBOARD_SETTLE_DELAY_SECONDS = 0.1


def _always_ctrl_v() -> PasteMethod:
    return PasteMethod.CTRL_V


class PasteAndClipboardAdapter(TextOutputHandler):

    def __init__(
        self,
        sleeper: Sleeper,
        writer: ClipboardWriter,
        key_sender: KeySender,
        paste_method: Callable[[], PasteMethod] | None = None,
    ) -> None:
        self._sleeper = sleeper
        self._writer = writer
        self._key_sender = key_sender
        self._paste_method = paste_method or _always_ctrl_v

    def output(self, text: str) -> None:
        _logger.debug("paste_and_clipboard: writer.set_text")
        self._writer.set_text(text)
        _logger.debug("paste_and_clipboard: settle sleep")
        self._sleeper.sleep(_CLIPBOARD_SETTLE_DELAY_SECONDS)
        self._paste(text)
        _logger.debug("paste_and_clipboard: done")

    def _paste(self, text: str) -> None:
        try:
            self._send_paste(text)
        except Exception as e:
            _logger.warning("Auto-paste failed (%s), text is in clipboard.", e)

    def _send_paste(self, text: str) -> None:
        method = self._paste_method()
        _logger.debug("paste_and_clipboard: send %s", method.value)
        self._key_sender.send_paste(method, text)
