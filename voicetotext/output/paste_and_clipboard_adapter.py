import logging

import pyautogui

from voicetotext.output.clipboard_writer import ClipboardWriter
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.sleeper import Sleeper


_logger = logging.getLogger(__name__)
_CLIPBOARD_SETTLE_DELAY_SECONDS = 0.1


class PasteAndClipboardAdapter(TextOutputHandler):

    def __init__(self, sleeper: Sleeper, writer: ClipboardWriter) -> None:
        self._sleeper = sleeper
        self._writer = writer

    def output(self, text: str) -> None:
        self._writer.set_text(text)
        self._sleeper.sleep(_CLIPBOARD_SETTLE_DELAY_SECONDS)
        try:
            pyautogui.hotkey("ctrl", "v")
        except Exception as e:
            _logger.warning("Auto-paste failed (%s), text is in clipboard.", e)
