import logging

import pyautogui
import pyperclip

from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.sleeper import Sleeper


_logger = logging.getLogger(__name__)
_CLIPBOARD_SETTLE_DELAY_SECONDS = 0.1


class PasteAndClipboardAdapter(TextOutputHandler):

    def __init__(self, sleeper: Sleeper) -> None:
        self._sleeper = sleeper

    def output(self, text: str) -> None:
        pyperclip.copy(text)
        self._sleeper.sleep(_CLIPBOARD_SETTLE_DELAY_SECONDS)
        try:
            pyautogui.hotkey("ctrl", "v")
        except Exception as e:
            _logger.warning("Auto-paste failed (%s), text is in clipboard.", e)
