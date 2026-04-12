import logging
import time

import pyautogui
import pyperclip

from voicetotext.output.handler import TextOutputHandler


_logger = logging.getLogger(__name__)
_CLIPBOARD_SETTLE_DELAY_SECONDS = 0.1


class AutoPasteAdapter(TextOutputHandler):

    def output(self, text: str) -> None:
        pyperclip.copy(text)
        time.sleep(_CLIPBOARD_SETTLE_DELAY_SECONDS)
        try:
            pyautogui.hotkey("ctrl", "v")
        except Exception as e:
            _logger.warning("Auto-paste failed (%s), text is in clipboard.", e)
