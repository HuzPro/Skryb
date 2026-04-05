import time

import pyautogui
import pyperclip

from voicetotext.output.handler import TextOutputHandler


class AutoPasteAdapter(TextOutputHandler):

    def output(self, text: str) -> None:
        pyperclip.copy(text)
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")
