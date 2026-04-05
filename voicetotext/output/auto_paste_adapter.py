import time

import pyautogui
import pyperclip

from voicetotext.output.handler import TextOutputHandler


class AutoPasteAdapter(TextOutputHandler):

    def output(self, text: str) -> None:
        pyperclip.copy(text)
        time.sleep(0.1)
        try:
            pyautogui.hotkey("ctrl", "v")
        except Exception as e:
            print(f"  WARNING: Auto-paste failed ({e}), text is in clipboard.")
