import logging
import time

import pyautogui
import pyperclip

from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.handler import TextOutputHandler


_logger = logging.getLogger(__name__)

# Clipboard writes on Windows are asynchronous at the driver level.
# Without a settle delay, the subsequent Ctrl+V can read stale data.
# 100ms matches the existing PasteAndClipboardAdapter and has proven
# reliable across tested machines.
_CLIPBOARD_WRITE_SETTLE_SECONDS = 0.1

# After simulating Ctrl+V, the target application reads the clipboard
# asynchronously via its message loop.  We must wait before restoring
# the original clipboard contents.  This is a best-effort heuristic —
# there is no Windows API to detect when a target app has finished
# processing a paste.  See the plan for the full tradeoff analysis.
_PASTE_CONSUME_DELAY_SECONDS = 0.05


class DirectPasteAdapter(TextOutputHandler):

    def __init__(self, preserver: ClipboardPreserver) -> None:
        self._preserver = preserver

    def output(self, text: str) -> None:
        save_succeeded = self._try_save()
        try:
            pyperclip.copy(text)
            time.sleep(_CLIPBOARD_WRITE_SETTLE_SECONDS)
            try:
                pyautogui.hotkey("ctrl", "v")
            except Exception as exc:
                _logger.warning("Paste keystroke failed (%s); text is in clipboard.", exc)
            time.sleep(_PASTE_CONSUME_DELAY_SECONDS)
        finally:
            if save_succeeded:
                self._try_restore()

    def _try_save(self) -> bool:
        try:
            self._preserver.save()
            return True
        except Exception as exc:
            _logger.warning("Clipboard save failed (%s); pasting without preservation.", exc)
            return False

    def _try_restore(self) -> None:
        try:
            self._preserver.restore()
        except Exception as exc:
            _logger.error(
                "Clipboard restore failed (%s); original clipboard contents lost.", exc
            )
