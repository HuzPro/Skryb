"""Paste only where text can go."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.last_dictation import LastDictation
from voicetotext.output.paste_method import PasteMethod

_logger = logging.getLogger(__name__)


class DeliveryFailure(Enum):
    NO_TEXT_FIELD = "no_text_field"


OnDeliveryFailed = Callable[[DeliveryFailure, str], None]


@dataclass(frozen=True)
class DeliveryHooks:
    """What the composition root lends the output chain for app-aware delivery."""

    last_dictation: LastDictation
    paste_method: Callable[[], PasteMethod]
    on_failed: OnDeliveryFailed
    on_pasted: Callable[[], None] = lambda: None


class AppAwareDelivery(TextOutputHandler):
    """Pastes only into a text field.

    Runs on the transcription worker, before the hop to the GUI thread:
    asking UI Automation about the focused control can take longer than
    the GUI thread may block. Without a text field the text goes to the
    clipboard and nothing is typed, so a menu, the desktop or a game never
    receives a stray paste.

    A paste is not read back afterwards. Some chat boxes read as empty to
    UI Automation whatever they hold, so a check there reported pastes
    that had landed as lost.
    """

    def __init__(
        self,
        paste: TextOutputHandler,
        copy: TextOutputHandler,
        gui: GuiThreadDispatcher,
        on_failed: OnDeliveryFailed,
        *,
        text_field_focused: Callable[[], bool],
    ) -> None:
        self._paste = paste
        self._copy = copy
        self._gui = gui
        self._on_failed = on_failed
        self._text_field_focused = text_field_focused

    def output(self, text: str) -> None:
        if not self._text_field_focused():
            _logger.info("No text field has the focus; the text went to the clipboard.")
            self._gui.dispatch(lambda: self._copy.output(text))
            self._on_failed(DeliveryFailure.NO_TEXT_FIELD, text)
            return
        self._gui.dispatch(lambda: self._paste.output(text))
