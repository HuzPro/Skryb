"""Paste only where text can go, and notice when a paste does not arrive."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from voicetotext.output.caret_context import NO_CONTEXT, CaretContext
from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.last_dictation import LastDictation
from voicetotext.output.paste_method import PasteMethod

_logger = logging.getLogger(__name__)

# The tail of the pasted text that must show up before the caret.
_CONFIRM_CHARS = 20
# A busy app can take a moment to show a paste, so a miss is checked once
# more before it is reported; a false alarm invites a double paste.
_CONFIRM_DELAYS_SECONDS = (0.3, 0.7)
# The GUI thread holds the paste for the modifier release (up to 1 s) and
# the clipboard restore; past this something else is wrong.
_PASTE_TIMEOUT_SECONDS = 5.0


class DeliveryFailure(Enum):
    NO_TEXT_FIELD = "no_text_field"
    NOT_LANDED = "not_landed"


OnDeliveryFailed = Callable[[DeliveryFailure, str], None]


@dataclass(frozen=True)
class DeliveryHooks:
    """What the composition root lends the output chain for app-aware delivery."""

    last_dictation: LastDictation
    paste_method: Callable[[], PasteMethod]
    on_failed: OnDeliveryFailed


def paste_landed(text: str, context: CaretContext) -> bool:
    """Whether the text before the caret ends the way the paste did.

    Only letters and digits are compared, case-folded, so an app that
    reflows whitespace, curls quotes or upper-cases for Caps Lock still
    counts as a paste that landed.
    """
    expected = _comparable(text.rstrip()[-_CONFIRM_CHARS:])
    return _comparable(context.before + context.selected).endswith(expected)


def _comparable(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def _start_daemon_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="paste-confirm", daemon=True).start()


class AppAwareDelivery(TextOutputHandler):
    """Pastes only into a text field, then checks the text arrived.

    Runs on the transcription worker, before the hop to the GUI thread:
    asking UI Automation about the focused control can take longer than
    the GUI thread may block. Without a text field the text goes to the
    clipboard and nothing is typed, so a menu, the desktop or a game never
    receives a stray paste. The check after a paste runs on its own thread
    so the app is ready for the next dictation meanwhile, and it never
    pastes again by itself.
    """

    def __init__(
        self,
        paste: TextOutputHandler,
        copy: TextOutputHandler,
        gui: GuiThreadDispatcher,
        on_failed: OnDeliveryFailed,
        *,
        text_field_focused: Callable[[], bool],
        read_context: Callable[[], CaretContext],
        foreground: Callable[[], int],
        start_thread: Callable[[Callable[[], None]], None] = _start_daemon_thread,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._paste = paste
        self._copy = copy
        self._gui = gui
        self._on_failed = on_failed
        self._text_field_focused = text_field_focused
        self._read_context = read_context
        self._foreground = foreground
        self._start_thread = start_thread
        self._sleep = sleep

    def output(self, text: str) -> None:
        if not self._text_field_focused():
            _logger.info("No text field has the focus; the text went to the clipboard.")
            self._gui.dispatch(lambda: self._copy.output(text))
            self._on_failed(DeliveryFailure.NO_TEXT_FIELD, text)
            return
        pasted = threading.Event()
        target = self._foreground()
        self._gui.dispatch(lambda: self._paste_then_signal(text, pasted))
        self._start_thread(lambda: self._confirm(text, pasted, target))

    def _paste_then_signal(self, text: str, pasted: threading.Event) -> None:
        try:
            self._paste.output(text)
        finally:
            pasted.set()

    def _confirm(self, text: str, pasted: threading.Event, target: int) -> None:
        if not pasted.wait(_PASTE_TIMEOUT_SECONDS):
            return
        for delay in _CONFIRM_DELAYS_SECONDS:
            self._sleep(delay)
            if self._foreground() != target:
                return  # the user moved on; the field read now is another one
            context = self._read_context()
            if context == NO_CONTEXT or paste_landed(text, context):
                return
        _logger.warning("The paste did not show up in the text field.")
        self._on_failed(DeliveryFailure.NOT_LANDED, text)
