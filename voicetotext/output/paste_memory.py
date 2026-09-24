import logging
import threading
from collections.abc import Callable, Iterable

from voicetotext.output.caret_context import CONTEXT_CHARS, NO_CONTEXT, CaretContext

_logger = logging.getLogger(__name__)

_VK_CAPITAL = 0x14
_MODIFIER_KEYS = frozenset({
    0x10, 0x11, 0x12,  # generic Shift, Ctrl, Alt
    0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5,  # left and right Shift, Ctrl, Alt
    0x5B, 0x5C,  # left and right Windows
    _VK_CAPITAL,
})


class PasteMemory:
    """What this app last pasted, for as long as it is still next to the caret.

    The stand-in for reading the text box when the box cannot be read. It
    only vouches for the caret while nothing could have moved it: any key
    the user types, any mouse click and any change of window forget it, so
    it can never invent context for a caret the user put somewhere else.
    """

    def __init__(
        self,
        foreground: Callable[[], int],
        click_count: Callable[[], int],
        own_input_tag: int,
    ) -> None:
        self._foreground = foreground
        self._click_count = click_count
        self._own_input_tag = own_input_tag
        self._lock = threading.Lock()
        self._text = ""
        self._hwnd = 0
        self._clicks = 0
        self._held: set[int] = set()
        self._hotkey: frozenset[int] = frozenset()

    def set_hotkey(self, vks: Iterable[int]) -> None:
        with self._lock:
            self._hotkey = frozenset(vks)

    def remember(self, text: str) -> None:
        """Record a paste that is about to land in the foreground window."""
        hwnd = self._foreground()
        with self._lock:
            follows_last_paste = self._still_in_place(hwnd)
            self._text = ((self._text if follows_last_paste else "") + text)[
                -CONTEXT_CHARS:
            ]
            self._hwnd = hwnd
            self._clicks = self._click_count()

    def context(self) -> CaretContext:
        hwnd = self._foreground()
        with self._lock:
            if not self._still_in_place(hwnd):
                self._text = ""
                return NO_CONTEXT
            return CaretContext(self._text, "", "")

    def sync_clicks(self) -> None:
        """Forget the paste if the mouse was clicked anywhere since last asked.

        Polled every frame while the recording pill is up, so a click
        elsewhere is caught before a click on the pill could be excused
        along with it.
        """
        with self._lock:
            if self._click_count() != self._clicks:
                self._text = ""

    def excuse_click(self) -> None:
        """Let a click on the pill itself leave the paste in place.

        The pill never takes the focus, so clicking it cannot move the caret.
        """
        with self._lock:
            if self._text:
                self._clicks = self._click_count()

    def observe_key(self, is_key_down: bool, vk: int, flags: int, extra: int) -> None:
        if extra == self._own_input_tag:
            return
        with self._lock:
            if not is_key_down:
                self._held.discard(vk)
                return
            if vk in _MODIFIER_KEYS:
                self._held.add(vk)
                return
            if self._is_hotkey_press(vk):
                return
            self._text = ""

    def _still_in_place(self, hwnd: int) -> bool:
        return (
            bool(self._text)
            and hwnd != 0
            and hwnd == self._hwnd
            and self._click_count() == self._clicks
        )

    def _is_hotkey_press(self, vk: int) -> bool:
        return vk in self._hotkey and self._hotkey - {vk} <= self._held


def read_or_remember(
    read: Callable[[], CaretContext],
    memory: PasteMemory,
    wait: Callable[[], None],
) -> Callable[[], CaretContext]:
    """Read the text box, try once more, then fall back to the last paste."""

    def context() -> CaretContext:
        found = read()
        if found == NO_CONTEXT:
            wait()
            found = read()
        if found == NO_CONTEXT:
            found = memory.context()
            if found != NO_CONTEXT:
                _logger.debug("Caret context unreadable; using the last paste.")
        return found

    return context
