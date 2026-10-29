import logging
import threading
from collections.abc import Callable, Iterable

from skryb.output.caret_context import (
    CONTEXT_CHARS,
    NO_CONTEXT,
    SELECTION_OUTSIDE_FIELD,
    CaretContext,
)

_logger = logging.getLogger(__name__)

_VK_BACK = 0x08
_VK_CAPITAL = 0x14
_SHIFT_KEYS = frozenset({0x10, 0xA0, 0xA1})
_CTRL_ALT_KEYS = frozenset({0x11, 0x12, 0xA2, 0xA3, 0xA4, 0xA5})
_MODIFIER_KEYS = frozenset({
    *_SHIFT_KEYS, *_CTRL_ALT_KEYS,
    0x5B, 0x5C,  # left and right Windows
    _VK_CAPITAL,
})
_LLKHF_INJECTED = 0x10

KeyCharacter = Callable[[int, bool], str]
"""The character a key types (vk, shift held), or "" when it types none."""


def _types_nothing(vk: int, shift: bool) -> str:
    return ""


class PasteMemory:
    """The text just before the caret, as far as this app saw it go in.

    The stand-in for reading the text box when the box cannot be read: what
    this app pasted, followed along as the user types characters and
    deletes them with Backspace, or what the user typed since the caret was
    last put somewhere. It only vouches for the caret while nothing could
    have moved it: any other key (Enter, an arrow, a shortcut), any mouse
    click and any change of window forget it, so it can never invent
    context for a caret the user put somewhere else.
    """

    def __init__(
        self,
        foreground: Callable[[], int],
        click_count: Callable[[], int],
        own_input_tag: int,
        key_character: KeyCharacter = _types_nothing,
    ) -> None:
        self._foreground = foreground
        self._click_count = click_count
        self._own_input_tag = own_input_tag
        self._key_character = key_character
        self._lock = threading.Lock()
        self._text = ""
        self._hwnd = 0
        self._clicks = 0
        self._held: set[int] = set()
        self._hotkeys: tuple[frozenset[int], ...] = ()

    def set_hotkeys(self, key_sets: Iterable[Iterable[int]]) -> None:
        """The shortcuts' keys, whose presses are not typing."""
        with self._lock:
            self._hotkeys = tuple(frozenset(vks) for vks in key_sets)

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
            self._follow_typing(vk, flags)

    def _follow_typing(self, vk: int, flags: int) -> None:
        """Track a character typed or deleted at the caret; forget on anything else.

        Keys another program injects and shortcuts with Ctrl or Alt (paste,
        undo, delete a word) can change the text in ways the memory cannot
        follow.
        """
        if flags & _LLKHF_INJECTED or self._held & _CTRL_ALT_KEYS:
            self._text = ""
            return
        if vk == _VK_BACK:
            self._text = self._text[:-1]
            return
        typed = self._key_character(vk, bool(self._held & _SHIFT_KEYS))
        if not typed:
            self._text = ""
            return
        hwnd = self._foreground()
        if not self._still_in_place(hwnd):
            # Typing starts a fresh memory: the caret is right after what is typed.
            self._text, self._hwnd, self._clicks = "", hwnd, self._click_count()
        self._text = (self._text + typed)[-CONTEXT_CHARS:]

    def _still_in_place(self, hwnd: int) -> bool:
        return (
            bool(self._text)
            and hwnd != 0
            and hwnd == self._hwnd
            and self._click_count() == self._clicks
        )

    def _is_hotkey_press(self, vk: int) -> bool:
        return any(
            vk in hotkey and hotkey - {vk} <= self._held for hotkey in self._hotkeys
        )


def read_or_remember(
    read: Callable[[], CaretContext],
    memory: PasteMemory,
    wait: Callable[[], None],
) -> Callable[[], CaretContext]:
    """Read the text box, try once more, then fall back to the last paste.

    A selection outside the field gets no second try: in 328 dictations it
    never read differently 100 ms later.
    """

    def context() -> CaretContext:
        found = read()
        if found == NO_CONTEXT and found is not SELECTION_OUTSIDE_FIELD:
            wait()
            found = read()
        if found == NO_CONTEXT:
            found = memory.context()
            if found != NO_CONTEXT:
                _logger.debug("Caret context unreadable; using the last paste.")
        return found

    return context
