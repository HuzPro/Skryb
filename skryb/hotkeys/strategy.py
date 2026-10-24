from abc import ABC, abstractmethod
from collections.abc import Callable

from skryb.hotkeys.keys import HotkeyKey


class HotkeyStrategy(ABC):

    @abstractmethod
    def on_key_press(self, key: HotkeyKey) -> bool:
        """Handle a key going down; True when it completed the combination."""

    @abstractmethod
    def on_key_release(self, key: HotkeyKey) -> None:
        pass

    @abstractmethod
    def set_callbacks(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        pass


class ComboMatcher:
    """Follows one combination pressed as an exact ordered sequence.

    The matched prefix is exactly the set of combination keys still held,
    so a repeated key-down of one of them is autorepeat and changes
    nothing. Releasing a held key drops the match back to just before it,
    which lets the last key be tapped again while the others stay down.
    Any other key pressed mid-sequence starts the match over; once the
    whole combination is down, other keys are ignored until it breaks.
    """

    def __init__(self, keys: tuple[HotkeyKey, ...]) -> None:
        self._keys = tuple(keys)
        self._matched = 0

    def press(self, key: HotkeyKey) -> bool:
        """True when *key* completes the combination."""
        if self.is_complete() or key in self._keys[:self._matched]:
            return False
        if key == self._keys[self._matched]:
            self._matched += 1
            return self.is_complete()
        self._matched = 0
        return False

    def release(self, key: HotkeyKey) -> bool:
        """True when releasing *key* breaks a fully held combination."""
        if key not in self._keys[:self._matched]:
            return False
        was_complete = self.is_complete()
        self._matched = self._keys.index(key)
        return was_complete

    def is_complete(self) -> bool:
        return self._matched == len(self._keys)
