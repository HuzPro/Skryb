import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from skryb.core.config import ConfigurationStore
from skryb.core.exceptions import HotkeyError
from skryb.core.json_config_adapter import DEFAULT_CONFIG
from skryb.hotkeys.key_parser import parse_hotkey_combo
from skryb.hotkeys.keys import MODIFIER_VKS, MOUSE_BUTTON_VKS, HotkeyKey
from skryb.hotkeys.strategy import ComboMatcher

_logger = logging.getLogger(__name__)


class BindingMode(StrEnum):
    TOGGLE = "toggle"
    HOLD = "hold"
    PRESS = "press"


class BindingAction(StrEnum):
    DICTATE = "dictate"
    PASTE_LAST = "paste_last"
    COPY_LAST = "copy_last"


_PRESS_ACTIONS = frozenset({BindingAction.PASTE_LAST, BindingAction.COPY_LAST})
_ACTION_NAMES = frozenset(action.value for action in BindingAction)
_LEGACY_KEYS = ("combination", "mode")
_LEGACY_DEFAULT_COMBO = "ctrl+alt+m"


class BindingError(HotkeyError):
    """Shortcuts that cannot be saved together; the message says why."""


@dataclass(frozen=True)
class Binding:
    """One shortcut: the keys as typed in config, how it fires, what it does."""

    combo: str
    mode: BindingMode
    action: BindingAction

    @property
    def keys(self) -> tuple[HotkeyKey, ...]:
        return parse_hotkey_combo(self.combo)

    def uses_mouse(self) -> bool:
        return any(key.vk in MOUSE_BUTTON_VKS for key in self.keys)


def binding_for(combo: str, mode: str, action: BindingAction) -> Binding:
    """A binding whose mode is one *action* can have: single-press actions
    only fire on press, and everything else is a toggle unless it is a hold.
    """
    if action in _PRESS_ACTIONS:
        return Binding(combo, BindingMode.PRESS, action)
    held = mode == BindingMode.HOLD
    return Binding(combo, BindingMode.HOLD if held else BindingMode.TOGGLE, action)


def parse_bindings(config: ConfigurationStore) -> tuple[Binding, ...]:
    """The shortcuts saved in config, skipping any entry that cannot work."""
    rows: object = config.get("hotkey.bindings")
    return _bindings_from_rows(_default_rows() if rows is None else rows)


def _bindings_from_rows(rows: object) -> tuple[Binding, ...]:
    if not isinstance(rows, list):
        _logger.warning("hotkey.bindings is not a list; no shortcuts are active.")
        return ()
    bindings: list[Binding] = []
    for row in rows:
        binding = _binding_from_row(row)
        if binding is not None:
            bindings.append(binding)
    return tuple(bindings)


def _binding_from_row(row: object) -> Binding | None:
    if not isinstance(row, dict):
        _logger.warning("Ignoring shortcut %r: not an object.", row)
        return None
    combo, mode, action = row.get("keys"), row.get("mode"), row.get("action")
    if not isinstance(combo, str) or not combo.strip():
        return None
    if not isinstance(action, str) or action not in _ACTION_NAMES:
        _logger.warning("Ignoring shortcut %r: unknown action %r.", combo, action)
        return None
    try:
        parse_hotkey_combo(combo)
    except ValueError as exc:
        _logger.warning("Ignoring shortcut %r: %s.", combo, exc)
        return None
    return binding_for(combo, str(mode), BindingAction(action))


def bindings_to_config(bindings: Sequence[Binding]) -> list[object]:
    return [
        {"keys": binding.combo, "mode": binding.mode.value, "action": binding.action.value}
        for binding in bindings
    ]


def validate_bindings(bindings: Sequence[Binding]) -> None:
    """Raise :class:`BindingError` unless every shortcut can fire on its own."""
    if not any(binding.action is BindingAction.DICTATE for binding in bindings):
        raise BindingError("Add at least one dictation shortcut.")
    parsed = [(binding, _keys_that_work_alone(binding)) for binding in bindings]
    for (first, first_keys), (second, second_keys) in itertools.permutations(parsed, 2):
        if frozenset(first_keys) == frozenset(second_keys):
            raise BindingError(
                f"Shortcut {second.combo} is used twice. Give each shortcut its own keys.",
            )
        if _fires_while_pressing(first_keys, second_keys):
            raise BindingError(
                f"Pressing {second.combo} would also trigger {first.combo}. "
                "Change one of them.",
            )


def _keys_that_work_alone(binding: Binding) -> tuple[HotkeyKey, ...]:
    keys = _parsed_keys(binding)
    if len(keys) == 1 and keys[0].vk in MODIFIER_VKS:
        raise BindingError(
            f"Shortcut {binding.combo} needs another key: a modifier on its "
            "own fires with every shortcut that uses it.",
        )
    if len(set(keys)) != len(keys):
        raise BindingError(f"Shortcut {binding.combo} repeats a key.")
    return keys


def _parsed_keys(binding: Binding) -> tuple[HotkeyKey, ...]:
    try:
        return binding.keys
    except ValueError as exc:
        raise BindingError(f"Shortcut {binding.combo!r} is not valid: {exc}.") from exc


def _fires_while_pressing(
    keys: tuple[HotkeyKey, ...], other: tuple[HotkeyKey, ...],
) -> bool:
    matcher = ComboMatcher(keys)
    return any(matcher.press(key) for key in other)


def migrate_legacy_hotkey(config: ConfigurationStore) -> bool:
    """Fold a pre-bindings ``hotkey.combination``/``hotkey.mode`` into one
    dictation binding and drop the old keys. True when config changed.

    The old pair wins over bindings that are still the default, because
    loading an old file fills the default bindings in beside it.
    """
    data = config.get_all()
    hotkey = data.get("hotkey")
    if not isinstance(hotkey, dict) or not any(key in hotkey for key in _LEGACY_KEYS):
        return False
    if hotkey.get("bindings", _default_rows()) == _default_rows():
        mode = BindingMode.HOLD if hotkey.get("mode") == "push_to_talk" else BindingMode.TOGGLE
        combo = hotkey.get("combination") or _LEGACY_DEFAULT_COMBO
        hotkey["bindings"] = bindings_to_config(
            [Binding(str(combo), mode, BindingAction.DICTATE)],
        )
    for key in _LEGACY_KEYS:
        hotkey.pop(key, None)
    config.replace_all(data)
    return True


def _default_rows() -> object:
    hotkey = DEFAULT_CONFIG["hotkey"]
    return hotkey["bindings"] if isinstance(hotkey, dict) else None


DEFAULT_BINDINGS = _bindings_from_rows(_default_rows())
