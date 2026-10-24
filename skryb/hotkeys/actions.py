import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from skryb.hotkeys.bindings import Binding, BindingAction, BindingMode
from skryb.hotkeys.hold_strategy import HoldStrategy
from skryb.hotkeys.strategy import HotkeyStrategy
from skryb.hotkeys.toggle_strategy import ToggleStrategy

_logger = logging.getLogger(__name__)

Action = Callable[[], None]


@dataclass(frozen=True)
class HotkeyActions:
    """What each kind of shortcut does. A shortcut whose action is ``None``
    here is left inert, so its keys and buttons pass through untouched.

    ``*_activate`` is a toggle request, like
    ``RecordAndTranscribeUseCase.on_hotkey_activated``: it starts when idle
    and stops what is running. Toggle shortcuts send it on every press, and
    a hold shortcut sends it when pressed and on the press that ends a
    double-tap lock; ``*_deactivate`` is sent when a hold is let go.
    """

    dictate_activate: Action | None = None
    dictate_deactivate: Action | None = None
    edit_activate: Action | None = None
    edit_deactivate: Action | None = None
    paste_last: Action | None = None
    copy_last: Action | None = None

    def for_action(self, action: BindingAction) -> tuple[Action | None, Action | None]:
        return {
            BindingAction.DICTATE: (self.dictate_activate, self.dictate_deactivate),
            BindingAction.EDIT_SELECTION: (self.edit_activate, self.edit_deactivate),
            BindingAction.PASTE_LAST: (self.paste_last, None),
            BindingAction.COPY_LAST: (self.copy_last, None),
        }[action]


def _nothing() -> None:
    return None


def strategies_for(
    bindings: Sequence[Binding],
    actions: HotkeyActions,
    double_tap_locks_hold: bool,
) -> list[HotkeyStrategy]:
    strategies: list[HotkeyStrategy] = []
    for binding in bindings:
        activate, deactivate = actions.for_action(binding.action)
        holds = binding.mode is BindingMode.HOLD
        if activate is None or (holds and deactivate is None):
            _logger.debug(
                "Shortcut %s (%s) has nothing to do yet; ignored.",
                binding.combo, binding.action.value,
            )
            continue
        strategy: HotkeyStrategy = (
            HoldStrategy(binding.keys, double_tap_locks_hold)
            if holds
            else ToggleStrategy(binding.keys)
        )
        strategy.set_callbacks(activate, deactivate or _nothing)
        strategies.append(strategy)
    return strategies
