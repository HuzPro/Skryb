from typing import Protocol, runtime_checkable


@runtime_checkable
class ModifierKeyStateReader(Protocol):
    """Reports whether any physical modifier key is currently held.

    Kept behind a port so the paste path stays free of OS keyboard
    APIs and tests can script the user's fingers.
    """

    def any_modifier_down(self) -> bool: ...
