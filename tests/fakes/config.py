import copy
from typing import cast

from voicetotext.core.config import ConfigurationStore, ConfigValue


class FakeConfig(ConfigurationStore):
    """In-memory :class:`ConfigurationStore` with a ``save_count``.

    Tests can seed the config with an initial dict and assert that
    ``save()`` was called the expected number of times.
    """

    def __init__(self, initial: dict[str, ConfigValue] | None = None) -> None:
        self._data: dict[str, object] = {}
        for key, value in (initial or {}).items():
            self.set(key, value)
        self.save_count: int = 0

    def get(self, key: str, default: ConfigValue = None) -> ConfigValue:
        current: object = self._data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        if current is None or isinstance(current, (str, int, float, bool)):
            return current
        return default

    def set(self, key: str, value: ConfigValue) -> None:
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = cast(dict[str, object], child)
        current[parts[-1]] = value

    def load(self) -> None:
        pass

    def save(self) -> None:
        self.save_count += 1

    def get_all(self) -> dict[str, object]:
        return copy.deepcopy(self._data)

    def replace_all(self, data: dict[str, object]) -> None:
        self._data = copy.deepcopy(data)
