from voicetotext.core.config import ConfigurationStore, ConfigValue


class FakeConfig(ConfigurationStore):
    """In-memory :class:`ConfigurationStore` with a ``save_count``.

    Tests can seed the config with an initial dict and assert that
    ``save()`` was called the expected number of times.
    """

    def __init__(self, initial: dict[str, ConfigValue] | None = None) -> None:
        self._data: dict[str, ConfigValue] = dict(initial or {})
        self.save_count: int = 0

    def get(self, key: str, default: ConfigValue = None) -> ConfigValue:
        return self._data.get(key, default)

    def set(self, key: str, value: ConfigValue) -> None:
        self._data[key] = value

    def load(self) -> None:
        pass

    def save(self) -> None:
        self.save_count += 1

    def get_all(self) -> dict[str, object]:
        return dict(self._data)
