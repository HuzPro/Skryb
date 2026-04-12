from abc import ABC, abstractmethod


ConfigValue = str | int | float | bool | None


class ConfigurationStore(ABC):

    @abstractmethod
    def get(self, key: str, default: ConfigValue = None) -> ConfigValue:
        pass

    @abstractmethod
    def set(self, key: str, value: ConfigValue) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def get_all(self) -> dict[str, object]:
        pass
