from abc import ABC, abstractmethod
from typing import Any


class ConfigurationStore(ABC):

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def get_all(self) -> dict:
        pass
