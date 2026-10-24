from abc import ABC, abstractmethod


class ClipboardPreserver(ABC):

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def restore(self) -> None:
        pass
