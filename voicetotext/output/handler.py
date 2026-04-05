from abc import ABC, abstractmethod


class TextOutputHandler(ABC):

    @abstractmethod
    def output(self, text: str) -> None:
        pass
