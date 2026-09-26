from abc import ABC, abstractmethod


class TextOutputHandler(ABC):

    @abstractmethod
    def output(self, text: str) -> None:
        pass

    def output_verbatim(self, text: str) -> None:
        """Deliver ``text`` without the text pipeline, where one applies.

        A voice edit's replacement is already finished text; the dictation
        cleanup stages would only rewrite it again.
        """
        self.output(text)
