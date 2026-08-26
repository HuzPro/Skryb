from voicetotext.output.dictionary_config import load_dictionary
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.text_pipeline import TextPipeline


class TextFilterDecorator(TextOutputHandler):
    def __init__(self, inner: TextOutputHandler, pipeline: TextPipeline) -> None:
        self._inner = inner
        self._pipeline = pipeline

    def output(self, text: str) -> None:
        self._inner.output(self._pipeline.process(text))


__all__ = ["TextFilterDecorator", "load_dictionary"]
