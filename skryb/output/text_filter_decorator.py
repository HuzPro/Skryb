from skryb.output.dictation_trace import current_trace
from skryb.output.dictionary_config import load_dictionary
from skryb.output.handler import TextOutputHandler
from skryb.output.text_pipeline import TextPipeline


class TextFilterDecorator(TextOutputHandler):
    def __init__(self, inner: TextOutputHandler, pipeline: TextPipeline) -> None:
        self._inner = inner
        self._pipeline = pipeline

    def output(self, text: str) -> None:
        processed = self._pipeline.process(text)
        trace = current_trace()
        if trace is not None:
            trace.typed = processed
        self._inner.output(processed)


__all__ = ["TextFilterDecorator", "load_dictionary"]
