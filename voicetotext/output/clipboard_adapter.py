from voicetotext.output.clipboard_writer import ClipboardWriter
from voicetotext.output.handler import TextOutputHandler


class ClipboardAdapter(TextOutputHandler):

    def __init__(self, writer: ClipboardWriter) -> None:
        self._writer = writer

    def output(self, text: str) -> None:
        self._writer.set_text(text)
