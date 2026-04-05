import pyperclip

from voicetotext.output.handler import TextOutputHandler


class ClipboardAdapter(TextOutputHandler):

    def output(self, text: str) -> None:
        pyperclip.copy(text)
