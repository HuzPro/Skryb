"""Arabic to English: the sentence in English, and an English gloss under each written word."""

import re
from dataclasses import dataclass

from skryb.translation.gloss import Complete, Gloss, gloss_line, is_arabic

# Hy-MT2's documented English prompt, which every candidate model was scored with.
TRANSLATE_PROMPT = (
    "Translate the following text into English. Note that you should only output the translated result "
    "without any additional explanation:\n\n{text}"
)
# A long dictation goes to the model in pieces of at most this many words, cut after a sentence where it can be.
WORDS_PER_PIECE = 160
_SENTENCE_END = re.compile(r"(?<=[.!?؟\n])\s+")


@dataclass(frozen=True)
class Translation:
    english: str
    words: Gloss

    def json(self) -> dict[str, object]:
        return {"english": self.english, "words": [list(pair) for pair in self.words]}


def has_arabic(line: str) -> bool:
    return any(is_arabic(word) for word in line.split())


class Translator:
    def __init__(self, complete: Complete) -> None:
        self._complete = complete

    def translate(self, line: str) -> str:
        """The English sentence; a line with no Arabic in it is its own translation."""
        if not has_arabic(line):
            return " ".join(line.split())
        return " ".join(
            " ".join(self._complete(TRANSLATE_PROMPT.format(text=piece), 64 + 4 * len(piece.split()), None).split())
            for piece in pieces(line)
        )

    def gloss(self, line: str, translation: str = "") -> Gloss:
        return gloss_line(line, self._complete, translation)

    def annotate(self, line: str) -> Translation:
        english = self.translate(line)
        return Translation(english, self.gloss(line, english))


def pieces(text: str) -> list[str]:
    """*text* in runs of whole sentences of at most ``WORDS_PER_PIECE`` words; a longer sentence is cut by words."""
    words_of = [sentence.split() for sentence in _SENTENCE_END.split(text) if sentence.strip()]
    runs: list[list[str]] = []
    for words in words_of:
        while len(words) > WORDS_PER_PIECE:
            runs.append(words[:WORDS_PER_PIECE])
            words = words[WORDS_PER_PIECE:]
        if runs and len(runs[-1]) + len(words) <= WORDS_PER_PIECE:
            runs[-1] = runs[-1] + words
        elif words:
            runs.append(words)
    return [" ".join(run) for run in runs]
