"""One English gloss per written Arabic word, asked for in one grammar-locked JSON call.

The grammar spells out the line's Arabic words as fixed literals, in order, and leaves the model
only the glosses to write, so it cannot skip, merge or reorder a word. Prefixes and suffixes stay
on their word and fold into its gloss, as word-by-word Quran translations do: "وبيتهم" is "and
their house". A word with no Arabic letter (English, a number, punctuation) is its own gloss and
is never sent.
"""

import json
import re
from collections.abc import Callable

# Arabic letters only: "؟" or "،" standing alone is punctuation, its own gloss.
_ARABIC_LETTER = re.compile(
    "[\u0620-\u063f\u0641-\u064a\u066e\u066f\u0671-\u06d3\u06d5\u06ee\u06ef\u06fa-\u06fc\u06ff"
    "\u0750-\u077f\u08a0-\u08c9\ufb50-\ufdfb\ufe70-\ufefc]",
)
# A gloss is English: no Arabic script, quote, backslash or control character, and no leading space.
_GLOSS_CHAR = r'[^"\\\x00-\x1f\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]'
_GLOSS_START = r'[^ "\\\x00-\x1f\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]'
MAX_GLOSS_CHARS = 60
# A long dictation is glossed this many Arabic words per call, so no answer outgrows the context.
WORDS_PER_CALL = 48

Gloss = list[tuple[str, str]]
# prompt, max_tokens, grammar -> the model's answer
Complete = Callable[[str, int, str | None], str]


def is_arabic(word: str) -> bool:
    return _ARABIC_LETTER.search(word) is not None


def gloss_line(line: str, complete: Complete, translation: str = "") -> Gloss:
    """Every written word of *line* with its English gloss, in order."""
    words = line.split()
    arabic = [word for word in words if is_arabic(word)]
    chunked = len(arabic) > WORDS_PER_CALL
    glosses: list[str] = []
    for start in range(0, len(arabic), WORDS_PER_CALL):
        batch = arabic[start:start + WORDS_PER_CALL]
        context, hint = (" ".join(batch), "") if chunked else (line, translation)
        glosses += read_glosses(complete(gloss_prompt(context, batch, hint), _budget(batch), gloss_grammar(batch)), batch)
    found = iter(glosses)
    return [(word, next(found) if is_arabic(word) else word) for word in words]


def gloss_prompt(line: str, words: list[str], translation: str = "") -> str:
    example = json.dumps([{"w": "وبيتهم", "g": "and their house"}, {"w": "بالسيارة", "g": "in the car"}], ensure_ascii=False, separators=(",", ":"))
    lines = [
        "Gloss this spoken Arabic (often Gulf dialect) word by word in English, in order.",
        "Give every written word one short English gloss that includes its attached prefixes and suffixes,"
        " as a word-by-word Quran translation does. For example: " + example,
        "",
        f"Sentence: {line}",
    ]
    if translation:
        lines.append(f"Translation: {translation}")
    lines += ["", f"Answer with a JSON list of these {len(words)} words, each with its gloss."]
    return "\n".join(lines)


def gloss_grammar(words: list[str]) -> str:
    """A llama.cpp grammar whose only answers are *words*, in order, each with one gloss."""
    rules = [
        f'w{index} ::= "{{\\"w\\":" {_literal(json.dumps(word, ensure_ascii=False))} ",\\"g\\":\\"" gloss "\\"}}"'
        for index, word in enumerate(words)
    ]
    root = 'root ::= "[" ' + ' "," '.join(f"w{index}" for index in range(len(words))) + ' "]"'
    gloss = f"gloss ::= {_GLOSS_START} {_GLOSS_CHAR}{{0,{MAX_GLOSS_CHARS - 1}}}"
    return "\n".join([root, *rules, gloss]) + "\n"


def read_glosses(answer: str, words: list[str]) -> list[str]:
    """The glosses in *answer*; ``ValueError`` unless it holds exactly *words*, in order."""
    items = json.loads(answer)
    if not isinstance(items, list) or len(items) != len(words):
        raise ValueError(f"expected {len(words)} glossed words, got {answer[:200]!r}")
    glosses = []
    for item, word in zip(items, words, strict=True):
        if not isinstance(item, dict) or item.get("w") != word or not isinstance(item.get("g"), str):
            raise ValueError(f"the gloss of {word!r} came back as {item!r}")
        glosses.append(" ".join(item["g"].split()))
    return glosses


def _literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _budget(words: list[str]) -> int:
    """Tokens enough for the JSON of *words*: the words echoed back plus a gloss each."""
    return 64 + sum(2 * len(word) + 24 for word in words)
