import re
from dataclasses import dataclass

_NEWLINES = "\n\r"
_SENTENCE_ENDINGS = ".!?" + _NEWLINES
_MID_SENTENCE_PUNCTUATION = ",;:)]}"
_TERMINAL_PUNCTUATION = ".!?,;:"
_PUNCTUATION_THAT_FOLLOWS = ".,;:!?)]}\"'"
_CLOSING_MARKS = ")]}\"'\u201d\u2019"
_SPACED_PUNCTUATION = ".!?,;:\u2026"
_INVISIBLE_CHARACTERS = "\ufeff\u200b\u200c\u200d\u2060\ufffc"
_WITHOUT_INVISIBLE_CHARACTERS = str.maketrans("", "", _INVISIBLE_CHARACTERS)
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

CONTEXT_CHARS = 40


@dataclass(frozen=True)
class CaretContext:
    before: str
    selected: str
    after: str


NO_CONTEXT = CaretContext("", "", "")


def fit_to_context(text: str, context: CaretContext) -> str:
    if context == NO_CONTEXT:
        return text
    return _fit_to_visible_context(text, _without_invisible_characters(context))


def _fit_to_visible_context(text: str, context: CaretContext) -> str:
    core = _match_preceding_case(text.strip(), context.before)
    lead = _lead(core, context)
    trail = _trailing_whitespace(context.selected)
    following = context.after.lstrip(" \t")
    if not following or following[0] in _NEWLINES:
        return lead + core + (trail or _trailing_whitespace(text))
    if following[0] in _PUNCTUATION_THAT_FOLLOWS:
        return lead + _strip_terminal_punctuation(core) + trail
    if _continues_the_sentence(following[0]):
        core = _strip_terminal_punctuation(core)
    elif not _starts_a_new_sentence(following[0]):
        return lead + core + trail
    if not _starts_with_whitespace(context.after) and not trail:
        core += " "
    return lead + core + trail


def _without_invisible_characters(context: CaretContext) -> CaretContext:
    """Drop the markers Lexical and ProseMirror leave around an empty line.

    A zero-width no-break space or an object replacement character is not
    text the speaker can see, so neither may decide a case or a full stop.
    """
    return CaretContext(
        before=context.before.translate(_WITHOUT_INVISIBLE_CHARACTERS),
        selected=context.selected.translate(_WITHOUT_INVISIBLE_CHARACTERS),
        after=context.after.translate(_WITHOUT_INVISIBLE_CHARACTERS),
    )


def _lead(core: str, context: CaretContext) -> str:
    if _starts_with_punctuation(core):
        return ""
    return _leading_whitespace(context.selected) or _glue_space(context.before)


def _starts_with_punctuation(core: str) -> bool:
    return bool(core) and core[0] in _TERMINAL_PUNCTUATION


def _match_preceding_case(core: str, before: str) -> str:
    preceding = _unwrapped(before.rstrip(" \t"))
    if not preceding or preceding[-1] in _SENTENCE_ENDINGS:
        return _capitalize_first_word(core)
    if _ends_an_unfinished_sentence(preceding[-1]):
        return _lowercase_first_word(core)
    return core


def _ends_an_unfinished_sentence(character: str) -> bool:
    return _is_word_character(character) or character in _MID_SENTENCE_PUNCTUATION


def _continues_the_sentence(character: str) -> bool:
    return character.islower() or character.isdigit()


def _starts_a_new_sentence(character: str) -> bool:
    return character.isupper() or character in _NEWLINES


def _capitalize_first_word(core: str) -> str:
    match = _WORD_PATTERN.search(core)
    if match is None:
        return core
    index = match.start()
    return core[:index] + core[index].upper() + core[index + 1:]


def _lowercase_first_word(core: str) -> str:
    match = _WORD_PATTERN.search(core)
    if match is None or _is_deliberate_capital(match.group(0)):
        return core
    index = match.start()
    return core[:index] + core[index].lower() + core[index + 1:]


def _is_deliberate_capital(word: str) -> bool:
    return word == "I" or any(character.isupper() for character in word[1:])


def _strip_terminal_punctuation(core: str) -> str:
    return core.rstrip(_TERMINAL_PUNCTUATION).rstrip() or core


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def _glue_space(before: str) -> str:
    preceding = _unwrapped(before)
    if preceding and (
        _is_word_character(preceding[-1]) or preceding[-1] in _SPACED_PUNCTUATION
    ):
        return " "
    return ""


def _unwrapped(text: str) -> str:
    return text.rstrip(_CLOSING_MARKS)


def _starts_with_whitespace(text: str) -> bool:
    return bool(text) and text[0].isspace()


def _leading_whitespace(text: str) -> str:
    return text[:len(text) - len(text.lstrip())]


def _trailing_whitespace(text: str) -> str:
    return text[len(text.rstrip()):]
