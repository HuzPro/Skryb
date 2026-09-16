import re
from dataclasses import dataclass

_SENTENCE_ENDINGS = ".!?\n\r"
_TERMINAL_PUNCTUATION = ".!?,;:"
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class CaretContext:
    before: str
    selected: str
    after: str


NO_CONTEXT = CaretContext("", "", "")


def fit_to_context(text: str, context: CaretContext) -> str:
    if context == NO_CONTEXT:
        return text
    lead = _leading_whitespace(context.selected) or _glue_space(context.before)
    trail = _trailing_whitespace(context.selected)
    core = _match_preceding_case(text.strip(), context.before)
    following = context.after.lstrip(" \t")
    if not following:
        return lead + core + (trail or _trailing_whitespace(text))
    if _is_punctuation(following[0]):
        return lead + _strip_terminal_punctuation(core) + trail
    if following[0].islower() or following[0].isdigit():
        core = _strip_terminal_punctuation(core)
    if not _starts_with_whitespace(context.after) and not trail:
        core += " "
    return lead + core + trail


def _match_preceding_case(core: str, before: str) -> str:
    preceding = before.rstrip(" \t")
    if not preceding or preceding[-1] in _SENTENCE_ENDINGS:
        return _capitalize_first_word(core)
    return _lowercase_first_word(core)


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
    return core.rstrip(_TERMINAL_PUNCTUATION).rstrip()


def _is_punctuation(character: str) -> bool:
    return not character.isspace() and not character.isalnum() and character != "_"


def _glue_space(before: str) -> str:
    if before and (before[-1].isalnum() or before[-1] == "_"):
        return " "
    return ""


def _starts_with_whitespace(text: str) -> bool:
    return bool(text) and text[0].isspace()


def _leading_whitespace(text: str) -> str:
    return text[:len(text) - len(text.lstrip())]


def _trailing_whitespace(text: str) -> str:
    return text[len(text.rstrip()):]
