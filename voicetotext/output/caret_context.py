import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from text_to_num import alpha2digit

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
# A list item's marker at the start of a line: "3. ", "3) ", "- ", "* " or "• ".
_LIST_MARKER = re.compile(r"^[ \t]*(?:(?P<number>\d+)(?P<dot>[.)])|(?P<bullet>[-*•]))(?:[ \t]+|$)")
# Two or more whole numbers joined by commas at the end of a line: "1, 2, 3".
_NUMBER_LIST_END = re.compile(r"(?<![\d.])\d+(?:[ \t]*,[ \t]*\d+)+[ \t]*$")
_SPOKEN_NUMBER_RUN = re.compile(r"(?P<numbers>\d+(?:(?:[ \t]*,[ \t]*|[ \t]+)\d+)*)(?P<end>[.!?]?)")

CONTEXT_CHARS = 40


class ListLayout(Enum):
    """How a spoken list is laid out at the caret, going by the text already there."""

    INLINE = "inline"
    VERTICAL = "vertical"


@dataclass(frozen=True)
class CaretContext:
    before: str
    selected: str
    after: str


NO_CONTEXT = CaretContext("", "", "")

_LOGGED_TEXT_CHARS = 20


def _never() -> bool:
    return False


_logs_caret_text: Callable[[], bool] = _never


def log_caret_text_when(enabled: Callable[[], bool]) -> None:
    """Let caret log lines quote the text around the cursor while ``enabled()`` holds.

    That text belongs to whichever app has the focus, so otherwise a log
    line only says how long each part is.
    """
    global _logs_caret_text
    _logs_caret_text = enabled


def caret_log_line(context: CaretContext) -> str:
    if _logs_caret_text():
        return (
            f"before={context.before[-_LOGGED_TEXT_CHARS:]!r} "
            f"selected={context.selected[:_LOGGED_TEXT_CHARS]!r} "
            f"after={context.after[:_LOGGED_TEXT_CHARS]!r}"
        )
    return (
        f"{len(context.before)} characters before, {len(context.selected)} selected, "
        f"{len(context.after)} after"
    )


def fit_to_context(text: str, context: CaretContext) -> str:
    if context == NO_CONTEXT:
        return text
    return _fit_to_visible_context(text, _without_invisible_characters(context))


def list_layout(context: CaretContext) -> ListLayout | None:
    """The layout the text at the caret already uses, or ``None`` when it does not say.

    At the end of a numbered or bulleted item, or on the empty line after
    one, a spoken list goes on as more items. Anywhere else inside a line
    it stays on that line. An empty line, an empty box or a line ending in
    a colon leave the choice to the speaker's setting.
    """
    if context == NO_CONTEXT or context.selected.strip():
        return None
    visible = _without_invisible_characters(context)
    if _last_item(visible) is not None:
        return ListLayout.VERTICAL
    line = _current_line(visible.before)
    rest_of_line = _lines(visible.after)[0]
    if (line.strip() and not line.rstrip().endswith(":")) or rest_of_line.strip():
        return ListLayout.INLINE
    return None


@dataclass(frozen=True)
class _ListItem:
    marker: str
    number: int | None
    empty: bool
    on_current_line: bool
    lowercase: bool


def _last_item(context: CaretContext) -> _ListItem | None:
    """The list item the caret sits at the end of, or on the empty line after."""
    if _lines(context.after)[0].strip():
        return None
    lines = _lines(context.before)
    index = len(lines) - 1
    on_current_line = bool(lines[index].strip()) or index == 0
    if not on_current_line:
        index -= 1
    # The first line read may have been cut off mid-line; only when the
    # reading stopped at the start of the field is its start in view.
    if index == 0 and len(context.before) >= CONTEXT_CHARS:
        return None
    match = _LIST_MARKER.match(lines[index])
    if match is None:
        return None
    numbered = match["number"] is not None
    text = lines[index][match.end():].strip()
    return _ListItem(
        marker="{}" + match["dot"] + " " if numbered else match["bullet"] + " ",
        number=int(match["number"]) if numbered else None,
        empty=not text,
        on_current_line=on_current_line,
        lowercase=text[:1].islower(),
    )


def _lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _current_line(before: str) -> str:
    return _lines(before)[-1]


def _continue_vertical_list(text: str, item: _ListItem) -> str:
    """The dictation as the next items of the list the caret is in."""
    entries = [
        line[match.end():].strip() if (match := _LIST_MARKER.match(line)) else line.strip()
        for line in _lines(text)
    ]
    match_case = _lowercase_first_word if item.lowercase else _capitalize_first_word
    entries = [match_case(entry) for entry in entries if entry]
    if not entries:
        return text
    fills_empty_slot = item.on_current_line and item.empty
    first = item.number if fills_empty_slot or item.number is None else item.number + 1
    lines = []
    for offset, entry in enumerate(entries):
        number = None if first is None else first + offset
        marker = item.marker.format(number)
        lines.append(entry if offset == 0 and fills_empty_slot else marker + entry)
    return ("\n" if item.on_current_line and not item.empty else "") + "\n".join(lines)


def _continue_number_list(core: str, before: str) -> str | None:
    """Spoken numbers after "1, 2, 3" as ", 4, 5", or ``None`` when that does not apply."""
    if not _NUMBER_LIST_END.search(_current_line(before)):
        return None
    run = _SPOKEN_NUMBER_RUN.fullmatch(alpha2digit(core, "en", threshold=0).strip())
    if run is None:
        return None
    return ", " + ", ".join(re.findall(r"\d+", run["numbers"])) + run["end"]


def _fit_to_visible_context(text: str, context: CaretContext) -> str:
    item = _last_item(context) if not context.selected.strip() else None
    if item is not None:
        return _continue_vertical_list(text, item)
    core = _continue_number_list(text.strip(), context.before) or text.strip()
    core = _match_preceding_case(core, context.before)
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
