import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryRule,
    LiteralRule,
    RegexRule,
)


class PipelineMode(Enum):
    BATCH = "batch"
    STREAMING = "streaming"


class TokenKind(Enum):
    WORD = "word"
    WHITESPACE = "whitespace"
    PUNCTUATION = "punctuation"


@dataclass(frozen=True)
class TextToken:
    text: str
    start: int
    end: int
    kind: TokenKind


@dataclass(frozen=True)
class _Replacement:
    start: int
    end: int
    value: str
    priority: int


@dataclass(frozen=True)
class _Command:
    spoken: str
    replacement: str
    spacing: str
    capitalize_after: bool = False


_TOKEN_PATTERN = re.compile(r"\w+|\s+|[^\w\s]+", re.UNICODE)
_FILLER_PATTERN = re.compile(r"(?<!\w)(?:um+|uh+|erm+)(?!\w)", re.IGNORECASE)
_COMMA_BEFORE_FILLER = re.compile(r",[ \t]*$")


def _phrase_pattern(phrase: str) -> str:
    return r"\s+".join(re.escape(word) for word in phrase.split())


_COMMANDS = (
    _Command("question mark", "?", "punctuation", True),
    _Command("exclamation mark", "!", "punctuation", True),
    _Command("new paragraph", "\n\n", "break", True),
    _Command("open paren", "(", "opener"),
    _Command("close paren", ")", "punctuation"),
    _Command("open quote", '"', "opener"),
    _Command("close quote", '"', "punctuation"),
    _Command("new line", "\n", "break"),
    _Command("semicolon", ";", "punctuation"),
    _Command("period", ".", "punctuation", True),
    _Command("comma", ",", "punctuation"),
    _Command("colon", ":", "punctuation"),
    _Command("tab", "\t", "break"),
    _Command("dash", "-", "punctuation"),
)
_COMMAND_BY_SPOKEN = {command.spoken: command for command in _COMMANDS}
_COMMAND_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(_phrase_pattern(command.spoken) for command in _COMMANDS)
    + r")(?!\w)",
    re.IGNORECASE,
)
_LITERAL_PATTERN = re.compile(r"(?<!\w)literal(?!\w)\s+", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


class TextPipeline:
    def __init__(self, stages: tuple[Callable[[str], str], ...]) -> None:
        self._stages = stages

    @classmethod
    def for_batch(cls, config: DictionaryConfig) -> "TextPipeline":
        stages: list[Callable[[str], str]] = []
        if config.options.remove_fillers:
            stages.append(remove_fillers)
        if config.options.spoken_commands:
            stages.append(apply_spoken_commands)
        stages.append(_RuleStage(config.rules).apply)
        return cls(tuple(stages))

    @classmethod
    def for_streaming(cls, config: DictionaryConfig) -> "TextPipeline":
        stages: list[Callable[[str], str]] = []
        if config.options.remove_fillers:
            stages.append(remove_fillers)
        safe_rules = tuple(
            safe_rule
            for rule in config.rules
            if (safe_rule := _streaming_rule(rule)) is not None
        )
        stages.append(_RuleStage(safe_rules).apply)
        return cls(tuple(stages))

    def process(self, text: str) -> str:
        result = text
        for stage in self._stages:
            result = stage(result)
        return result


def build_pipeline(
    config: DictionaryConfig, mode: PipelineMode,
) -> TextPipeline:
    if mode is PipelineMode.STREAMING:
        return TextPipeline.for_streaming(config)
    return TextPipeline.for_batch(config)


def tokenize(text: str) -> tuple[TextToken, ...]:
    return tuple(
        TextToken(
            text=match.group(0),
            start=match.start(),
            end=match.end(),
            kind=_token_kind(match.group(0)),
        )
        for match in _TOKEN_PATTERN.finditer(text)
    )


def remove_fillers(text: str) -> str:
    replacements = [
        _filler_replacement(text, match)
        for match in _FILLER_PATTERN.finditer(text)
    ]
    if not replacements:
        return text
    return _apply_replacements(text, replacements)


def apply_spoken_commands(text: str) -> str:
    actions = _command_actions(text)
    if not actions:
        return text
    formatter = _CommandFormatter()
    cursor = 0
    for action, command in actions:
        formatter.append_text(text[cursor:action.start])
        if command is None:
            formatter.append_text(action.value)
        else:
            formatter.append_command(command)
        cursor = action.end
    formatter.append_text(text[cursor:])
    return formatter.value()


class _RuleStage:
    def __init__(self, rules: tuple[DictionaryRule, ...]) -> None:
        self._finders = tuple(
            finder
            for index, rule in enumerate(rules)
            if (finder := _make_rule_finder(rule, index)) is not None
        )

    def apply(self, text: str) -> str:
        replacements = [
            replacement
            for finder in self._finders
            for replacement in finder(text)
        ]
        return _apply_replacements(text, replacements)


class _CommandFormatter:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._pending_spacing: str | None = None
        self._capitalize_next = False

    def append_text(self, text: str) -> None:
        if not text:
            return
        prepared = self._with_pending_spacing(text)
        if not prepared:
            return
        if self._capitalize_next:
            prepared, changed = _capitalize_first_word(prepared)
            self._capitalize_next = not changed
        self._parts.append(prepared)

    def append_command(self, command: _Command) -> None:
        if command.spacing == "opener":
            self._append_opener(command.replacement)
        elif command.spacing == "break":
            self._append_break(command.replacement)
        else:
            self._append_punctuation(command.replacement)
        self._capitalize_next = self._capitalize_next or command.capitalize_after

    def value(self) -> str:
        return "".join(self._parts)

    def _with_pending_spacing(self, text: str) -> str:
        if self._pending_spacing is None:
            return text
        stripped = text.lstrip()
        if not stripped:
            return ""
        prepared = self._pending_spacing + stripped
        self._pending_spacing = None
        return prepared

    def _append_opener(self, value: str) -> None:
        pending = self._pending_spacing
        self._pending_spacing = None
        self._rstrip_horizontal_space()
        current = "".join(self._parts)
        if pending is not None:
            self._parts.append(pending)
        elif current and not current[-1].isspace() and current[-1] not in "([\"":
            self._parts.append(" ")
        self._parts.append(value)
        self._pending_spacing = ""

    def _append_break(self, value: str) -> None:
        self._pending_spacing = None
        self._rstrip_horizontal_space()
        self._parts.append(value)
        self._pending_spacing = ""

    def _append_punctuation(self, value: str) -> None:
        self._pending_spacing = None
        self._rstrip_horizontal_space()
        self._parts.append(value)
        self._pending_spacing = " "

    def _rstrip_horizontal_space(self) -> None:
        if not self._parts:
            return
        self._parts[-1] = self._parts[-1].rstrip(" \t")


def _command_actions(text: str) -> list[tuple[_Replacement, _Command | None]]:
    actions: list[tuple[_Replacement, _Command | None]] = []
    escaped_ranges: list[tuple[int, int]] = []
    for literal_match in _LITERAL_PATTERN.finditer(text):
        escaped = _escaped_text_at(text, literal_match.end())
        if escaped is None:
            continue
        end, value = escaped
        action = _Replacement(literal_match.start(), end, value, -1)
        actions.append((action, None))
        escaped_ranges.append((literal_match.start(), end))
    for match in _COMMAND_PATTERN.finditer(text):
        if any(start <= match.start() < end for start, end in escaped_ranges):
            continue
        spoken = " ".join(match.group(0).casefold().split())
        command = _COMMAND_BY_SPOKEN[spoken]
        actions.append(
            (_Replacement(match.start(), match.end(), command.replacement, 0), command)
        )
    actions.sort(key=lambda item: (item[0].start, item[0].priority))
    selected: list[tuple[_Replacement, _Command | None]] = []
    cursor = 0
    for item in actions:
        if item[0].start < cursor:
            continue
        selected.append(item)
        cursor = item[0].end
    return selected


def _escaped_text_at(text: str, start: int) -> tuple[int, str] | None:
    command_match = _COMMAND_PATTERN.match(text, start)
    if command_match is None:
        return None
    return command_match.end(), command_match.group(0)


def _make_rule_finder(
    rule: DictionaryRule, rule_index: int,
) -> Callable[[str], list[_Replacement]] | None:
    if not rule.enabled:
        return None
    if isinstance(rule, LiteralRule):
        return _literal_finder(rule, rule_index)
    return _regex_finder(rule, rule_index)


def _literal_finder(
    rule: LiteralRule, rule_index: int,
) -> Callable[[str], list[_Replacement]]:
    patterns = tuple(
        re.compile(
            _literal_pattern(alias, rule.whole_words),
            0 if rule.case_sensitive else re.IGNORECASE,
        )
        for alias in rule.spoken
    )

    def find(text: str) -> list[_Replacement]:
        return [
            _Replacement(
                match.start(),
                match.end(),
                rule.replacement,
                rule_index,
            )
            for pattern in patterns
            for match in pattern.finditer(text)
        ]

    return find


def _regex_finder(
    rule: RegexRule, rule_index: int,
) -> Callable[[str], list[_Replacement]]:
    pattern = re.compile(rule.pattern)

    def find(text: str) -> list[_Replacement]:
        return [
            _Replacement(match.start(), match.end(), match.expand(rule.replacement), rule_index)
            for match in pattern.finditer(text)
            if match.start() != match.end()
        ]

    return find


def _apply_replacements(text: str, replacements: list[_Replacement]) -> str:
    if not replacements:
        return text
    ordered = sorted(
        replacements,
        key=lambda replacement: (
            replacement.start,
            -(replacement.end - replacement.start),
            replacement.priority,
        ),
    )
    parts: list[str] = []
    cursor = 0
    for replacement in ordered:
        if replacement.start < cursor:
            continue
        parts.append(text[cursor:replacement.start])
        parts.append(replacement.value)
        cursor = replacement.end
    parts.append(text[cursor:])
    return "".join(parts)


def _literal_pattern(alias: str, whole_words: bool) -> str:
    parts = [
        r"\s+" if token.kind is TokenKind.WHITESPACE else re.escape(token.text)
        for token in tokenize(alias)
    ]
    pattern = "".join(parts)
    if whole_words:
        return r"(?<!\w)(?:" + pattern + r")(?!\w)"
    return pattern


def _token_kind(text: str) -> TokenKind:
    if text.isspace():
        return TokenKind.WHITESPACE
    if text[0].isalnum() or text[0] == "_":
        return TokenKind.WORD
    return TokenKind.PUNCTUATION


def _streaming_rule(rule: DictionaryRule) -> DictionaryRule | None:
    if isinstance(rule, RegexRule):
        return rule
    safe_aliases = tuple(alias for alias in rule.spoken if len(alias.split()) == 1)
    if not safe_aliases:
        return None
    return LiteralRule(
        spoken=safe_aliases,
        replacement=rule.replacement,
        case_sensitive=rule.case_sensitive,
        whole_words=rule.whole_words,
        phonetic=rule.phonetic,
        enabled=rule.enabled,
    )


def _filler_replacement(text: str, match: re.Match[str]) -> _Replacement:
    start = match.start()
    end = match.end()
    if text[end:end + 1] == ",":
        return _Replacement(start, _past_comma_spacing(text, end + 1), "", 0)
    preceding_comma = _COMMA_BEFORE_FILLER.search(text, 0, start)
    if preceding_comma is not None and _filler_ends_clause(text, end):
        return _Replacement(preceding_comma.start(), end, "", 0)
    if end < len(text) and text[end] in " \t":
        end += 1
    elif start > 0 and text[start - 1] in " \t":
        start -= 1
    return _Replacement(start, end, "", 0)


def _past_comma_spacing(text: str, index: int) -> int:
    if text[index:index + 1] in (" ", "\t"):
        return index + 1
    return index


def _filler_ends_clause(text: str, end: int) -> bool:
    following = text[end:end + 1]
    return not (following.isalnum() or following.isspace())


def _capitalize_first_word(text: str) -> tuple[str, bool]:
    match = _WORD_PATTERN.search(text)
    if match is None:
        return text, False
    word = match.group(0)
    capitalized = word[:1].upper() + word[1:]
    return text[:match.start()] + capitalized + text[match.end():], True
