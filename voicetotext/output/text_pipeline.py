import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jellyfish
from text_to_num import alpha2digit

from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryRule,
    LiteralRule,
    RegexRule,
)
from voicetotext.output.punctuation_stage import (
    PunctuationBackend,
    PunctuationStage,
    create_punctuation_backend,
)

_logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    BATCH = "batch"
    STREAMING = "streaming"
    # Verbatim transcription is a request for what was actually said, so
    # every cleanup stage below would be working against it: filler
    # removal deletes the disfluencies it exists to keep, and the
    # dictionary rewrites the words it captured.
    VERBATIM = "verbatim"


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
class _PhoneticRule:
    replacement: str
    targets_by_key: dict[str, tuple[str, ...]]
    priority: int


@dataclass(frozen=True)
class _Command:
    spoken: str
    replacement: str
    spacing: str
    capitalize_after: bool = False


_SENTENCE_ENDINGS = (".", "?", "!")
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
_NUMBER = r"\d+(?:\.\d+)?"
_PERCENT_PATTERN = re.compile(
    rf"\b({_NUMBER})\s+percent\b", re.IGNORECASE,
)
_DOLLARS_AND_CENTS_PATTERN = re.compile(
    rf"\b({_NUMBER})\s+dollars\s+and\s+(\d+)\s+cents\b", re.IGNORECASE,
)
_DOLLARS_PATTERN = re.compile(
    rf"\b({_NUMBER})\s+dollars\b", re.IGNORECASE,
)
_PHONETIC_PRIORITY_BASE = 1 << 60
_PHONETIC_DISTANCE_LIMIT = 0.34
_MAX_PHONETIC_WORDS = 4
_FUNCTION_WORD_STOPLIST = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "if", "in", "is", "it", "of", "on", "or", "the", "to",
    "was", "with",
})


class TextPipeline:
    def __init__(self, stages: tuple[Callable[[str], str], ...]) -> None:
        self._stages = stages

    @classmethod
    def for_batch(
        cls,
        config: DictionaryConfig,
        punctuation_backend: PunctuationBackend | None = None,
        models_dir: Path | None = None,
    ) -> "TextPipeline":
        stages: list[Callable[[str], str]] = []
        if config.options.punctuation_model:
            backend = punctuation_backend
            if backend is None:
                if models_dir is None:
                    from voicetotext.core.paths import AppPaths

                    models_dir = AppPaths().models_dir
                backend = create_punctuation_backend(models_dir)
            stages.append(PunctuationStage(backend).apply)
        if config.options.remove_fillers:
            stages.append(remove_fillers)
        if config.options.spoken_commands:
            stages.append(apply_spoken_commands)
        stages.append(_RuleStage(config.rules).apply)
        if config.options.numbers:
            stages.append(format_numbers)
        stages.append(add_trailing_sentence_space)
        return cls(tuple(stages))

    @classmethod
    def passthrough(cls) -> "TextPipeline":
        return cls(())

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
    config: DictionaryConfig,
    mode: PipelineMode,
    punctuation_backend: PunctuationBackend | None = None,
    models_dir: Path | None = None,
) -> TextPipeline:
    if mode is PipelineMode.VERBATIM:
        return TextPipeline.passthrough()
    if mode is PipelineMode.STREAMING:
        return TextPipeline.for_streaming(config)
    return TextPipeline.for_batch(config, punctuation_backend, models_dir)


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


def add_trailing_sentence_space(text: str) -> str:
    return text + " " if text.endswith(_SENTENCE_ENDINGS) else text


def format_numbers(text: str) -> str:
    converted = alpha2digit(text, "en")
    converted = _PERCENT_PATTERN.sub(r"\1%", converted)
    converted = _DOLLARS_AND_CENTS_PATTERN.sub(_format_dollars_and_cents, converted)
    return _DOLLARS_PATTERN.sub(r"$\1", converted)


class _RuleStage:
    def __init__(self, rules: tuple[DictionaryRule, ...]) -> None:
        self._finders = tuple(
            finder
            for index, rule in enumerate(rules)
            if (finder := _make_rule_finder(rule, index)) is not None
        )
        self._phonetic_rules = tuple(
            phonetic_rule
            for index, rule in enumerate(rules)
            if (phonetic_rule := _make_phonetic_rule(rule, index)) is not None
        )

    def apply(self, text: str) -> str:
        replacements = [
            replacement
            for finder in self._finders
            for replacement in finder(text)
        ]
        replacements.extend(_phonetic_replacements(text, self._phonetic_rules))
        selected = _select_replacements(replacements)
        for replacement in selected:
            if replacement.priority >= _PHONETIC_PRIORITY_BASE:
                _logger.info(
                    "phonetic: '%s' -> '%s'",
                    text[replacement.start:replacement.end],
                    replacement.value,
                )
        return _render_replacements(text, selected)


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


def _make_phonetic_rule(
    rule: DictionaryRule, rule_index: int,
) -> _PhoneticRule | None:
    if not isinstance(rule, LiteralRule) or not rule.enabled or not rule.phonetic:
        return None
    targets: dict[str, list[str]] = {}
    for value in (*rule.spoken, rule.replacement):
        normalized = _normalize_phonetic_text(value)
        key = jellyfish.metaphone(normalized)
        if key:
            targets.setdefault(key, []).append(normalized)
    if not targets:
        return None
    return _PhoneticRule(
        replacement=rule.replacement,
        targets_by_key={key: tuple(values) for key, values in targets.items()},
        priority=_PHONETIC_PRIORITY_BASE + rule_index,
    )


def _phonetic_replacements(
    text: str, rules: tuple[_PhoneticRule, ...],
) -> list[_Replacement]:
    if not rules:
        return []
    matches: dict[tuple[int, int], list[tuple[float, _PhoneticRule]]] = {}
    for words in _word_runs(text):
        for start_index in range(len(words)):
            upper = min(len(words), start_index + _MAX_PHONETIC_WORDS)
            for end_index in range(start_index + 1, upper + 1):
                ngram = words[start_index:end_index]
                candidate = "".join(token.text for token in ngram).lower()
                if len(ngram) == 1 and candidate in _FUNCTION_WORD_STOPLIST:
                    continue
                key = jellyfish.metaphone(candidate)
                if not key:
                    continue
                span = (ngram[0].start, ngram[-1].end)
                for rule in rules:
                    replacement = _normalize_phonetic_text(rule.replacement)
                    if candidate == replacement:
                        continue
                    targets = rule.targets_by_key.get(key)
                    if targets is None:
                        continue
                    distance = min(
                        _normalized_distance(candidate, target) for target in targets
                    )
                    if distance <= _PHONETIC_DISTANCE_LIMIT:
                        matches.setdefault(span, []).append((distance, rule))
    replacements: list[_Replacement] = []
    for (start, end), candidates in matches.items():
        best_by_rule: dict[int, tuple[float, _PhoneticRule]] = {}
        for distance, rule in candidates:
            current = best_by_rule.get(rule.priority)
            if current is None or distance < current[0]:
                best_by_rule[rule.priority] = (distance, rule)
        ranked = sorted(best_by_rule.values(), key=lambda item: item[0])
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            continue
        _, rule = ranked[0]
        replacements.append(_Replacement(start, end, rule.replacement, rule.priority))
    return replacements


def _word_runs(text: str) -> tuple[tuple[TextToken, ...], ...]:
    runs: list[tuple[TextToken, ...]] = []
    current: list[TextToken] = []
    for token in tokenize(text):
        if token.kind is TokenKind.WORD:
            current.append(token)
        elif token.kind is TokenKind.PUNCTUATION and current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))
    return tuple(runs)


def _normalize_phonetic_text(text: str) -> str:
    return "".join(text.split()).lower()


def _normalized_distance(left: str, right: str) -> float:
    length = max(len(left), len(right))
    if length == 0:
        return 0.0
    return jellyfish.damerau_levenshtein_distance(left, right) / length


def _apply_replacements(text: str, replacements: list[_Replacement]) -> str:
    return _render_replacements(text, _select_replacements(replacements))


def _select_replacements(replacements: list[_Replacement]) -> list[_Replacement]:
    ordered = sorted(
        replacements,
        key=lambda replacement: (
            replacement.priority >= _PHONETIC_PRIORITY_BASE,
            replacement.start,
            -(replacement.end - replacement.start),
            replacement.priority,
        ),
    )
    selected: list[_Replacement] = []
    for replacement in ordered:
        if any(_replacements_overlap(replacement, current) for current in selected):
            continue
        selected.append(replacement)
    return sorted(selected, key=lambda replacement: replacement.start)


def _render_replacements(text: str, replacements: list[_Replacement]) -> str:
    if not replacements:
        return text
    parts: list[str] = []
    cursor = 0
    for replacement in replacements:
        parts.append(text[cursor:replacement.start])
        parts.append(replacement.value)
        cursor = replacement.end
    parts.append(text[cursor:])
    return "".join(parts)


def _replacements_overlap(left: _Replacement, right: _Replacement) -> bool:
    return left.start < right.end and right.start < left.end


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
        phonetic=False,
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


def _format_dollars_and_cents(match: re.Match[str]) -> str:
    cents = int(match.group(2))
    return f"${match.group(1)}.{cents:02d}"
