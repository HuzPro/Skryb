import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DictionaryOptions:
    remove_fillers: bool = False
    spoken_commands: bool = False
    numbers: bool = False
    punctuation_model: bool = False


@dataclass(frozen=True)
class LiteralRule:
    spoken: tuple[str, ...]
    replacement: str
    case_sensitive: bool = False
    whole_words: bool = True
    phonetic: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class RegexRule:
    pattern: str
    replacement: str
    enabled: bool = True


DictionaryRule = LiteralRule | RegexRule


@dataclass(frozen=True)
class DictionaryConfig:
    options: DictionaryOptions = DictionaryOptions()
    rules: tuple[DictionaryRule, ...] = ()
    version: int = 2


def load_dictionary(path: Path) -> DictionaryConfig:
    if not path.exists():
        return DictionaryConfig()
    try:
        document: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _logger.warning("Dictionary ignored: %s", exc)
        return DictionaryConfig()
    if not isinstance(document, dict):
        _logger.warning("Dictionary ignored: root must be a JSON object")
        return DictionaryConfig()
    raw = cast(dict[object, object], document)
    if "version" not in raw:
        return _migrate_v1(raw)
    return _load_v2(raw)


def dictionary_to_json(config: DictionaryConfig) -> dict[str, object]:
    return {
        "version": 2,
        "options": {
            "remove_fillers": config.options.remove_fillers,
            "spoken_commands": config.options.spoken_commands,
            "numbers": config.options.numbers,
            "punctuation_model": config.options.punctuation_model,
        },
        "rules": [_rule_to_json(rule) for rule in config.rules],
    }


def _migrate_v1(document: dict[object, object]) -> DictionaryConfig:
    rules: list[DictionaryRule] = []
    for key, value in document.items():
        if not isinstance(key, str) or not key.strip():
            _logger.warning("Dictionary entry ignored: key must be a non-empty string")
            continue
        if not isinstance(value, str) or not value:
            _logger.warning(
                "Dictionary entry '%s' ignored: value must be a non-empty string",
                key,
            )
            continue
        rules.append(
            LiteralRule(
                spoken=(key,),
                replacement=value,
                case_sensitive=True,
                whole_words=True,
            )
        )
    return DictionaryConfig(rules=tuple(rules))


def _load_v2(document: dict[object, object]) -> DictionaryConfig:
    if not _is_version_two(document.get("version")):
        _logger.warning("Dictionary ignored: version must be 2")
        return DictionaryConfig()
    raw_options = document.get("options")
    raw_rules = document.get("rules")
    if not isinstance(raw_options, dict) or not isinstance(raw_rules, list):
        _logger.warning("Dictionary ignored: options and rules have invalid types")
        return DictionaryConfig()
    options = _load_options(cast(dict[object, object], raw_options))
    if options is None:
        return DictionaryConfig()
    rules: list[DictionaryRule] = []
    for index, raw_rule in enumerate(cast(list[object], raw_rules)):
        rule = _load_rule(raw_rule, index)
        if rule is not None:
            rules.append(rule)
    return DictionaryConfig(options=options, rules=tuple(rules))


def _is_version_two(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 2


def _load_options(document: dict[object, object]) -> DictionaryOptions | None:
    names = (
        "remove_fillers",
        "spoken_commands",
        "numbers",
        "punctuation_model",
    )
    values: dict[str, bool] = {}
    for name in names:
        value = document.get(name, False)
        if not isinstance(value, bool):
            _logger.warning("Dictionary ignored: option '%s' must be boolean", name)
            return None
        values[name] = value
    return DictionaryOptions(
        remove_fillers=values["remove_fillers"],
        spoken_commands=values["spoken_commands"],
        numbers=values["numbers"],
        punctuation_model=values["punctuation_model"],
    )


def _load_rule(raw_rule: object, index: int) -> DictionaryRule | None:
    if not isinstance(raw_rule, dict):
        _warn_bad_rule(index, "rule must be an object")
        return None
    document = cast(dict[object, object], raw_rule)
    rule_type = document.get("type")
    if rule_type == "literal":
        return _load_literal_rule(document, index)
    if rule_type == "regex":
        return _load_regex_rule(document, index)
    _warn_bad_rule(index, "type must be 'literal' or 'regex'")
    return None


def _load_literal_rule(
    document: dict[object, object], index: int,
) -> LiteralRule | None:
    raw_spoken = document.get("spoken")
    replacement = document.get("replacement")
    if not isinstance(raw_spoken, list):
        _warn_bad_rule(index, "spoken must be a list")
        return None
    spoken_values = cast(list[object], raw_spoken)
    if not spoken_values or not all(
        isinstance(alias, str) and bool(alias.strip()) for alias in spoken_values
    ):
        _warn_bad_rule(index, "spoken aliases must be non-empty strings")
        return None
    if not isinstance(replacement, str):
        _warn_bad_rule(index, "replacement must be a string")
        return None
    booleans = _literal_booleans(document, index)
    if booleans is None:
        return None
    case_sensitive, whole_words, phonetic, enabled = booleans
    return LiteralRule(
        spoken=tuple(alias.strip() for alias in cast(list[str], raw_spoken)),
        replacement=replacement,
        case_sensitive=case_sensitive,
        whole_words=whole_words,
        phonetic=phonetic,
        enabled=enabled,
    )


def _literal_booleans(
    document: dict[object, object], index: int,
) -> tuple[bool, bool, bool, bool] | None:
    defaults = {
        "case_sensitive": False,
        "whole_words": True,
        "phonetic": False,
        "enabled": True,
    }
    values: list[bool] = []
    for name, default in defaults.items():
        value = document.get(name, default)
        if not isinstance(value, bool):
            _warn_bad_rule(index, f"{name} must be boolean")
            return None
        values.append(value)
    return values[0], values[1], values[2], values[3]


def _load_regex_rule(
    document: dict[object, object], index: int,
) -> RegexRule | None:
    pattern = document.get("pattern")
    replacement = document.get("replacement")
    enabled = document.get("enabled", True)
    if not isinstance(pattern, str) or not pattern:
        _warn_bad_rule(index, "pattern must be a non-empty string")
        return None
    if not isinstance(replacement, str):
        _warn_bad_rule(index, "replacement must be a string")
        return None
    if not isinstance(enabled, bool):
        _warn_bad_rule(index, "enabled must be boolean")
        return None
    try:
        compiled = re.compile(pattern)
        compiled.sub(replacement, "")
    except re.error as exc:
        _warn_bad_rule(index, f"invalid regex or replacement: {exc}")
        return None
    return RegexRule(pattern=pattern, replacement=replacement, enabled=enabled)


def _warn_bad_rule(index: int, reason: str) -> None:
    _logger.warning("Dictionary rule %d ignored: %s", index + 1, reason)


def _rule_to_json(rule: DictionaryRule) -> dict[str, object]:
    if isinstance(rule, LiteralRule):
        return {
            "type": "literal",
            "spoken": list(rule.spoken),
            "replacement": rule.replacement,
            "case_sensitive": rule.case_sensitive,
            "whole_words": rule.whole_words,
            "phonetic": rule.phonetic,
            "enabled": rule.enabled,
        }
    return {
        "type": "regex",
        "pattern": rule.pattern,
        "replacement": rule.replacement,
        "enabled": rule.enabled,
    }
