import json
import logging
import re
from pathlib import Path

from voicetotext.output.handler import TextOutputHandler


_logger = logging.getLogger(__name__)


def load_dictionary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _logger.warning("Dictionary ignored: %s", exc)
        return {}
    if not isinstance(document, dict):
        _logger.warning("Dictionary ignored: root must be a JSON object")
        return {}
    rules: dict[str, str] = {}
    for key, value in document.items():
        if not isinstance(key, str) or not key:
            _logger.warning("Dictionary entry ignored: key must be a non-empty string")
            continue
        if any(character.isspace() for character in key):
            _logger.warning("Dictionary entry '%s' ignored: phrases are unsupported", key)
            continue
        if not isinstance(value, str) or not value:
            _logger.warning("Dictionary entry '%s' ignored: value must be a non-empty string", key)
            continue
        rules[key] = value
    return rules


class TextFilterDecorator(TextOutputHandler):
    def __init__(self, inner: TextOutputHandler, rules: dict[str, str]) -> None:
        self._inner = inner
        self._rules = dict(rules)
        escaped = sorted(
            (re.escape(key) for key in self._rules),
            key=len,
            reverse=True,
        )
        self._pattern = (
            re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)")
            if escaped
            else None
        )

    def output(self, text: str) -> None:
        if self._pattern is None:
            self._inner.output(text)
            return
        filtered = self._pattern.sub(
            lambda match: self._rules[match.group(0)],
            text,
        )
        self._inner.output(filtered)
