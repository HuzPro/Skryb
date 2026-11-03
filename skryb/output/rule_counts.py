"""How many times each dictionary rule has fired, so a rule that never does can be seen and dropped.

Counted per rule as written: editing a rule starts its count again.
"""

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from skryb.output.dictionary_config import DictionaryRule, LiteralRule

_logger = logging.getLogger(__name__)


def rule_key(rule: DictionaryRule) -> str:
    if isinstance(rule, LiteralRule):
        return json.dumps(["literal", list(rule.spoken), rule.replacement], ensure_ascii=False)
    return json.dumps(["regex", rule.pattern, rule.replacement], ensure_ascii=False)


class RuleCounts:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._counts: dict[str, int] | None = None

    def add(self, rules: Sequence[DictionaryRule]) -> None:
        """One dictation's firings; saved at once, so a crash loses none."""
        with self._lock:
            counts = self._loaded()
            for rule in rules:
                key = rule_key(rule)
                counts[key] = counts.get(key, 0) + 1
            self._save(counts)

    def count(self, rule: DictionaryRule) -> int:
        with self._lock:
            return self._loaded().get(rule_key(rule), 0)

    def _loaded(self) -> dict[str, int]:
        if self._counts is None:
            self._counts = self._read()
        return self._counts

    def _read(self) -> dict[str, int]:
        try:
            document: object = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _logger.warning("Rule counts start again: %s", exc)
            return {}
        if not isinstance(document, dict):
            return {}
        return {
            key: value for key, value in cast(dict[str, object], document).items()
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }

    def _save(self, counts: dict[str, int]) -> None:
        try:
            self._write(counts)
        except OSError as exc:
            _logger.warning("Could not save the rule counts: %s", exc)

    def _write(self, counts: dict[str, int]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        staged = self._path.with_suffix(".tmp")
        staged.write_text(json.dumps(counts, ensure_ascii=False), encoding="utf-8")
        staged.replace(self._path)
