from pathlib import Path

import pytest

from skryb.output.dictionary_config import LiteralRule, RegexRule
from skryb.output.rule_counts import RuleCounts

_TEH = LiteralRule(("teh",), "the")
_ISSUE = RegexRule(r"\bissue (\d+)\b", r"#\1")


def test_counts_add_up_and_are_still_there_after_a_restart(tmp_path: Path) -> None:
    counts = RuleCounts(tmp_path / "rule_counts.json")
    counts.add([_TEH, _ISSUE, _TEH])
    counts.add([_TEH])

    again = RuleCounts(tmp_path / "rule_counts.json")
    assert (again.count(_TEH), again.count(_ISSUE)) == (3, 1)


def test_an_edited_rule_counts_from_zero(tmp_path: Path) -> None:
    counts = RuleCounts(tmp_path / "rule_counts.json")
    counts.add([_TEH])
    assert counts.count(LiteralRule(("teh", "hte"), "the")) == 0
    assert counts.count(LiteralRule(("teh",), "The")) == 0


@pytest.mark.parametrize("content", [b"", b"[3]", "{\"x\": 1}".encode("utf-16"), b'{"a": true, "b": -2, "c": "4"}'])
def test_an_unreadable_counts_file_starts_again_from_zero(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "rule_counts.json"
    path.write_bytes(content)
    counts = RuleCounts(path)

    counts.add([_TEH])

    assert RuleCounts(path).count(_TEH) == 1


def test_thousands_of_firings_of_unicode_rules_are_counted_exactly(tmp_path: Path) -> None:
    rule = LiteralRule(("tokyo", "toukyou"), "東京")
    counts = RuleCounts(tmp_path / "rule_counts.json")
    for _ in range(2000):
        counts.add([rule])
    assert RuleCounts(tmp_path / "rule_counts.json").count(rule) == 2000
