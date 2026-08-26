import json
from pathlib import Path

from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryOptions,
    LiteralRule,
    RegexRule,
)
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.text_filter_decorator import (
    TextFilterDecorator,
    load_dictionary,
)
from voicetotext.output.text_pipeline import TextPipeline
from voicetotext.transcription.streaming.streaming_output_writer import (
    StreamingOutputWriter,
)


class CapturingOutput(TextOutputHandler):
    def __init__(self) -> None:
        self.values: list[str] = []

    def output(self, text: str) -> None:
        self.values.append(text)


def _filter(
    rules: tuple[LiteralRule | RegexRule, ...],
) -> tuple[TextFilterDecorator, CapturingOutput]:
    inner = CapturingOutput()
    pipeline = TextPipeline.for_batch(DictionaryConfig(rules=rules))
    return TextFilterDecorator(inner, pipeline), inner


class TestTextFilterDecorator:
    def test_should_replace_tokens_adjacent_to_punctuation(self) -> None:
        handler, inner = _filter((LiteralRule(("god",), "God", True),))

        handler.output('god, (god)! "god"')

        assert inner.values == ['God, (God)! "God"']

    def test_should_apply_unicode_rules(self) -> None:
        handler, inner = _filter((
            LiteralRule(("café",), "Café", True),
            LiteralRule(("猫",), "貓", True),
        ))

        handler.output("café 猫.")

        assert inner.values == ["Café 貓."]

    def test_should_match_longest_overlapping_key(self) -> None:
        handler, inner = _filter((
            LiteralRule(("C",), "cee", True),
            LiteralRule(("C++",), "cpp", True),
        ))

        handler.output("C++ and C")

        assert inner.values == ["cpp and cee"]

    def test_should_not_cascade_replacements(self) -> None:
        handler, inner = _filter((
            LiteralRule(("god",), "God", True),
            LiteralRule(("God",), "Deity", True),
        ))

        handler.output("god God")

        assert inner.values == ["God Deity"]

    def test_should_filter_streaming_deltas_with_the_same_safe_rules(self) -> None:
        inner = CapturingOutput()
        pipeline = TextPipeline.for_streaming(
            DictionaryConfig(rules=(LiteralRule(("god",), "God", True),))
        )
        writer = StreamingOutputWriter(TextFilterDecorator(inner, pipeline))

        writer.emit("god")
        writer.emit("god is good")

        assert inner.values == ["God", " is good"]


class TestLoadDictionary:
    def test_should_return_disabled_dictionary_for_malformed_json(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text("{broken", encoding="utf-8")

        assert load_dictionary(path) == DictionaryConfig()

    def test_should_migrate_valid_v1_entries_and_allow_phrases(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text(
            json.dumps({
                "": "empty key",
                "empty": "",
                "number": 2,
                "two words": "Phrase",
                "valid": "Valid",
            }),
            encoding="utf-8",
        )

        assert load_dictionary(path) == DictionaryConfig(rules=(
            LiteralRule(("two words",), "Phrase", True, True),
            LiteralRule(("valid",), "Valid", True, True),
        ))

    def test_should_preserve_v2_options_and_rules(self, tmp_path: Path) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text(json.dumps({
            "version": 2,
            "options": {
                "remove_fillers": True,
                "spoken_commands": True,
                "numbers": False,
                "punctuation_model": False,
            },
            "rules": [
                {
                    "type": "literal",
                    "spoken": ["git hub", "get hub"],
                    "replacement": "GitHub",
                    "case_sensitive": False,
                    "whole_words": True,
                    "phonetic": False,
                    "enabled": True,
                },
                {
                    "type": "regex",
                    "pattern": "\\bum+\\b",
                    "replacement": "",
                    "enabled": True,
                },
            ],
        }), encoding="utf-8")

        result = load_dictionary(path)

        assert result == DictionaryConfig(
            options=DictionaryOptions(remove_fillers=True, spoken_commands=True),
            rules=(
                LiteralRule(("git hub", "get hub"), "GitHub"),
                RegexRule(r"\bum+\b", ""),
            ),
        )

    def test_should_skip_invalid_v2_rules(self, tmp_path: Path) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text(json.dumps({
            "version": 2,
            "options": {},
            "rules": [
                {"type": "regex", "pattern": "[", "replacement": ""},
                {"type": "regex", "pattern": "ok", "replacement": r"\2"},
                {"type": "literal", "spoken": [], "replacement": "no"},
                {"type": "regex", "pattern": r"\bok\b", "replacement": "yes"},
            ],
        }), encoding="utf-8")

        assert load_dictionary(path).rules == (RegexRule(r"\bok\b", "yes"),)

    def test_should_return_disabled_dictionary_for_non_object_document(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text("[]", encoding="utf-8")

        assert load_dictionary(path) == DictionaryConfig()

    def test_should_disable_dictionary_when_v2_shape_is_malformed(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text(
            json.dumps({"version": 2, "options": [], "rules": []}),
            encoding="utf-8",
        )

        assert load_dictionary(path) == DictionaryConfig()
