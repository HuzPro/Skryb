import json

from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.text_filter_decorator import (
    TextFilterDecorator,
    load_dictionary,
)
from voicetotext.transcription.streaming.streaming_output_writer import (
    StreamingOutputWriter,
)


class CapturingOutput(TextOutputHandler):
    def __init__(self) -> None:
        self.values: list[str] = []

    def output(self, text: str) -> None:
        self.values.append(text)


def _filter(rules: dict[str, str]) -> tuple[TextFilterDecorator, CapturingOutput]:
    inner = CapturingOutput()
    return TextFilterDecorator(inner, rules), inner


class TestTextFilterDecorator:
    def test_should_replace_tokens_adjacent_to_punctuation(self) -> None:
        handler, inner = _filter({"god": "God"})

        handler.output('god, (god)! "god"')

        assert inner.values == ['God, (God)! "God"']

    def test_should_apply_unicode_rules(self) -> None:
        handler, inner = _filter({"café": "Café", "猫": "貓"})

        handler.output("café 猫.")

        assert inner.values == ["Café 貓."]

    def test_should_match_longest_overlapping_key(self) -> None:
        handler, inner = _filter({"C": "cee", "C++": "cpp"})

        handler.output("C++ and C")

        assert inner.values == ["cpp and cee"]

    def test_should_not_cascade_replacements(self) -> None:
        handler, inner = _filter({"god": "God", "God": "Deity"})

        handler.output("god God")

        assert inner.values == ["God Deity"]

    def test_should_filter_streaming_deltas_with_the_same_rules(self) -> None:
        handler, inner = _filter({"god": "God"})
        writer = StreamingOutputWriter(handler)

        writer.emit("god")
        writer.emit("god is good")

        assert inner.values == ["God", " is good"]


class TestLoadDictionary:
    def test_should_return_empty_for_malformed_json(self, tmp_path) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text("{broken", encoding="utf-8")

        assert load_dictionary(path) == {}

    def test_should_skip_empty_non_string_and_phrase_entries(self, tmp_path) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text(
            json.dumps({
                "": "empty key",
                "empty": "",
                "number": 2,
                "two words": "phrase",
                "valid": "Valid",
            }),
            encoding="utf-8",
        )

        assert load_dictionary(path) == {"valid": "Valid"}

    def test_should_return_empty_for_non_object_document(self, tmp_path) -> None:
        path = tmp_path / "dictionary.json"
        path.write_text("[]", encoding="utf-8")

        assert load_dictionary(path) == {}
