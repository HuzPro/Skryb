"""Fixes made by dictating over selected words, followed until they settle and then learned."""

from dataclasses import replace
from pathlib import Path

import pytest

from voicetotext.output.caret_context import CaretContext
from voicetotext.output.dictionary_config import LiteralRule, load_dictionary
from voicetotext.writing.learning import Learner, word_corrections
from voicetotext.writing.redictation import RedictationTracker, replacement_between
from voicetotext.writing.settings import WritingSettings

_CLAUDE = 197508
_DISCORD = 40120
_MESSAGE = "I love what you did with the {} Three's ASR engine. It is fast."


class Box:
    """The focused text box, as UI Automation would read it."""

    def __init__(self, text: str, hwnd: int = _CLAUDE) -> None:
        self.text: str | None = text
        self.hwnd = hwnd
        self.reads = 0

    def read(self, limit: int) -> str | None:
        self.reads += 1
        return None if self.text is None else self.text[:limit]


class Rig:
    def __init__(self, tmp_path: Path, settings: WritingSettings | None = None) -> None:
        self.settings = settings or WritingSettings()
        self.dictionary = tmp_path / "dictionary.json"
        self.learner = Learner(
            tmp_path / "learning.json", self.dictionary, lambda: self.settings, lambda: None, "session",
        )
        self.learned_pairs: list[tuple[str, str, str]] = []
        self.box = Box(_MESSAGE.format("Quin"))
        self.now = 1000.0
        self.tracker = RedictationTracker(
            learn=self._learn,
            foreground=lambda: self.box.hwnd,
            read_text=self.box.read,
            enabled=lambda: self.settings.learn_words,
            clock=lambda: self.now,
            spawn=lambda _target: None,
        )

    def _learn(self, pasted: str, final: str, fix: str) -> None:
        self.learned_pairs.append((pasted, final, fix))
        self.learner.observe_correction(pasted, final, fix)

    def redictate(self, selected: str, said: str) -> None:
        """Select ``selected`` in the box and dictate ``said`` over it."""
        text = self.box.text or ""
        at = text.index(selected)
        self.tracker.caret_read(CaretContext(text[:at], selected, text[at + len(selected):]))
        self.tracker.pasting(said)
        self.box.text = text[:at] + said + text[at + len(selected):]
        self.tracker.poll()

    def type_over(self, old: str, new: str) -> None:
        self.box.text = (self.box.text or "").replace(old, new, 1)
        self.tracker.poll()

    def send(self) -> None:
        self.box.text = "\n"
        self.tracker.poll()

    def learned(self) -> list[tuple[tuple[str, ...], str]]:
        if not self.dictionary.exists():
            return []
        return [
            (rule.spoken, rule.replacement)
            for rule in load_dictionary(self.dictionary).rules
            if isinstance(rule, LiteralRule) and rule.learned
        ]

    def spellings(self) -> set[tuple[str, str]]:
        """The (wrong, right) word pairs handed to the learner."""
        return {
            (correction.spoken, correction.written)
            for pasted, final, _fix in self.learned_pairs
            for correction in word_corrections(pasted, final)
        }


class TestTheFixTheUserGaveUpOn:
    def test_typing_the_name_after_failed_attempts_teaches_the_close_misspellings(self, tmp_path):
        rig = Rig(tmp_path)

        rig.redictate("Quin", "Quentin")
        rig.redictate("Quentin", "Quinn")
        rig.type_over("Quinn", "Qwen")
        rig.send()

        # Every attempt goes to the learner; it keeps only spellings close
        # enough to be the same word misheard ("Quinn" is too far from "Qwen").
        assert len(rig.learned_pairs) == 3
        assert rig.spellings() == {("Quin", "Qwen"), ("Quentin", "Qwen")}

    def test_one_fix_is_not_enough_to_teach(self, tmp_path):
        rig = Rig(tmp_path)

        rig.redictate("Quin", "Quentin")
        rig.type_over("Quentin", "Qwen")
        rig.send()

        assert rig.learned() == []

    def test_two_separate_fixes_that_agree_become_a_dictionary_rule(self, tmp_path):
        rig = Rig(tmp_path)
        for _ in range(2):
            rig.box.text = _MESSAGE.format("Quin")
            rig.redictate("Quin", "Quentin")
            rig.type_over("Quentin", "Qwen")
            rig.send()

        assert (("Quin",), "Qwen") in rig.learned()
        assert (("Quentin",), "Qwen") in rig.learned()


class TestAFixMadeByVoice:
    def test_a_re_dictation_that_is_kept_teaches_the_original_spelling(self, tmp_path):
        rig = Rig(tmp_path)

        rig.redictate("Quin", "Quentin")
        rig.redictate("Quentin", "Qwen")
        rig.send()

        assert rig.spellings() == {("Quin", "Qwen"), ("Quentin", "Qwen")}

    def test_putting_the_original_back_teaches_nothing(self, tmp_path):
        rig = Rig(tmp_path)

        rig.redictate("Quin", "Quentin")
        rig.type_over("Quentin", "Quin")
        rig.send()

        assert rig.learned_pairs == []

    def test_a_re_dictation_somewhere_else_starts_its_own_fix(self, tmp_path):
        rig = Rig(tmp_path)

        rig.redictate("Quin", "Quentin")
        rig.redictate("fast", "quick")
        rig.send()

        assert rig.spellings() >= {("Quin", "Quentin")}
        assert len({fix for _pasted, _final, fix in rig.learned_pairs}) == 2


class TestWhenTheFixSettles:
    def test_it_stays_open_while_the_user_keeps_editing_the_box(self, tmp_path):
        rig = Rig(tmp_path)

        rig.redictate("Quin", "Quentin")
        rig.type_over("Quentin", "Qwe")

        assert rig.learned_pairs == []
        assert rig.tracker.poll() is True

    def test_switching_to_another_window_settles_it_with_what_was_last_read(self, tmp_path):
        rig = Rig(tmp_path)
        rig.redictate("Quin", "Quentin")
        rig.type_over("Quentin", "Qwen")

        rig.box.hwnd = _DISCORD
        rig.tracker.poll()

        assert ("Quentin", "Qwen") in rig.spellings()

    def test_it_gives_up_after_a_few_minutes_with_what_was_last_read(self, tmp_path):
        rig = Rig(tmp_path)
        rig.redictate("Quin", "Quentin")
        rig.type_over("Quentin", "Qwen")

        rig.now += 181
        rig.tracker.poll()

        assert ("Quentin", "Qwen") in rig.spellings()

    def test_a_box_that_cannot_be_read_teaches_nothing(self, tmp_path):
        rig = Rig(tmp_path)
        rig.box.text = None

        rig.tracker.caret_read(CaretContext("with the ", "Quin", " Three's"))
        rig.tracker.pasting("Quentin")
        rig.tracker.poll()

        assert rig.learned_pairs == []
        assert rig.tracker.poll() is False

    def test_rewriting_the_text_around_the_fix_teaches_nothing(self, tmp_path):
        rig = Rig(tmp_path)
        rig.redictate("Quin", "Quentin")
        rig.type_over("Quentin", "Qwen")

        rig.box.text = "Never mind, scrap all of that."

        assert rig.tracker.poll() is False
        assert rig.learned_pairs == []

    def test_deleting_the_word_teaches_nothing(self, tmp_path):
        rig = Rig(tmp_path)
        rig.redictate("Quin", "Quentin")

        rig.type_over("Quentin ", "")

        assert rig.tracker.poll() is False
        assert rig.learned_pairs == []

    def test_a_box_that_stops_answering_mid_fix_teaches_nothing(self, tmp_path):
        rig = Rig(tmp_path)
        rig.redictate("Quin", "Quentin")

        rig.box.text = None

        assert rig.tracker.poll() is False
        assert rig.learned_pairs == []


class TestWhatCountsAsAFix:
    def test_dictating_without_a_selection_is_not_a_fix(self, tmp_path):
        rig = Rig(tmp_path)

        rig.tracker.caret_read(CaretContext("with the ", "", " Three's"))
        rig.tracker.pasting("Qwen")

        assert rig.tracker.poll() is False

    def test_rephrasing_a_long_selection_is_not_a_spelling_fix(self, tmp_path):
        rig = Rig(tmp_path)

        rig.redictate("Three's ASR engine. It is", "new model, it is")

        assert rig.tracker.poll() is False

    def test_nothing_is_followed_while_learning_words_is_off(self, tmp_path):
        rig = Rig(tmp_path, replace(WritingSettings(), learn_words=False))

        rig.redictate("Quin", "Quentin")

        assert rig.tracker.poll() is False

    def test_a_dictation_with_no_caret_reading_is_never_taken_for_a_fix(self, tmp_path):
        rig = Rig(tmp_path)

        rig.tracker.pasting("Quentin")

        assert rig.tracker.poll() is False

    def test_names_with_accents_are_learned_as_written(self, tmp_path):
        rig = Rig(tmp_path)
        rig.box.text = _MESSAGE.format("Zoe")

        rig.redictate("Zoe", "Zoey")
        rig.type_over("Zoey", "Zoë")
        rig.send()

        assert rig.spellings() == {("Zoe", "Zoë"), ("Zoey", "Zoë")}

    def test_a_flood_of_fixes_keeps_only_the_newest_twenty_open(self, tmp_path):
        rig = Rig(tmp_path)
        words = [f"word{i}" for i in range(1000)]
        rig.box.text = " ".join(words)
        for word in words:
            text = rig.box.text
            at = text.index(word)
            rig.tracker.caret_read(CaretContext(text[:at], word, text[at + len(word):]))
            rig.tracker.pasting(word.upper())

        assert len(rig.tracker._fixes) == 20


class TestReplacementBetween:
    @pytest.mark.parametrize(
        ("text", "before", "after", "expected"),
        [
            ("with the Qwen Three's", "with the ", " Three's", "Qwen"),
            ("Qwen Three's", "", " Three's", "Qwen"),
            ("with the Qwen", "with the ", "", "Qwen"),
            ("the a, the b Three's", "the ", " Three's", "b"),
            ("with the Three's", "with the ", " Three's", None),
            ("with the Qwen engine", "with the ", " Three's", None),
            ("hello there", "with the ", " Three's", None),
            ("with the " + "x" * 200 + " Three's", "with the ", " Three's", None),
        ],
        ids=[
            "between both anchors",
            "at the start of the box",
            "at the end of the box",
            "the first anchor repeats earlier",
            "the words were deleted and the spaces merged",
            "the second anchor is gone",
            "both anchors are gone",
            "too much between them to be a fix",
        ],
    )
    def test_finds_what_sits_between_the_anchors(self, text, before, after, expected):
        assert replacement_between(text, before, after) == expected
