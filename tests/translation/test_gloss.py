import json

import pytest

from skryb.translation.gloss import WORDS_PER_CALL, gloss_grammar, gloss_line, gloss_prompt, is_arabic, read_glosses
from tests.fakes.llama_server import words_in_grammar

LINE = "وين حطيت مفاتيح المحل؟"


class Model:
    """Answers each gloss call with every word the grammar allows, glossed ``g:<word>``; records the calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None]] = []

    def __call__(self, prompt: str, max_tokens: int, grammar: str | None) -> str:
        self.calls.append((prompt, max_tokens, grammar))
        assert grammar is not None
        return json.dumps([{"w": word, "g": f"g:{word}"} for word in words_in_grammar(grammar)], ensure_ascii=False)


def test_every_written_arabic_word_gets_one_gloss_in_the_order_written() -> None:
    model = Model()
    assert gloss_line(LINE, model) == [(word, f"g:{word}") for word in LINE.split()]
    assert len(model.calls) == 1


def test_a_word_said_twice_is_glossed_twice_in_place() -> None:
    line = "زين زين، خلاص زين"
    assert [word for word, _ in gloss_line(line, Model())] == ["زين", "زين،", "خلاص", "زين"]


def test_english_words_numbers_and_punctuation_in_an_arabic_line_are_their_own_gloss_and_never_sent() -> None:
    model = Model()
    pairs = gloss_line("خلنا نراجع الـ budget قبل 3:30 ، OK؟", model)
    assert ("budget", "budget") in pairs and ("3:30", "3:30") in pairs and ("،", "،") in pairs and ("OK؟", "OK؟") in pairs
    grammar = model.calls[0][2]
    assert grammar is not None and words_in_grammar(grammar) == ["خلنا", "نراجع", "الـ", "قبل"]


@pytest.mark.parametrize("line", ["", "   ", "\n\t", "Send the invoice today", "12 / 30 - 45"])
def test_a_line_with_no_arabic_letter_needs_no_model(line: str) -> None:
    model = Model()
    assert gloss_line(line, model) == [(word, word) for word in line.split()]
    assert model.calls == []


def test_a_line_of_thousands_of_words_is_glossed_in_calls_small_enough_for_the_context() -> None:
    words = [f"كلمة{index}" for index in range(3_000)]
    model = Model()
    pairs = gloss_line(" ".join(words), model)
    assert [word for word, _ in pairs] == words
    assert [gloss for _, gloss in pairs] == [f"g:{word}" for word in words]
    assert len(model.calls) == -(-3_000 // WORDS_PER_CALL)
    assert all(len(words_in_grammar(grammar or "")) <= WORDS_PER_CALL for _, _, grammar in model.calls)


def test_the_translation_is_given_as_context_for_a_line_glossed_in_one_call() -> None:
    model = Model()
    gloss_line(LINE, model, "Where did you put the shop keys?")
    assert "Translation: Where did you put the shop keys?" in model.calls[0][0]


def test_quotes_and_backslashes_inside_a_word_survive_the_grammar() -> None:
    words = ['قال"زين"', "مسار\\ملف", "عادي"]
    assert words_in_grammar(gloss_grammar(words)) == words


def test_the_grammar_leaves_only_english_glosses_open() -> None:
    gloss_rule = gloss_grammar(["زين"]).splitlines()[-1]
    assert gloss_rule.startswith("gloss ::= ")
    assert "\\u0600-\\u06ff" in gloss_rule and '"' in gloss_rule and "\\\\" in gloss_rule


def test_the_prompt_asks_for_prefixes_and_suffixes_folded_into_each_gloss() -> None:
    prompt = gloss_prompt(LINE, LINE.split())
    assert "prefixes and suffixes" in prompt and "and their house" in prompt and f"Sentence: {LINE}" in prompt


@pytest.mark.parametrize(
    "answer",
    [
        '[{"w":"وين","g":"where"}]',  # a word skipped
        '[{"w":"حطيت","g":"put"},{"w":"وين","g":"where"}]',  # reordered
        '[{"w":"وين حطيت","g":"where put"}]',  # merged
        '{"w":"وين","g":"where"}',  # not a list
        '[{"w":"وين"},{"w":"حطيت","g":"put"}]',  # a gloss missing
    ],
)
def test_an_answer_that_does_not_hold_exactly_the_words_is_refused(answer: str) -> None:
    with pytest.raises(ValueError):
        read_glosses(answer, ["وين", "حطيت"])


def test_whitespace_inside_a_gloss_is_tidied() -> None:
    assert read_glosses('[{"w":"وين","g":"  where   now "}]', ["وين"]) == ["where now"]


@pytest.mark.parametrize(("word", "arabic"), [("وين", True), ("؟", False), ("،", False), ("١٢", False), ("budget", False), ("الـiPhone", True), ("ـ", False)])
def test_only_words_with_an_arabic_letter_count_as_arabic(word: str, arabic: bool) -> None:
    assert is_arabic(word) is arabic
