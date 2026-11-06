import pytest

from skryb.translation.llama_server import chat
from skryb.translation.translator import WORDS_PER_PIECE, Translation, Translator, has_arabic, pieces
from tests.fakes.llama_server import FakeLlamaServer

LINE = "ترى العميل يبي الفاتورة قبل الخميس."


def translator(server: FakeLlamaServer) -> Translator:
    return Translator(lambda prompt, max_tokens, grammar: chat(server.endpoint, prompt, max_tokens, grammar))


def test_an_arabic_line_comes_back_as_its_english_and_a_gloss_under_each_word() -> None:
    with FakeLlamaServer() as server:
        result = translator(server).annotate(LINE)
    assert result.english == f"EN: {LINE}"
    assert result.words == [(word, f"gloss of {word}") for word in LINE.split()]
    assert result.json() == {"english": f"EN: {LINE}", "words": [[word, f"gloss of {word}"] for word in LINE.split()]}


def test_the_gloss_call_is_one_request_with_a_grammar_and_the_translation_as_context() -> None:
    with FakeLlamaServer() as server:
        translator(server).annotate(LINE)
    translate, gloss = server.requests
    assert "grammar" not in translate and isinstance(gloss["grammar"], str)
    assert f"Translation: EN: {LINE}" in str(gloss["messages"])


@pytest.mark.parametrize("line", ["", "   ", "Send the stall photos tonight.", "12:30 - 13:45"])
def test_a_line_with_no_arabic_is_its_own_translation_and_never_reaches_the_model(line: str) -> None:
    with FakeLlamaServer() as server:
        result = translator(server).annotate(line)
    assert result == Translation(" ".join(line.split()), [(word, word) for word in line.split()])
    assert server.requests == []


def test_a_mixed_line_keeps_its_english_words_as_their_own_gloss() -> None:
    line = "خلنا نراجع الـ budget قبل الاجتماع"
    with FakeLlamaServer() as server:
        result = translator(server).annotate(line)
    assert ("budget", "budget") in result.words
    assert [word for word, _ in result.words] == line.split()


def test_a_dictation_of_thousands_of_words_is_translated_in_pieces_and_glossed_whole() -> None:
    sentence = "المحل يفتح الساعة تسع الصبح، بس الجمعة بعد الصلاة."
    text = " ".join([sentence] * 400)
    with FakeLlamaServer() as server:
        result = translator(server).annotate(text)
    assert len(result.words) == 3_600
    assert [word for word, _ in result.words] == text.split()
    translations = [request for request in server.requests if "grammar" not in request]
    assert len(translations) > 1


def test_pieces_end_at_sentence_ends_and_never_exceed_the_word_limit() -> None:
    short = "وين حطيت مفاتيح المحل؟"
    long = " ".join(["كلمة"] * (WORDS_PER_PIECE * 2 + 7))
    cut = pieces(f"{short} {long}\n{short}")
    assert all(len(piece.split()) <= WORDS_PER_PIECE for piece in cut)
    assert " ".join(cut).split() == f"{short} {long} {short}".split()
    assert cut[0].startswith(short)


def test_blank_text_has_no_pieces() -> None:
    assert pieces(" \n\t ") == []


@pytest.mark.parametrize(("text", "arabic"), [("وين؟", True), ("؟ ، ؛", False), ("OK", False), ("", False), ("١٢٣", False)])
def test_a_line_counts_as_arabic_only_with_an_arabic_letter_in_it(text: str, arabic: bool) -> None:
    assert has_arabic(text) is arabic
