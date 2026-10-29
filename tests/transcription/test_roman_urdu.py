"""Spoken Urdu, which Qwen3-ASR writes in Hindi script, turned into Roman Urdu."""

import time

import pytest

from skryb.transcription.roman_urdu import romanize, urdu_spelling


@pytest.mark.parametrize("hindi, roman", [
    ("नहीं", "nahin"),
    ("करते", "karte"),
    ("उसके अंदर", "uske andar"),
    ("समझते", "samajhte"),
    ("हम", "ham"),
    ("लोगों", "logon"),
    ("दोस्तों", "doston"),
    ("कपड़े", "kapre"),
    ("ज़्यादा", "zyada"),
    ("काम", "kaam"),
    ("आप", "aap"),
    ("हमारे साथ", "hamare saath"),
    ("साफ़", "saaf"),
    ("का", "ka"),
    ("था", "tha"),
    ("क्या", "kya"),
    ("आ रहे", "aa rahe"),
    ("पांच", "paanch"),
])
def test_common_urdu_words_come_out_the_way_they_are_usually_typed(hindi: str, roman: str) -> None:
    assert romanize(hindi) == roman


def test_english_words_inside_a_mixed_line_are_left_alone() -> None:
    line = "Slack से निकाल रहे हैं, Slack में डाल रहे हैं।"
    assert romanize(line) == "Slack se nikal rahe hain, Slack men daal rahe hain."


def test_lines_without_hindi_script_are_returned_unchanged() -> None:
    for line in ["Launch Board first.", "", "ٹھیک ہے", "Price is $12 ✓", "  spaced  "]:
        assert romanize(line) == line


def test_hindi_digits_become_ascii_digits() -> None:
    assert romanize("२० लोग") == "20 log"


def test_precomposed_and_combining_nukta_letters_read_the_same() -> None:
    precomposed, combining = "ज़्यादा", "ज़्यादा"
    assert romanize(precomposed) == romanize(combining) == "zyada"


def test_a_long_meeting_worth_of_mixed_lines_converts_quickly() -> None:
    line = "अब जो market खोल रही है software engineering की market में से लोग fire कर रहे हैं"
    started = time.monotonic()
    out = [romanize(line) for _ in range(5000)]
    assert time.monotonic() - started < 5
    assert out[0] == "ab jo market khol rahi hai software engineering ki market men se log fire kar rahe hain"


@pytest.mark.parametrize("hinglish, urdu", [
    ("vah kah rahe the", "woh kah rahe the"),
    ("Vah sahi hai", "Woh sahi hai"),
    ("yah kyonki ham vahaan the", "yeh kyunke hum wahan the"),
    ("mainne kuchh nahin kaha", "maine kuch nahin kaha"),
    ("bachchon ko zyaada vakt chaahie", "bachon ko zyada waqt chahiye"),
    ("Achchha, thik hai.", "Acha, theek hai."),
    ("unhonne pahle kaha", "unhon ne pehle kaha"),
])
def test_the_commonest_hindi_spellings_are_respelled_the_urdu_way(hinglish: str, urdu: str) -> None:
    assert urdu_spelling(hinglish) == urdu


@pytest.mark.parametrize("text", [
    "The van was vacant, so we took the vote.",  # v-words that aren't Urdu
    "a yahoo account, the vahana and thikness",  # only whole words are respelled
    "woh yeh hum wahan",  # already Urdu spelling
    "",
    "Café résumé naïve Zoë",
    "وہاں پر کیا ہو رہا ہے",
])
def test_text_without_hindi_spellings_comes_back_unchanged(text: str) -> None:
    assert urdu_spelling(text) == text


def test_respelling_a_long_meeting_transcript_stays_fast() -> None:
    text = "vah kah rahe the ki yah free text box basically kaam nahin karta. " * 2000
    started = time.monotonic()

    out = urdu_spelling(text)

    assert out.count("woh") == 2000 and "vah" not in out
    assert time.monotonic() - started < 1.0
