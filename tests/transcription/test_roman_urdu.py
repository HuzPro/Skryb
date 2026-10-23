"""Spoken Urdu, which Qwen3-ASR writes in Hindi script, turned into Roman Urdu."""

import time

import pytest

from voicetotext.transcription.roman_urdu import romanize


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
    for line in ["Mission Control first.", "", "ٹھیک ہے", "Price is $12 ✓", "  spaced  "]:
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
