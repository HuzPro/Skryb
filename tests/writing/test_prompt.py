import re

from skryb.writing.prompt import cleanup_system_prompt
from skryb.writing.settings import WritingSettings

# Words a real dictation is full of. An example built from them gets copied
# into a user's list: "Bread" turned up in a trip packing list.
_EVERYDAY_WORDS = {
    "milk", "eggs", "egg", "bread", "butter", "cheese", "apples", "bananas", "coffee",
    "tea", "sugar", "rice", "water", "juice", "snacks", "groceries", "buy", "shopping",
    "chargers", "passports", "tickets", "trip", "meeting", "meet", "call", "email",
    "report", "review", "code", "tests", "docs", "documentation", "deadline", "agenda",
    "tasks", "bug", "logs", "contract", "invoice", "lunch", "dinner", "budget", "client",
    "team", "project", "slides", "notes", "minutes", "today", "tomorrow", "tonight",
    "weekend", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
_JSON_KEYS = {"dictation", "spellings", "input", "output"}


def _example_words(prompt: str) -> set[str]:
    examples = prompt.split("Examples:\n", 1)[1]
    return set(re.findall(r"[^\W\d_]+", examples.casefold())) - _JSON_KEYS


def test_the_examples_use_no_everyday_shopping_or_agenda_words():
    prompt = cleanup_system_prompt(WritingSettings())

    words = _example_words(prompt)

    assert len(words) > 30
    assert not words & _EVERYDAY_WORDS
    assert "Never take words from the examples." in prompt
    assert "Never make lists" in prompt


def test_a_part_note_goes_after_the_shared_prompt_so_every_part_reuses_its_cache():
    settings = WritingSettings(style_notes="British spelling")
    whole = cleanup_system_prompt(settings, ("No sign-off.",), "")

    second = cleanup_system_prompt(settings, ("No sign-off.",), "", part=(2, 3))

    assert second == whole + "\nThis is part 2 of 3 of one dictation; clean only this part."
    assert cleanup_system_prompt(settings, ("No sign-off.",), "", part=(1, 1)) == whole


def test_the_rules_and_examples_are_the_same_for_every_setting_so_the_server_cache_holds():
    prefixes = {
        cleanup_system_prompt(WritingSettings(cleanup=level, lists=lists)).split("\nLevel: ")[0]
        for level in ("light", "polished")
        for lists in (True, False)
    }

    assert len(prefixes) == 1


def test_the_model_is_told_to_keep_the_speakers_words_slang_and_swearing():
    prompt = cleanup_system_prompt(WritingSettings(cleanup="polished"))

    assert "slang, swearing" in prompt
    assert "Do not fix grammar" in prompt
    assert "never replace, add or remove a word" in prompt
