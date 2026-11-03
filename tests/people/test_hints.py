from collections.abc import Sequence

import numpy as np
import pytest

from skryb.people.hints import MIN_HINT_SECONDS, echoes_hint, hint_text, transcribe_with_hint, transcribe_with_terms
from skryb.people.terms import Term
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.result import TranscriptionResult

_RATE = 16000
_TERMS = (Term("Skryb", note="the app I'm building"), Term("Ngozi Adébáyọ̀"), Term("QuoteDesk"))
_HINT = "Names and terms that may come up: Skryb (the app I'm building), Ngozi Adébáyọ̀, QuoteDesk."


def _clip(seconds: float) -> AudioSegment:
    return AudioSegment(np.full(int(seconds * _RATE), 0.01, dtype=np.float32), _RATE)


def _result(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, language="en", language_probability=0.0, transcription_seconds=0.1)


class FakeQwen(TranscriptionEngine):
    """Answers with the hint read back when told to, else with what was said."""

    def __init__(self, said: str = "Skryb pasted it.", echo: str = "") -> None:
        self.said, self.echo = said, echo
        self.calls: list[tuple[float, str | None, str | None]] = []

    def load_model(self) -> None:
        return None

    def is_model_loaded(self) -> bool:
        return True

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        raise AssertionError("Qwen3-ASR is always asked with its prompt")

    def transcribe_prompted(self, audio: AudioSegment, context: str | None, language: str | None) -> TranscriptionResult:
        self.calls.append((round(audio.duration_seconds, 2), context, language))
        return _result(self.echo if context and self.echo else self.said)

    def dispose(self) -> None:
        return None


class FakePhonon(TranscriptionEngine):
    def __init__(self) -> None:
        self.hotwords: list[list[str]] = []

    def load_model(self) -> None:
        return None

    def is_model_loaded(self) -> bool:
        return True

    def set_hotwords(self, words: Sequence[str]) -> None:
        self.hotwords.append(list(words))

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        return _result("Ngozi said hi.")

    def dispose(self) -> None:
        return None


def test_the_hint_lists_every_entry_with_its_note_then_new_speaker_names() -> None:
    names = ["", "  ", "ngozi adébáyọ̀", "Omar Haddad", "Omar Haddad", "نادیہ رحمان"]
    assert hint_text(_TERMS, names) == (
        "Names and terms that may come up: Skryb (the app I'm building), Ngozi Adébáyọ̀, QuoteDesk, Omar Haddad, نادیہ رحمان."
    )


def test_an_empty_list_and_no_names_give_no_hint() -> None:
    assert hint_text((), ["", " "]) is None


@pytest.mark.parametrize("text", [
    _HINT,
    "Lorem and terms that may come up and terms that may come up.",
    "Skryb (the app I'm building), Ngozi Adébáyọ̀, QuoteDesk.",
])
def test_a_transcript_that_reads_the_hint_back_is_an_echo(text: str) -> None:
    assert echoes_hint(text, _HINT)


@pytest.mark.parametrize("text", [
    "Problems that may come up later are mine.",
    "Ngozi Adébáyọ̀, QuoteDesk is broken again.",
    "Skryb pasted it into QuoteDesk for Ngozi Adébáyọ̀.",
    "",
])
def test_ordinary_speech_naming_entries_is_not_an_echo(text: str) -> None:
    assert not echoes_hint(text, _HINT)


def test_a_clip_shorter_than_the_threshold_is_decoded_without_the_hint() -> None:
    engine = FakeQwen()

    transcribe_with_hint(engine, _clip(MIN_HINT_SECONDS - 0.01), _HINT, "en")
    transcribe_with_hint(engine, _clip(MIN_HINT_SECONDS), _HINT, "en")

    assert engine.calls == [(MIN_HINT_SECONDS - 0.01, None, "en"), (MIN_HINT_SECONDS, _HINT, "en")]


def test_a_transcript_that_echoes_the_hint_is_decoded_again_without_it() -> None:
    engine = FakeQwen(said="Ship it on Friday.", echo=_HINT)

    result = transcribe_with_hint(engine, _clip(3.3), _HINT, "en")

    assert result.text == "Ship it on Friday."
    assert engine.calls == [(3.3, _HINT, "en"), (3.3, None, "en")]


def test_dictation_hands_qwen_the_list_in_the_language_set() -> None:
    engine = FakeQwen()

    result = transcribe_with_terms(engine, _clip(4.0), _TERMS, "hi")

    assert result.text == "Skryb pasted it."
    assert engine.calls == [(4.0, _HINT, "hi")]


def test_dictation_with_an_empty_list_asks_qwen_without_a_hint() -> None:
    engine = FakeQwen()
    transcribe_with_terms(engine, _clip(4.0), (), "en")
    assert engine.calls == [(4.0, None, "en")]


def test_phonon_favours_the_first_twenty_five_entries() -> None:
    engine = FakePhonon()
    terms = tuple(Term(f"Name{index}") for index in range(1000))

    result = transcribe_with_terms(engine, _clip(1.0), terms, "en")

    assert result.text == "Ngozi said hi."
    assert engine.hotwords == [[f"Name{index}" for index in range(25)]]
