"""Hardware-independent tests for the torch-free Whisper token path.

The tokenizer here is a miniature stand-in built with the same
``tokenizers`` library the real model ships: a byte-level BPE base, five
genuinely special Whisper control tokens, and the ten mode tags as
NON-special added tokens. That last property is the whole reason
``skip_special_tokens`` is not enough on its own.
"""

from pathlib import Path

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.transcription.whisper_tokens import (
    MODE_TAG_COUNT,
    WhisperTokens,
    strip_prompt_artifacts,
)


pytestmark = pytest.mark.integration

_CONTROL_TOKENS = [
    "<|startoftranscript|>",
    "<|en|>",
    "<|fr|>",
    "<|transcribe|>",
    "<|notimestamps|>",
]
_TIMESTAMP_TOKEN = "<|0.00|>"
_CONTEXT_MARKERS = ["<ctx>", "<ectx>"]
_ALPHABET = sorted(pre_tokenizers.ByteLevel.alphabet())
_START_OF_TRANSCRIPT_ID = len(_ALPHABET)
_ENGLISH_ID = _START_OF_TRANSCRIPT_ID + 1
_FRENCH_ID = _START_OF_TRANSCRIPT_ID + 2
_TRANSCRIBE_ID = _START_OF_TRANSCRIPT_ID + 3
_NO_TIMESTAMPS_ID = _START_OF_TRANSCRIPT_ID + 4
_FIRST_INTENDED_ID = _NO_TIMESTAMPS_ID + 1
_FIRST_VERBATIM_ID = _FIRST_INTENDED_ID + MODE_TAG_COUNT
_TIMESTAMP_ID = _FIRST_VERBATIM_ID + MODE_TAG_COUNT
_CONTEXT_START_ID = _TIMESTAMP_ID + 1
_CONTEXT_END_ID = _TIMESTAMP_ID + 2


def _mode_tags(mode: str) -> list[str]:
    return [f"[{mode}_{index}]" for index in range(1, MODE_TAG_COUNT + 1)]


def _write_tokenizer(directory: Path, *, modes: tuple[str, ...]) -> Path:
    tokenizer = Tokenizer(
        models.BPE(
            vocab={char: index for index, char in enumerate(_ALPHABET)},
            merges=[],
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([
        AddedToken(token, normalized=False, special=True)
        for token in _CONTROL_TOKENS
    ])
    tokenizer.add_tokens([
        AddedToken(tag, normalized=True, special=False)
        for mode in ("intended", "verbatim")
        for tag in (_mode_tags(mode) if mode in modes else [])
    ] + [
        AddedToken(token, normalized=True, special=False)
        for token in [_TIMESTAMP_TOKEN, *_CONTEXT_MARKERS]
    ])
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))
    return directory


def _text_ids(model_dir: Path, text: str) -> list[int]:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return tokenizer.encode(text, add_special_tokens=False).ids


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return _write_tokenizer(tmp_path / "model", modes=("intended", "verbatim"))


class TestDecoderPrompt:
    def test_puts_five_mode_tags_before_the_whisper_sequence(
        self,
        model_dir: Path,
    ) -> None:
        assert WhisperTokens(model_dir).decoder_prompt("en", "intended") == [
            *range(_FIRST_INTENDED_ID, _FIRST_INTENDED_ID + MODE_TAG_COUNT),
            _START_OF_TRANSCRIPT_ID,
            _ENGLISH_ID,
            _TRANSCRIBE_ID,
            _NO_TIMESTAMPS_ID,
        ]

    def test_intended_and_verbatim_differ_only_in_the_mode_tags(
        self,
        model_dir: Path,
    ) -> None:
        tokens = WhisperTokens(model_dir)

        intended = tokens.decoder_prompt("en", "intended")
        verbatim = tokens.decoder_prompt("en", "verbatim")

        assert intended[MODE_TAG_COUNT:] == verbatim[MODE_TAG_COUNT:]
        assert verbatim[:MODE_TAG_COUNT] == list(
            range(_FIRST_VERBATIM_ID, _FIRST_VERBATIM_ID + MODE_TAG_COUNT)
        )

    def test_takes_the_language_id_from_the_model_vocabulary(
        self,
        model_dir: Path,
    ) -> None:
        tokens = WhisperTokens(model_dir)
        language_position = MODE_TAG_COUNT + 1

        french = tokens.decoder_prompt("fr", "intended")
        english = tokens.decoder_prompt("en", "intended")

        assert french[language_position] == _FRENCH_ID
        assert english[language_position] == _ENGLISH_ID

    def test_wraps_the_context_in_markers_between_the_tags_and_the_whisper_sequence(
        self,
        model_dir: Path,
    ) -> None:
        tokens = WhisperTokens(model_dir)
        plain = tokens.decoder_prompt("en", "intended")

        prompt = tokens.decoder_prompt("en", "intended", "at least the")

        context = prompt[MODE_TAG_COUNT:-4]
        assert prompt[:MODE_TAG_COUNT] == plain[:MODE_TAG_COUNT]
        assert prompt[-4:] == plain[MODE_TAG_COUNT:]
        assert context[1] == _CONTEXT_START_ID
        assert context[-1] == _CONTEXT_END_ID
        assert tokens.decode(context[2:-1]).strip() == "at least the"

    def test_rejects_a_language_the_model_was_not_trained_on(
        self,
        model_dir: Path,
    ) -> None:
        with pytest.raises(TranscriptionError, match=r"no <\|de\|> token"):
            WhisperTokens(model_dir).decoder_prompt("de", "intended")

    def test_rejects_a_tokenizer_without_atomic_mode_tags(
        self,
        tmp_path: Path,
    ) -> None:
        verbatim_only = _write_tokenizer(
            tmp_path / "verbatim-only",
            modes=("verbatim",),
        )

        with pytest.raises(
            TranscriptionError,
            match="does not contain atomic intended tags",
        ):
            WhisperTokens(verbatim_only).decoder_prompt("en", "intended")


class TestDecode:
    def test_turns_generated_ids_into_text(self, model_dir: Path) -> None:
        ids = _text_ids(model_dir, "Testing 1, 2, 3.")

        assert WhisperTokens(model_dir).decode(ids) == "Testing 1, 2, 3."

    def test_removes_mode_tags_that_are_not_special_tokens(
        self,
        model_dir: Path,
    ) -> None:
        spoken = _text_ids(model_dir, "Testing 1, 2, 3.")
        tagged = [_FIRST_INTENDED_ID, *spoken]
        survives_skip_special = Tokenizer.from_file(
            str(model_dir / "tokenizer.json")
        ).decode(tagged, skip_special_tokens=True)

        assert survives_skip_special == "[intended_1]Testing 1, 2, 3."
        assert WhisperTokens(model_dir).decode(tagged) == "Testing 1, 2, 3."

    def test_removes_an_echoed_decoder_prompt(self, model_dir: Path) -> None:
        tokens = WhisperTokens(model_dir)
        echoed = tokens.decoder_prompt("en", "verbatim") + _text_ids(
            model_dir, "Hello."
        )

        assert tokens.decode(echoed) == "Hello."

    def test_removes_timestamp_tokens(self, model_dir: Path) -> None:
        timestamped = [_TIMESTAMP_ID, *_text_ids(model_dir, "Hello.")]

        assert WhisperTokens(model_dir).decode(timestamped) == "Hello."


class TestLoading:
    def test_rejects_a_model_directory_without_a_tokenizer(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(TranscriptionError, match="no tokenizer.json"):
            WhisperTokens(tmp_path)


class TestStripPromptArtifacts:
    def test_collapses_the_whitespace_left_behind(self) -> None:
        artifacts = "  [verbatim_3] one   two\n[intended_1] three "

        assert strip_prompt_artifacts(artifacts) == "one two three"

    def test_removes_an_echoed_continuation_context(self) -> None:
        echoed = "<ctx> at least the <ectx> middle of next week."

        assert strip_prompt_artifacts(echoed) == "middle of next week."

    def test_keeps_the_disfluency_tags_the_model_emits(self) -> None:
        assert strip_prompt_artifacts("[UM] see [noise]") == "[UM] see [noise]"
