"""Whisper decoder prompts and transcript text, without torch or transformers.

``tokenizers`` is the Rust library Transformers already wraps, and it
reads ``tokenizer.json`` straight from the model directory. Importing it
costs ~20 ms and pulls in neither torch nor transformers, against ~4.7 s
to import ``WhisperProcessor``, so a CTranslate2 engine can build its
decoder prompt and decode what comes back for almost nothing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from skryb.core.exceptions import TranscriptionError
from skryb.transcription.backend_imports import (
    make_engine_dependency_error,
)
from skryb.transcription.engine_types import CrisperWhisperMode


if TYPE_CHECKING:
    from tokenizers import Tokenizer


MODE_TAG_COUNT = 5
_TOKENIZER_FILE = "tokenizer.json"
_START_OF_TRANSCRIPT = "<|startoftranscript|>"
_TRANSCRIBE = "<|transcribe|>"
_NO_TIMESTAMPS = "<|notimestamps|>"
_CONTEXT_START = "<ctx>"
_CONTEXT_END = "<ectx>"

# Nyra's mode tags and context markers are added tokens with
# ``special: false`` and Whisper's timestamps are ordinary vocabulary
# entries, so a decode that skips special tokens still leaves them in.
_PROMPT_ARTIFACTS = re.compile(
    r"\[(?:verbatim|intended)_\d+\]|<\|\d+\.\d+\|>|<ctx>.*?<ectx>"
)


def strip_prompt_artifacts(text: str) -> str:
    return " ".join(_PROMPT_ARTIFACTS.sub("", text).split()).strip()


def continuation_context(words: str) -> str:
    """Prompt text that asks the model to continue on from *words*.

    Nyra trained this exact spelling, spaces included, right after the
    mode tags: the model then transcribes only what follows *words*.
    """
    return f" {_CONTEXT_START} {words} {_CONTEXT_END}"


class WhisperTokens:
    """The model's tokenizer, loaded from its own ``tokenizer.json``.

    Reproduces the two things ``WhisperProcessor`` was doing for the
    Transformers adapter: the decoder prompt Nyra's model expects, and
    transcript text from generated token ids.
    """

    def __init__(self, model_dir: Path) -> None:
        self._tokenizer = _load_tokenizer(model_dir)

    def decoder_prompt(
        self,
        language: str,
        mode: CrisperWhisperMode,
        context: str = "",
    ) -> list[int]:
        """Five atomic mode tags, then Whisper's own prompt sequence.

        *language* is a Whisper language code such as ``"en"``; its id
        comes from the model's own vocabulary, so a model trained on a
        different language set cannot silently transcribe in English.
        Non-empty *context* goes between the two, see
        :func:`continuation_context`.
        """
        # ponytail: language codes only, the ten the settings page offers
        # are all codes. Accepting "english" would mean vendoring
        # Transformers' 100-entry name-to-code table.
        return self._mode_tag_ids(mode) + self._context_ids(context) + [
            self._token_id(_START_OF_TRANSCRIPT),
            self._token_id(f"<|{language.lower()}|>"),
            self._token_id(_TRANSCRIBE),
            self._token_id(_NO_TIMESTAMPS),
        ]

    def decode(self, token_ids: Sequence[int]) -> str:
        decoded = self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
        )
        return strip_prompt_artifacts(decoded)

    def _mode_tag_ids(self, mode: CrisperWhisperMode) -> list[int]:
        tags = "".join(
            f"[{mode}_{index}]"
            for index in range(1, MODE_TAG_COUNT + 1)
        )
        tag_ids = self._tokenizer.encode(tags, add_special_tokens=False).ids
        if len(tag_ids) != MODE_TAG_COUNT:
            raise TranscriptionError(
                f"Model tokenizer does not contain atomic {mode} tags"
            )
        return [int(tag_id) for tag_id in tag_ids]

    def _context_ids(self, context: str) -> list[int]:
        if not context:
            return []
        encoded = self._tokenizer.encode(
            continuation_context(context),
            add_special_tokens=False,
        )
        return [int(token_id) for token_id in encoded.ids]

    def _token_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise TranscriptionError(
                f"Model tokenizer has no {token} token"
            )
        return int(token_id)


def _load_tokenizer(model_dir: Path) -> Tokenizer:
    tokenizer_type = _import_tokenizer_type()
    tokenizer_file = model_dir / _TOKENIZER_FILE
    if not tokenizer_file.is_file():
        raise TranscriptionError(
            f"Model directory has no {_TOKENIZER_FILE}: {model_dir}"
        )
    return tokenizer_type.from_file(str(tokenizer_file))


def _import_tokenizer_type() -> type[Tokenizer]:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="tokenizers",
            install_hint=(
                "tokenizers not installed: pip install 'tokenizers>=0.23.1'"
            ),
            underlying=exc,
        ) from exc
    return Tokenizer
