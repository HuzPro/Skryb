"""The chat prompt around the audio that Qwen3-ASR transcribes, as its chat template renders it.

The system turn can carry names and terms to spell; the assistant turn can
force the language. Without a forced language the model writes
``language <Name><asr_text>`` itself, and the text after the marker is kept.
"""

from typing import Protocol, runtime_checkable

from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.result import TranscriptionResult

# Qwen3-ASR has no Urdu: spoken Urdu is asked for as Hindi, then romanized.
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi"}
_CONTROL_MARKS = ("<|", "|>")


def prompt_text(context: str | None, language: str | None) -> tuple[str, str]:
    """The text before the audio placeholders and the text after them."""
    hint = context or ""
    for mark in _CONTROL_MARKS:
        hint = hint.replace(mark, "")
    prefix = f"<|im_start|>system\n{hint}<|im_end|>\n<|im_start|>user\n<|audio_start|>"
    suffix = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
    if language:
        suffix += f"language {LANGUAGE_NAMES[language]}<asr_text>"
    return prefix, suffix


@runtime_checkable
class PromptedTranscriber(Protocol):
    """A Qwen3-ASR engine that takes names to spell and a language (a key of LANGUAGE_NAMES) with each clip."""

    def transcribe_prompted(
        self, audio: AudioSegment, context: str | None, language: str | None,
    ) -> TranscriptionResult: ...
