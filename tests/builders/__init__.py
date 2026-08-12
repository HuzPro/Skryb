"""Fluent Test Data Builders for VoiceToText domain objects.

Each builder exposes a module-level constructor that reads as a
noun-phrase - ``an_audio_segment()``, ``a_transcription_result()`` -
so test setup reads as a specification of the data, not a bag of
raw keyword arguments.
"""

from tests.builders.audio import AudioSegmentBuilder, an_audio_segment
from tests.builders.config import a_config
from tests.builders.models import ModelInfoBuilder, a_model_info
from tests.builders.transcription import (
    TranscriptionResultBuilder,
    a_transcription_result,
)

__all__ = [
    "AudioSegmentBuilder",
    "ModelInfoBuilder",
    "TranscriptionResultBuilder",
    "a_config",
    "a_model_info",
    "a_transcription_result",
    "an_audio_segment",
]
