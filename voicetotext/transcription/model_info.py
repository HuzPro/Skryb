import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from voicetotext.core.config import ConfigurationStore
from voicetotext.transcription.engine_types import EngineId


_logger = logging.getLogger(__name__)


_EMPTY_PRIORITIES: Mapping[EngineId, int] = MappingProxyType({})


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_mb: int
    description: str
    repo_id: str = ""
    ct2_repo_id: str = ""
    compatible_engines: tuple[EngineId, ...] = ("transformers",)
    engine_priorities: Mapping[EngineId, int] = field(default_factory=lambda: _EMPTY_PRIORITIES)

    @property
    def display_size(self) -> str:
        if self.size_mb >= 1000:
            return f"{self.size_mb / 1000:.1f} GB"
        return f"{self.size_mb} MB"


_WHISPER_ENGINE_PRIORITIES: Mapping[EngineId, int] = MappingProxyType(
    {"faster_whisper": 100, "transformers": 50},
)


WHISPER_MODELS: dict[str, ModelInfo] = {
    "tiny": ModelInfo(
        name="tiny", size_mb=151, description="Fastest, least accurate",
        repo_id="openai/whisper-tiny",
        ct2_repo_id="Systran/faster-whisper-tiny",
        compatible_engines=("transformers", "faster_whisper"),
        engine_priorities=_WHISPER_ENGINE_PRIORITIES,
    ),
    "base": ModelInfo(
        name="base", size_mb=290, description="Good balance",
        repo_id="openai/whisper-base",
        ct2_repo_id="Systran/faster-whisper-base",
        compatible_engines=("transformers", "faster_whisper"),
        engine_priorities=_WHISPER_ENGINE_PRIORITIES,
    ),
    "small": ModelInfo(
        name="small", size_mb=967, description="More accurate, slower",
        repo_id="openai/whisper-small",
        ct2_repo_id="Systran/faster-whisper-small",
        compatible_engines=("transformers", "faster_whisper"),
        engine_priorities=_WHISPER_ENGINE_PRIORITIES,
    ),
    "large-v3-turbo": ModelInfo(
        name="large-v3-turbo", size_mb=1620,
        description="Recommended: nearly large-v3 quality, 6-8x faster",
        repo_id="openai/whisper-large-v3-turbo",
        ct2_repo_id="Systran/faster-whisper-large-v3-turbo",
        compatible_engines=("transformers", "faster_whisper"),
        engine_priorities=_WHISPER_ENGINE_PRIORITIES,
    ),
    "large-v3": ModelInfo(
        name="large-v3", size_mb=3090,
        description="Max accuracy for rare languages and noisy audio",
        repo_id="openai/whisper-large-v3",
        ct2_repo_id="Systran/faster-whisper-large-v3",
        compatible_engines=("transformers", "faster_whisper"),
        engine_priorities=_WHISPER_ENGINE_PRIORITIES,
    ),
}


_REMOVED_MODEL_REPLACEMENTS: dict[str, str] = {
    "medium": "large-v3-turbo",
}


def migrate_removed_models(config: ConfigurationStore) -> None:
    current = config.get("transcription.model_size")
    replacement = _REMOVED_MODEL_REPLACEMENTS.get(current)
    if replacement is None:
        return
    _logger.warning(
        "Model '%s' has been removed; switching to '%s'.", current, replacement
    )
    config.set("transcription.model_size", replacement)
    config.save()


def migrate_engine_field(config: ConfigurationStore) -> None:
    """Add ``transcription.engine`` to legacy configs that pre-date #9.

    Defaults to ``"auto"`` so the engine selector picks the recommended
    engine for the user's hardware on next startup. Existing explicit
    values (e.g. set via the settings UI) are left alone.
    """
    if config.get("transcription.engine") is not None:
        return
    _logger.info("Adding 'transcription.engine'='auto' to legacy config.")
    config.set("transcription.engine", "auto")
    config.save()

