import logging
from dataclasses import dataclass

from voicetotext.core.config import ConfigurationStore


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_mb: int
    description: str
    repo_id: str = ""

    @property
    def display_size(self) -> str:
        if self.size_mb >= 1000:
            return f"{self.size_mb / 1000:.1f} GB"
        return f"{self.size_mb} MB"


TRANSFORMERS_WHISPER_MODELS: dict[str, ModelInfo] = {
    "tiny": ModelInfo(
        name="tiny", size_mb=151, description="Fastest, least accurate",
        repo_id="openai/whisper-tiny",
    ),
    "base": ModelInfo(
        name="base", size_mb=290, description="Good balance",
        repo_id="openai/whisper-base",
    ),
    "small": ModelInfo(
        name="small", size_mb=967, description="More accurate, slower",
        repo_id="openai/whisper-small",
    ),
    "large-v3-turbo": ModelInfo(
        name="large-v3-turbo", size_mb=1620,
        description="Recommended: nearly large-v3 quality, 6-8x faster",
        repo_id="openai/whisper-large-v3-turbo",
    ),
    "large-v3": ModelInfo(
        name="large-v3", size_mb=3090,
        description="Max accuracy for rare languages and noisy audio",
        repo_id="openai/whisper-large-v3",
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

