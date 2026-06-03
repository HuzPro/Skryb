import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.exceptions import ConfigurationError
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


_PARAKEET_ENGINE_PRIORITIES: Mapping[EngineId, int] = MappingProxyType(
    {"nano_parakeet": 100},
)


PARAKEET_MODELS: dict[str, ModelInfo] = {
    "parakeet-tdt-0.6b-v3": ModelInfo(
        name="parakeet-tdt-0.6b-v3",
        size_mb=2510,
        description="Best accuracy, built-in punctuation (25 languages)",
        repo_id="nvidia/parakeet-tdt-0.6b-v3",
        compatible_engines=("nano_parakeet",),
        engine_priorities=_PARAKEET_ENGINE_PRIORITIES,
    ),
}


_GRANITE_ENGINE_PRIORITIES: Mapping[EngineId, int] = MappingProxyType(
    {"granite_speech": 100},
)


GRANITE_MODELS: dict[str, ModelInfo] = {
    "granite-speech-4.1-2b": ModelInfo(
        name="granite-speech-4.1-2b",
        size_mb=4500,
        description="IBM Granite Speech: top OpenASR (English, WER 5.33)",
        repo_id="ibm-granite/granite-speech-4.1-2b",
        compatible_engines=("granite_speech",),
        engine_priorities=_GRANITE_ENGINE_PRIORITIES,
    ),
}


ALL_MODELS: dict[str, ModelInfo] = {
    **WHISPER_MODELS, **PARAKEET_MODELS, **GRANITE_MODELS,
}


def get_model_info(name: str) -> ModelInfo:
    info = ALL_MODELS.get(name)
    if info is None:
        raise ConfigurationError(
            f"Unknown model '{name}'. "
            f"Available: {', '.join(sorted(ALL_MODELS))}"
        )
    return info


_DEVICE_SELECTABLE_MODELS: frozenset[str] = frozenset({"tiny", "base"})


def allows_device_selection(name: str) -> bool:
    """Whether the user may pick a compute device for model *name*.

    Only the two smallest Whisper models expose the device picker. Larger
    models are pinned to ``auto`` (the GPU on accelerated hardware) so a
    tester cannot accidentally run a multi-gigabyte model on the CPU and
    trigger a multi-minute transcription or an out-of-memory error.
    """
    return name in _DEVICE_SELECTABLE_MODELS


_REMOVED_MODEL_REPLACEMENTS: dict[str, str] = {
    "medium": "large-v3-turbo",
    "parakeet-tdt-0.6b-v2": "parakeet-tdt-0.6b-v3",
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

