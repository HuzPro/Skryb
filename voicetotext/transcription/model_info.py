import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.exceptions import ConfigurationError
from voicetotext.transcription.engine_types import EngineId, HardwareProfile


_logger = logging.getLogger(__name__)


_EMPTY_PRIORITIES: Mapping[EngineId, int] = MappingProxyType({})


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_mb: int
    description: str
    repo_id: str = ""
    compatible_engines: tuple[EngineId, ...] = ("transformers",)
    engine_priorities: Mapping[EngineId, int] = field(default_factory=lambda: _EMPTY_PRIORITIES)

    @property
    def display_size(self) -> str:
        if self.size_mb >= 1000:
            return f"{self.size_mb / 1000:.1f} GB"
        return f"{self.size_mb} MB"


# Every Whisper model runs through the transformers engine (PyTorch, every
# backend). They carry no ``engine_priorities``: with a single compatible
# engine the selector falls back to the descriptor's ``base_priority``, so
# an explicit map would be dead config.
WHISPER_MODELS: dict[str, ModelInfo] = {
    "tiny": ModelInfo(
        name="tiny", size_mb=151, description="Fastest, least accurate",
        repo_id="openai/whisper-tiny",
        compatible_engines=("transformers",),
    ),
    "base": ModelInfo(
        name="base", size_mb=290, description="Good balance",
        repo_id="openai/whisper-base",
        compatible_engines=("transformers",),
    ),
    "small": ModelInfo(
        name="small", size_mb=967, description="More accurate, slower",
        repo_id="openai/whisper-small",
        compatible_engines=("transformers",),
    ),
    "large-v3-turbo": ModelInfo(
        name="large-v3-turbo", size_mb=1620,
        description="Recommended: nearly large-v3 quality, 6-8x faster",
        repo_id="openai/whisper-large-v3-turbo",
        compatible_engines=("transformers",),
    ),
    "large-v3": ModelInfo(
        name="large-v3", size_mb=3090,
        description="Max accuracy for rare languages and noisy audio",
        repo_id="openai/whisper-large-v3",
        compatible_engines=("transformers",),
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


_NEMOTRON_STREAMING_ENGINE_PRIORITIES: Mapping[EngineId, int] = MappingProxyType(
    {"nemotron_streaming": 100},
)


NEMOTRON_STREAMING_MODELS: dict[str, ModelInfo] = {
    "nemotron-speech-streaming-en-0.6b": ModelInfo(
        name="nemotron-speech-streaming-en-0.6b",
        size_mb=2470,
        description="NVIDIA cache-aware streaming (English, low-latency, GPU)",
        repo_id="nvidia/nemotron-speech-streaming-en-0.6b",
        compatible_engines=("nemotron_streaming",),
        engine_priorities=_NEMOTRON_STREAMING_ENGINE_PRIORITIES,
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
    **WHISPER_MODELS, **PARAKEET_MODELS, **NEMOTRON_STREAMING_MODELS,
    **GRANITE_MODELS,
}


# Models kept defined for the engine layer but withheld from the
# user-facing catalog (e.g. their load is currently unstable). They cannot
# be selected; an existing config pinned to one is migrated off it (see
# ``_REMOVED_MODEL_REPLACEMENTS``). Re-enable by removing the entry here.
ALWAYS_HIDDEN_MODELS: frozenset[str] = frozenset({"granite-speech-4.1-2b"})
GPU_RETIRED_MODELS: frozenset[str] = frozenset(WHISPER_MODELS)
CPU_RETIRED_MODELS: frozenset[str] = frozenset(
    set(WHISPER_MODELS) - {"tiny", "base"}
)


def hidden_models_for(
    profile: HardwareProfile | None,
) -> frozenset[str]:
    if profile is None:
        retired = frozenset()
    elif profile.kind == "cpu":
        retired = CPU_RETIRED_MODELS
    else:
        retired = GPU_RETIRED_MODELS
    return ALWAYS_HIDDEN_MODELS | retired


def is_catalog_model(
    name: str,
    profile: HardwareProfile | None = None,
) -> bool:
    """Whether *name* should appear in the user-facing model catalog."""
    return name in ALL_MODELS and name not in hidden_models_for(profile)


def default_model_for(profile: HardwareProfile | None) -> str:
    if profile is not None and profile.kind != "cpu":
        return "parakeet-tdt-0.6b-v3"
    return "base"


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
    "nemotron-streaming-compare": "nemotron-speech-streaming-en-0.6b",
    # Withdrawn from the catalog (unstable load); move quality-seeking
    # users to the recommended high-accuracy batch model.
    "granite-speech-4.1-2b": "large-v3-turbo",
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


_REMOVED_ENGINES: frozenset[str] = frozenset({"faster_whisper"})


def migrate_engine_field(config: ConfigurationStore) -> None:
    """Normalize ``transcription.engine`` for configs from older builds.

    Legacy configs that pre-date #9 have no ``transcription.engine`` key;
    they get ``"auto"`` so the selector picks the recommended engine for
    the user's hardware. Configs that pinned the now-removed
    ``faster_whisper`` engine are reset to ``"auto"`` so startup does not
    fail building a nonexistent engine. Any other explicit value (e.g.
    set via the settings UI) is left alone.
    """
    current = config.get("transcription.engine")
    if current is None:
        _logger.info("Adding 'transcription.engine'='auto' to legacy config.")
        _save_engine(config, "auto")
        return
    if current in _REMOVED_ENGINES:
        _logger.warning(
            "Engine '%s' has been removed; switching to 'auto'.", current,
        )
        _save_engine(config, "auto")


def _save_engine(config: ConfigurationStore, value: str) -> None:
    config.set("transcription.engine", value)
    config.save()

