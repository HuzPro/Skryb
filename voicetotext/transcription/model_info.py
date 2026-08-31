import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.exceptions import ConfigurationError
from voicetotext.transcription.engine_types import EngineId

_logger = logging.getLogger(__name__)


_EMPTY_PRIORITIES: Mapping[EngineId, int] = MappingProxyType({})
DEFAULT_MODEL = "parakeet-tdt-0.6b-v3"


@dataclass(frozen=True)
class ModelInfo:
    """Catalog entry for one downloadable model.

    ``revision`` is the single source of truth for the commit both the
    downloader fetches and the loader opens. An empty ``revision`` tracks
    the repo's ``main`` branch, which is what every unpinned entry wants.

    ``license_id`` names the licence a user must accept before this
    model may be downloaded, built, or loaded; an empty ``license_id``
    means ungated. It also names the shipped text
    (``voicetotext/licenses/<license_id>.md``) the acceptance is keyed
    to, so entries sharing one ``license_id`` share one acceptance.
    ``license_url`` points at the upstream copy of that text at the
    pinned revision, never at a moving branch.
    """
    name: str
    size_mb: int
    description: str
    repo_id: str = ""
    revision: str = ""
    license_id: str = ""
    license_url: str = ""
    compatible_engines: tuple[EngineId, ...] = ()
    engine_priorities: Mapping[EngineId, int] = field(default_factory=lambda: _EMPTY_PRIORITIES)

    @property
    def display_size(self) -> str:
        if self.size_mb >= 1000:
            return f"{self.size_mb / 1000:.1f} GB"
        return f"{self.size_mb} MB"


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


_CRISPERWHISPER_ENGINE_PRIORITIES: Mapping[EngineId, int] = MappingProxyType(
    {"crisperwhisper": 40},
)

# One licence covers both variants: the LICENSE.md bytes at the two
# pinned commits are byte-identical, so a single acceptance is correct
# rather than a convenient guess.
CRISPERWHISPER_LICENSE_ID = "crisperwhisper-nc"


CRISPERWHISPER_MODELS: dict[str, ModelInfo] = {
    "crisperwhisper-2.0-turbo": ModelInfo(
        name="crisperwhisper-2.0-turbo",
        size_mb=1630,
        description=(
            "Fast 0.8B intended/verbatim Whisper; "
            "weights/outputs under non-commercial research license"
        ),
        repo_id="nyralabs/CrisperWhisper2.0_turbo",
        revision="de0369c8a68025b7f6e86387b6eb5a3b369787c8",
        license_id=CRISPERWHISPER_LICENSE_ID,
        license_url=(
            "https://huggingface.co/nyralabs/CrisperWhisper2.0_turbo/blob/"
            "de0369c8a68025b7f6e86387b6eb5a3b369787c8/LICENSE.md"
        ),
        compatible_engines=("crisperwhisper",),
        engine_priorities=_CRISPERWHISPER_ENGINE_PRIORITIES,
    ),
    "crisperwhisper-2.0-large": ModelInfo(
        name="crisperwhisper-2.0-large",
        size_mb=3100,
        description=(
            "Accurate 1.55B intended/verbatim Whisper; "
            "weights/outputs under non-commercial research license"
        ),
        repo_id="nyralabs/CrisperWhisper2.0_large",
        revision="f4334f6e8193f2691212d49b20fa12d370e13896",
        license_id=CRISPERWHISPER_LICENSE_ID,
        license_url=(
            "https://huggingface.co/nyralabs/CrisperWhisper2.0_large/blob/"
            "f4334f6e8193f2691212d49b20fa12d370e13896/LICENSE.md"
        ),
        compatible_engines=("crisperwhisper",),
        engine_priorities=_CRISPERWHISPER_ENGINE_PRIORITIES,
    ),
}


ALL_MODELS: dict[str, ModelInfo] = {
    **PARAKEET_MODELS, **NEMOTRON_STREAMING_MODELS, **CRISPERWHISPER_MODELS,
}


def is_catalog_model(name: str) -> bool:
    """Whether *name* should appear in the user-facing model catalog."""
    return name in ALL_MODELS


def get_model_info(name: str) -> ModelInfo:
    info = ALL_MODELS.get(name)
    if info is None:
        raise ConfigurationError(
            f"Unknown model '{name}'. "
            f"Available: {', '.join(sorted(ALL_MODELS))}"
        )
    return info


_REMOVED_MODEL_REPLACEMENTS: dict[str, str] = {
    "tiny": DEFAULT_MODEL,
    "base": DEFAULT_MODEL,
    "small": DEFAULT_MODEL,
    "medium": DEFAULT_MODEL,
    "large-v3": DEFAULT_MODEL,
    "large-v3-turbo": DEFAULT_MODEL,
    "granite-speech-4.1-2b": DEFAULT_MODEL,
    "granite-speech-5.0-470m-turboctc": DEFAULT_MODEL,
    "ark-asr-3b": DEFAULT_MODEL,
    "qwen3-asr-1.7b": DEFAULT_MODEL,
    "parakeet-tdt-0.6b-v2": DEFAULT_MODEL,
    "nemotron-streaming-compare": "nemotron-speech-streaming-en-0.6b",
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


_REMOVED_ENGINES: frozenset[str] = frozenset({
    "ark_asr",
    "faster_whisper",
    "granite_speech",
    "granite_turboctc",
    "qwen3_asr",
    "transformers",
})


def migrate_engine_field(config: ConfigurationStore) -> None:
    """Normalize ``transcription.engine`` for configs from older builds.

    Legacy configs that pre-date #9 have no ``transcription.engine`` key;
    they get ``"auto"`` so the selector picks the recommended engine for
    the user's hardware. Configs that pinned a now-removed engine are
    reset to ``"auto"`` so startup does not fail building a nonexistent
    engine. Any other explicit value is left alone.
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

