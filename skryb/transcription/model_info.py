import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from skryb.core.config import ConfigurationStore
from skryb.core.exceptions import ConfigurationError
from skryb.transcription.engine_types import EngineId, HardwareKind

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
    (``skryb/licenses/<license_id>.md``) the acceptance is keyed
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
        return format_size_mb(self.size_mb)


def format_size_mb(size_mb: int) -> str:
    if size_mb >= 1000:
        return f"{size_mb / 1000:.1f} GB"
    return f"{size_mb} MB"


# Plus and Clean show as one model in Settings, so both carry its description.
_BARK_PLUS_DESCRIPTION = (
    "Underdog Bark 3.0 Plus: fewer errors than Bark 3.0. Downloads two versions: "
    "Clean drops ums, false starts and repeats (Intended), Plus keeps them (Verbatim). English only"
)


_PARAKEET_ENGINE_PRIORITIES: Mapping[EngineId, int] = MappingProxyType(
    {"nano_parakeet": 100},
)

# Transformers-layout checkpoints can also put the encoder on an Intel NPU.
_NPU_PARAKEET_ENGINES: tuple[EngineId, ...] = ("openvino_parakeet", "nano_parakeet")
_NPU_PARAKEET_ENGINE_PRIORITIES: Mapping[EngineId, int] = MappingProxyType(
    {"openvino_parakeet": 110, "nano_parakeet": 100},
)


PARAKEET_MODELS: dict[str, ModelInfo] = {
    "parakeet-tdt-0.6b-v3": ModelInfo(
        name="parakeet-tdt-0.6b-v3",
        size_mb=2510,
        description="NVIDIA's original, built-in punctuation (25 languages)",
        repo_id="nvidia/parakeet-tdt-0.6b-v3",
        compatible_engines=("nano_parakeet",),
        engine_priorities=_PARAKEET_ENGINE_PRIORITIES,
    ),
    # Moondream's post-trained v3: the same network saved in transformers'
    # layout. Pinned because the loader's key renames are written against
    # this exact file.
    "parakeet-ultra": ModelInfo(
        name="parakeet-ultra",
        size_mb=1260,
        description="Parakeet v3 retrained by Moondream: fewer errors, same 25 languages",
        repo_id="moondream/parakeet-ultra",
        revision="73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b",
        compatible_engines=_NPU_PARAKEET_ENGINES,
        engine_priorities=_NPU_PARAKEET_ENGINE_PRIORITIES,
    ),
    # Conway Research's Underdog Bark 3.0 family: Parakeet v3 post-trained
    # for English (CC-BY-4.0). The three small ones ship one ANS-coded
    # model.bark, unpacked once at first load (bark_pack); Pro is plain bf16
    # in transformers' layout, like parakeet-ultra. Pinned because the
    # unpacker is written against these files.
    "underdog-bark-3.0": ModelInfo(
        name="underdog-bark-3.0",
        size_mb=163,
        description="Underdog Bark 3.0: Parakeet v3 retrained, 163 MB, English only",
        repo_id="ConwayResearch/Underdog-Bark-3.0",
        revision="85db2ad18ab41e534a222ee253c29191d77ed919",
        compatible_engines=_NPU_PARAKEET_ENGINES,
        engine_priorities=_NPU_PARAKEET_ENGINE_PRIORITIES,
    ),
    "underdog-bark-3.0-plus": ModelInfo(
        name="underdog-bark-3.0-plus",
        size_mb=180,
        description=_BARK_PLUS_DESCRIPTION,
        repo_id="ConwayResearch/Underdog-Bark-3.0-Plus",
        revision="d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb",
        compatible_engines=_NPU_PARAKEET_ENGINES,
        engine_priorities=_NPU_PARAKEET_ENGINE_PRIORITIES,
    ),
    "underdog-bark-3.0-clean": ModelInfo(
        name="underdog-bark-3.0-clean",
        size_mb=180,
        description=_BARK_PLUS_DESCRIPTION,
        repo_id="ConwayResearch/Underdog-Bark-3.0-Clean",
        revision="6d34ed8639b2b9a33e764acf06748519bbb950cf",
        compatible_engines=_NPU_PARAKEET_ENGINES,
        engine_priorities=_NPU_PARAKEET_ENGINE_PRIORITIES,
    ),
    "underdog-bark-3.0-pro": ModelInfo(
        name="underdog-bark-3.0-pro",
        size_mb=1254,
        description="Underdog Bark 3.0 Pro: the most accurate Bark, full precision, English only",
        repo_id="ConwayResearch/Underdog-Bark-3.0-Pro",
        revision="4b5c51e04aa8d16dda9ecdff71bc29f7360149e4",
        compatible_engines=_NPU_PARAKEET_ENGINES,
        engine_priorities=_NPU_PARAKEET_ENGINE_PRIORITIES,
    ),
}


_PHONON2_ENGINE_PRIORITIES: Mapping[EngineId, int] = MappingProxyType(
    {"phonon2_cpu": 100},
)


# Fermion Research's 164 MB compression of parakeet-tdt-0.6b-v3, the
# English-only Phonon-2.5. Pinned because the CPU engine release
# (``fermion-research``) is pinned against this archive.
PHONON2_MODELS: dict[str, ModelInfo] = {
    "phonon-2.5-english": ModelInfo(
        name="phonon-2.5-english",
        size_mb=164,
        description="Parakeet v3 shrunk to 164 MB by Fermion Research: fast on CPU, English only",
        repo_id="FermionResearch/Phonon-2.5-English",
        revision="da200358d638886bc96c758c672ad5568828b119",
        compatible_engines=("phonon2_cpu",),
        engine_priorities=_PHONON2_ENGINE_PRIORITIES,
    ),
}

CPU_DEFAULT_MODEL = "phonon-2.5-english"


def default_model_for(hardware: HardwareKind) -> str:
    """The model a fresh install starts on: Phonon-2.5 English when there is no GPU."""
    return CPU_DEFAULT_MODEL if hardware == "cpu" else DEFAULT_MODEL


_QWEN3_ASR_ENGINE_PRIORITIES: Mapping[EngineId, int] = MappingProxyType(
    {"native_qwen3_asr": 100},
)

# The most accurate open model on the Open ASR Leaderboard as of Oct 2026,
# run by this repo's own HIP engine (native/engine). Pinned because the
# packer's tensor layout is written against this exact checkpoint.
QWEN3_ASR_MODELS: dict[str, ModelInfo] = {
    "qwen3-asr-1.7b": ModelInfo(
        name="qwen3-asr-1.7b",
        size_mb=4080,
        description="Qwen3-ASR 1.7B on the native AMD engine: most accurate, no PyTorch (RX 9060 XT)",
        repo_id="Qwen/Qwen3-ASR-1.7B-hf",
        revision="bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        compatible_engines=("native_qwen3_asr",),
        engine_priorities=_QWEN3_ASR_ENGINE_PRIORITIES,
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
    {"crisperwhisper": 40, "crisperwhisper_ct2": 45},
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
        compatible_engines=("crisperwhisper", "crisperwhisper_ct2"),
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
    **PARAKEET_MODELS, **PHONON2_MODELS, **NEMOTRON_STREAMING_MODELS,
    **CRISPERWHISPER_MODELS, **QWEN3_ASR_MODELS,
}


def has_crisperwhisper_modes(name: str) -> bool:
    """Whether *name* is decoded with an intended/verbatim prefix."""
    return name in CRISPERWHISPER_MODELS


# Bark Plus comes as two downloads that differ only in what they do with
# ums, false starts and repeats, so Settings shows them as one model and
# the speech style picks the download.
SPEECH_STYLE_FAMILIES: dict[str, dict[str, str]] = {
    "underdog-bark-3.0-plus": {
        "intended": "underdog-bark-3.0-clean",
        "verbatim": "underdog-bark-3.0-plus",
    },
}


def speech_style_of(name: str) -> tuple[str, str] | None:
    """``(family, style)`` for a model that is one speech style of a family."""
    for family, styles in SPEECH_STYLE_FAMILIES.items():
        for style, member in styles.items():
            if member == name:
                return family, style
    return None


def family_display_size(family: str) -> str:
    """Both styles download together, so the family's size is theirs added up."""
    return format_size_mb(sum(ALL_MODELS[member].size_mb for member in SPEECH_STYLE_FAMILIES[family].values()))


def has_speech_styles(name: str) -> bool:
    return has_crisperwhisper_modes(name) or speech_style_of(name) is not None


def transcribes_verbatim(name: str, crisperwhisper_mode: str) -> bool:
    """Whether *name* hands over the words as spoken, so cleanup must stay off.

    A family member is its style; a CrisperWhisper model follows the mode
    setting. A mode left behind from an earlier model never applies to a
    model without styles.
    """
    family_style = speech_style_of(name)
    if family_style is not None:
        return family_style[1] == "verbatim"
    return has_crisperwhisper_modes(name) and crisperwhisper_mode == "verbatim"


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
    "parakeet-tdt-0.6b-v2": DEFAULT_MODEL,
    "nemotron-streaming-compare": "nemotron-speech-streaming-en-0.6b",
    # Same size and engine, fewer errors on real dictation.
    "phonon-2": "phonon-2.5-english",
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

