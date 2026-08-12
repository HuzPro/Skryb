from collections.abc import Callable
from pathlib import Path

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.exceptions import ConfigurationError
from voicetotext.core.paths import AppPaths
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_registry import (
    EngineRegistry,
    EngineSpec,
)
from voicetotext.transcription.engine_selector import rank_engines
from voicetotext.transcription.engine_types import (
    EngineDescriptor,
    EngineId,
    HardwareProfile,
)
from voicetotext.transcription.hf_model_manager import repo_has_loadable_snapshot
from voicetotext.transcription.model_info import (
    ALL_MODELS,
    GRANITE_MODELS,
    NEMOTRON_STREAMING_MODELS,
    PARAKEET_MODELS,
    WHISPER_MODELS,
    ModelInfo,
    get_model_info,
)
from voicetotext.transcription.model_manager import ModelManager


_TRANSFORMERS_DESCRIPTOR = EngineDescriptor(
    engine_id="transformers",
    display_name="Transformers (HuggingFace)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=50,
)

_NANO_PARAKEET_DESCRIPTOR = EngineDescriptor(
    engine_id="nano_parakeet",
    display_name="nano-parakeet (PyTorch)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=90,
    supports_streaming=True,
)

_GRANITE_SPEECH_DESCRIPTOR = EngineDescriptor(
    engine_id="granite_speech",
    display_name="IBM Granite Speech (transformers)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=95,
)

_NEMOTRON_STREAMING_DESCRIPTOR = EngineDescriptor(
    engine_id="nemotron_streaming",
    display_name="Nemotron Streaming (cache-aware, GPU)",
    supported_hardware=("nvidia_cuda", "amd_rocm"),
    base_priority=85,
    supports_streaming=True,
    streaming_only=True,
)


# Per-engine beam_size defaults applied when the user has not set
# ``transcription.beam_size``. Granite ships IBM's tech-report recipe
# (``num_beams=4``); the whisper-family decoders converged on 5 as
# their HF default. Keeping these here - not on the
# adapters - means the config layer owns the "what does unset mean"
# question, and the adapters keep their own sane defaults intact for
# direct instantiation.
_DEFAULT_BEAM_SIZE_BY_ENGINE: dict[EngineId, int] = {
    "transformers": 5,
    "granite_speech": 4,
    "nano_parakeet": 1,
    "nemotron_streaming": 1,
}
_FALLBACK_BEAM_SIZE = 5


def build_engine_registry() -> EngineRegistry:
    """Wire the engine adapters this build ships with."""
    return EngineRegistry(
        descriptors=[
            _TRANSFORMERS_DESCRIPTOR,
            _NANO_PARAKEET_DESCRIPTOR,
            _GRANITE_SPEECH_DESCRIPTOR,
            _NEMOTRON_STREAMING_DESCRIPTOR,
        ],
        builders={
            "transformers": _build_transformers_engine,
            "nano_parakeet": _build_nano_parakeet_engine,
            "granite_speech": _build_granite_speech_engine,
            "nemotron_streaming": _build_nemotron_streaming_engine,
        },
    )


def _build_nemotron_streaming_engine(spec: EngineSpec) -> TranscriptionEngine:
    # Streaming-only: it has no batch TranscriptionEngine. The live streaming
    # engine is built directly in bootstrap (_load_streaming_engine); the batch
    # registry should never reach this model (the picker hides it in batch).
    raise ConfigurationError(
        "nemotron-speech-streaming is streaming-only; it has no batch engine."
    )


def _build_transformers_engine(spec: EngineSpec) -> TranscriptionEngine:
    from voicetotext.transcription.transformers_whisper_adapter import (
        TransformersWhisperAdapter,
    )

    return TransformersWhisperAdapter(
        model_size=spec.model_name,
        device=spec.device,
        language=spec.language,
        beam_size=spec.beam_size,
        download_root=str(spec.model_dir),
    )


def _build_nano_parakeet_engine(spec: EngineSpec) -> TranscriptionEngine:
    from voicetotext.transcription.nano_parakeet_adapter import (
        NanoParakeetAdapter,
    )

    return NanoParakeetAdapter(
        repo_id=spec.repo_id,
        device=spec.device,
        model_dir=spec.model_dir,
    )


def _build_granite_speech_engine(spec: EngineSpec) -> TranscriptionEngine:
    from voicetotext.transcription.granite_speech_adapter import (
        GraniteSpeechAdapter,
    )

    return GraniteSpeechAdapter(
        repo_id=spec.repo_id,
        device=spec.device,
        model_dir=spec.model_dir,
        num_beams=spec.beam_size,
    )


def resolve_engine_id(
    config: ConfigurationStore,
    model_info: ModelInfo,
    hardware: HardwareProfile,
    registry: EngineRegistry,
    *,
    installed_engines: frozenset[EngineId] = frozenset(),
) -> EngineId:
    explicit = config.get("transcription.engine", "auto")
    if explicit and explicit != "auto":
        return explicit  # type: ignore[return-value]
    candidates = rank_engines(model_info, hardware, registry.all_descriptors())
    if not candidates:
        raise ConfigurationError(
            f"No engine compatible with model "
            f"'{model_info.name}' on hardware '{hardware.kind}'"
        )
    if installed_engines:
        installed_candidate = next(
            (c for c in candidates if c.engine_id in installed_engines),
            None,
        )
        if installed_candidate is not None:
            return installed_candidate.engine_id
    return candidates[0].engine_id


def engines_with_model_on_disk(
    model_info: ModelInfo,
    models_dir: Path,
) -> frozenset[EngineId]:
    """Engines whose on-disk cache for *model_info* is actually loadable.

    Uses the same ``refs/main`` + complete-snapshot predicate as
    ``is_installed`` (see
    :func:`~voicetotext.transcription.hf_model_manager.repo_has_loadable_snapshot`),
    so a partial or cancelled download never biases engine selection toward
    an engine whose weights cannot actually be opened.
    """
    installed: set[EngineId] = set()
    for engine_id, repo_id in _engine_repo_ids(model_info):
        repo_dir = models_dir / f"models--{repo_id.replace('/', '--')}"
        if repo_has_loadable_snapshot(repo_dir):
            installed.add(engine_id)
    return frozenset(installed)


def _engine_repo_ids(model_info: ModelInfo) -> list[tuple[EngineId, str]]:
    # Every engine now loads from the same ``repo_id``; the per-engine
    # routing collapsed once faster-whisper (the only engine that used a
    # separate ``ct2_repo_id``) was removed.
    if not model_info.repo_id:
        return []
    return [
        (engine_id, model_info.repo_id)
        for engine_id in model_info.compatible_engines
    ]


def create_engine_for_active_model(
    config: ConfigurationStore,
    paths: AppPaths,
    registry: EngineRegistry,
    hardware: HardwareProfile,
    *,
    device_override: str | None = None,
) -> TranscriptionEngine:
    model_name = config.get("transcription.model_size", "base")
    model_info = get_model_info(model_name)
    installed = engines_with_model_on_disk(model_info, paths.models_dir)
    engine_id = resolve_engine_id(
        config, model_info, hardware, registry,
        installed_engines=installed,
    )
    spec = EngineSpec(
        model_name=model_name,
        repo_id=model_info.repo_id,
        device=device_override or _resolve_device(config, hardware),
        language=config.get("transcription.language", "en"),
        beam_size=_resolve_beam_size(config, engine_id),
        model_dir=paths.models_dir,
    )
    return registry.build(engine_id, spec)


def _resolve_beam_size(
    config: ConfigurationStore, engine_id: EngineId,
) -> int:
    explicit = config.get("transcription.beam_size", None)
    if explicit is None:
        return _DEFAULT_BEAM_SIZE_BY_ENGINE.get(engine_id, _FALLBACK_BEAM_SIZE)
    return explicit  # type: ignore[return-value]


def _resolve_device(
    config: ConfigurationStore, hardware: HardwareProfile,
) -> str:
    configured = config.get("transcription.device", "auto")
    if configured and configured != "auto":
        return configured  # type: ignore[return-value]
    return "cpu" if hardware.kind == "cpu" else "cuda"


def create_model_manager(
    models_dir: Path,
    download_manager: DownloadManager,
) -> ModelManager:
    return build_composite_model_manager(
        models_dir,
        download_manager,
        active_engine_provider=lambda: "transformers",
    )


def build_composite_model_manager(
    models_dir: Path,
    download_manager: DownloadManager,
    active_engine_provider: Callable[[], EngineId],
) -> ModelManager:
    """Build a Composite ModelManager covering every engine cache.

    Each engine has its own HF repo namespace (``openai/whisper-*`` for
    transformers, ``nvidia/parakeet-*`` for nano-parakeet, etc.) and
    therefore its own ``models--*`` directory under the shared models
    root. Listing aggregates across them; download/delete routes to
    the engine returned by ``active_engine_provider``.
    """
    import requests
    from huggingface_hub import HfApi

    from voicetotext.downloads.chunked_file_downloader import ChunkedFileDownloader
    from voicetotext.downloads.hf_repo_resolver import HfRepoResolver
    from voicetotext.downloads.requests_http_client import RequestsHttpClient
    from voicetotext.transcription.composite_model_manager import (
        CompositeModelManager,
    )
    from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager

    session = requests.Session()
    http = RequestsHttpClient(session=session)
    file_downloader = ChunkedFileDownloader(http=http)
    resolver = HfRepoResolver(api=HfApi())

    transformers_models = {
        name: info for name, info in WHISPER_MODELS.items() if info.repo_id
    }
    parakeet_models = {
        name: info for name, info in PARAKEET_MODELS.items() if info.repo_id
    }
    granite_models = {
        name: info for name, info in GRANITE_MODELS.items() if info.repo_id
    }
    nemotron_streaming_models = {
        name: info for name, info in NEMOTRON_STREAMING_MODELS.items() if info.repo_id
    }

    transformers_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=transformers_models,
    )
    nano_parakeet_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=parakeet_models,
    )
    granite_speech_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=granite_models,
    )
    nemotron_streaming_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=nemotron_streaming_models,
    )

    return CompositeModelManager(
        by_engine={
            "transformers": transformers_manager,
            "nano_parakeet": nano_parakeet_manager,
            "granite_speech": granite_speech_manager,
            "nemotron_streaming": nemotron_streaming_manager,
        },
        active_engine_provider=active_engine_provider,
    )
