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
    ARK_ASR_MODELS,
    DEFAULT_MODEL,
    NEMOTRON_STREAMING_MODELS,
    PARAKEET_MODELS,
    ModelInfo,
    get_model_info,
)
from voicetotext.transcription.model_manager import ModelManager


_NANO_PARAKEET_DESCRIPTOR = EngineDescriptor(
    engine_id="nano_parakeet",
    display_name="nano-parakeet (PyTorch)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=90,
    supports_streaming=True,
)

_NEMOTRON_STREAMING_DESCRIPTOR = EngineDescriptor(
    engine_id="nemotron_streaming",
    display_name="Nemotron Streaming (cache-aware, GPU)",
    supported_hardware=("nvidia_cuda", "amd_rocm"),
    base_priority=85,
    supports_streaming=True,
    streaming_only=True,
)


_ARK_ASR_DESCRIPTOR = EngineDescriptor(
    engine_id="ark_asr",
    display_name="ARK-ASR 3B (Transformers)",
    supported_hardware=("nvidia_cuda", "amd_rocm"),
    base_priority=80,
)


_DEFAULT_BEAM_SIZE_BY_ENGINE: dict[EngineId, int] = {
    "ark_asr": 1,
    "nano_parakeet": 1,
    "nemotron_streaming": 1,
}
_FALLBACK_BEAM_SIZE = 1


def build_engine_registry() -> EngineRegistry:
    """Wire the engine adapters this build ships with."""
    return EngineRegistry(
        descriptors=[
            _NANO_PARAKEET_DESCRIPTOR,
            _NEMOTRON_STREAMING_DESCRIPTOR,
            _ARK_ASR_DESCRIPTOR,
        ],
        builders={
            "nano_parakeet": _build_nano_parakeet_engine,
            "nemotron_streaming": _build_nemotron_streaming_engine,
            "ark_asr": _build_ark_asr_engine,
        },
    )


def _build_nemotron_streaming_engine(spec: EngineSpec) -> TranscriptionEngine:
    # Streaming-only: it has no batch TranscriptionEngine. The live streaming
    # engine is built directly in bootstrap (_load_streaming_engine); the batch
    # registry should never reach this model (the picker hides it in batch).
    raise ConfigurationError(
        "nemotron-speech-streaming is streaming-only; it has no batch engine."
    )


def _build_ark_asr_engine(spec: EngineSpec) -> TranscriptionEngine:
    from voicetotext.transcription.ark_asr_adapter import ArkAsrAdapter

    return ArkAsrAdapter(
        repo_id=spec.repo_id,
        device=spec.device,
        model_dir=spec.model_dir,
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
    model_name = config.get(
        "transcription.model_size", DEFAULT_MODEL,
    )
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
        active_engine_provider=lambda: "nano_parakeet",
    )


def build_composite_model_manager(
    models_dir: Path,
    download_manager: DownloadManager,
    active_engine_provider: Callable[[], EngineId],
) -> ModelManager:
    """Build a Composite ModelManager covering every engine cache.

    Each engine has its own HF repo namespace and therefore its own
    ``models--*`` directory under the shared models root. Listing aggregates
    across them; download/delete routes to the active engine.
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

    parakeet_models = {
        name: info for name, info in PARAKEET_MODELS.items() if info.repo_id
    }
    nemotron_streaming_models = {
        name: info for name, info in NEMOTRON_STREAMING_MODELS.items() if info.repo_id
    }
    ark_asr_models = {
        name: info for name, info in ARK_ASR_MODELS.items() if info.repo_id
    }

    nano_parakeet_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=parakeet_models,
    )
    nemotron_streaming_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=nemotron_streaming_models,
    )
    ark_asr_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=ark_asr_models,
    )

    return CompositeModelManager(
        by_engine={
            "nano_parakeet": nano_parakeet_manager,
            "nemotron_streaming": nemotron_streaming_manager,
            "ark_asr": ark_asr_manager,
        },
        active_engine_provider=active_engine_provider,
    )
