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
from voicetotext.transcription.model_info import (
    ALL_MODELS,
    GRANITE_MODELS,
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

_FASTER_WHISPER_DESCRIPTOR = EngineDescriptor(
    engine_id="faster_whisper",
    display_name="faster-whisper (CTranslate2)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=80,
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


# Per-engine beam_size defaults applied when the user has not set
# ``transcription.beam_size``. Granite ships IBM's tech-report recipe
# (``num_beams=4``); the whisper-family decoders converged on 5 as
# their HF/CTranslate2 default. Keeping these here — not on the
# adapters — means the config layer owns the "what does unset mean"
# question, and the adapters keep their own sane defaults intact for
# direct instantiation.
_DEFAULT_BEAM_SIZE_BY_ENGINE: dict[EngineId, int] = {
    "transformers": 5,
    "faster_whisper": 5,
    "granite_speech": 4,
    "nano_parakeet": 1,
}
_FALLBACK_BEAM_SIZE = 5


def build_engine_registry() -> EngineRegistry:
    """Wire the engine adapters this build ships with."""
    return EngineRegistry(
        descriptors=[
            _TRANSFORMERS_DESCRIPTOR,
            _FASTER_WHISPER_DESCRIPTOR,
            _NANO_PARAKEET_DESCRIPTOR,
            _GRANITE_SPEECH_DESCRIPTOR,
        ],
        builders={
            "transformers": _build_transformers_engine,
            "faster_whisper": _build_faster_whisper_engine,
            "nano_parakeet": _build_nano_parakeet_engine,
            "granite_speech": _build_granite_speech_engine,
        },
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


def _build_faster_whisper_engine(spec: EngineSpec) -> TranscriptionEngine:
    from voicetotext.transcription.faster_whisper_adapter import (
        FasterWhisperAdapter,
    )

    return FasterWhisperAdapter(
        model_name=spec.model_name,
        repo_id=spec.repo_id,
        device=spec.device,
        language=spec.language,
        beam_size=spec.beam_size,
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
    installed: set[EngineId] = set()
    for engine_id, repo_id in _engine_repo_ids(model_info):
        snapshots_dir = models_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            installed.add(engine_id)
    return frozenset(installed)


def _engine_repo_ids(model_info: ModelInfo) -> list[tuple[EngineId, str]]:
    pairs: list[tuple[EngineId, str]] = []
    if model_info.repo_id:
        if "transformers" in model_info.compatible_engines:
            pairs.append(("transformers", model_info.repo_id))
        if "nano_parakeet" in model_info.compatible_engines:
            pairs.append(("nano_parakeet", model_info.repo_id))
        if "granite_speech" in model_info.compatible_engines:
            pairs.append(("granite_speech", model_info.repo_id))
    if model_info.ct2_repo_id:
        pairs.append(("faster_whisper", model_info.ct2_repo_id))
    return pairs


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
        repo_id=_repo_id_for(model_info, engine_id),
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


def _repo_id_for(model_info: ModelInfo, engine_id: EngineId) -> str:
    if engine_id == "faster_whisper":
        return model_info.ct2_repo_id
    return model_info.repo_id  # transformers and nano_parakeet both use repo_id


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
    """Build a Composite ModelManager covering both engine caches.

    Each engine has its own HF repo namespace (``openai/whisper-*`` for
    transformers, ``Systran/faster-whisper-*`` for CTranslate2) and
    therefore its own ``models--*`` directory under the shared models
    root. Listing aggregates across both; download/delete routes to
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
    faster_whisper_models = _ct2_models()
    parakeet_models = {
        name: info for name, info in PARAKEET_MODELS.items() if info.repo_id
    }
    granite_models = {
        name: info for name, info in GRANITE_MODELS.items() if info.repo_id
    }

    transformers_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=transformers_models,
    )
    faster_whisper_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=faster_whisper_models,
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

    return CompositeModelManager(
        by_engine={
            "transformers": transformers_manager,
            "faster_whisper": faster_whisper_manager,
            "nano_parakeet": nano_parakeet_manager,
            "granite_speech": granite_speech_manager,
        },
        active_engine_provider=active_engine_provider,
    )


def _ct2_models() -> dict[str, ModelInfo]:
    """Mirror ``WHISPER_MODELS`` but swap ``repo_id`` for ``ct2_repo_id``.

    HuggingFaceModelManager looks up models by ``repo_id``, so a
    faster-whisper child needs entries pointing at the Systran mirror
    instead of the openai original.
    """
    return {
        name: ModelInfo(
            name=info.name,
            size_mb=info.size_mb,
            description=info.description,
            repo_id=info.ct2_repo_id,
            ct2_repo_id=info.ct2_repo_id,
            compatible_engines=info.compatible_engines,
            engine_priorities=info.engine_priorities,
        )
        for name, info in WHISPER_MODELS.items()
        if info.ct2_repo_id
    }
