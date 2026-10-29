import importlib.util
import threading
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path

from skryb.core.config import ConfigurationStore
from skryb.core.exceptions import ConfigurationError
from skryb.core.paths import AppPaths
from skryb.downloads.download_manager import DownloadManager
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.engine_registry import (
    EngineRegistry,
    EngineSpec,
)
from skryb.transcription.engine_selector import rank_engines
from skryb.transcription.engine_types import (
    ALL_HARDWARE_KINDS,
    CrisperWhisperMode,
    EngineCandidate,
    EngineDescriptor,
    EngineId,
    HardwareProfile,
)
from skryb.transcription.hardware_detector import HardwareDetector
from skryb.transcription.hf_model_manager import repo_has_loadable_snapshot
from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import (
    CRISPERWHISPER_MODELS,
    DEFAULT_MODEL,
    NEMOTRON_STREAMING_MODELS,
    PARAKEET_MODELS,
    PHONON2_MODELS,
    QWEN3_ASR_MODELS,
    ModelInfo,
    get_model_info,
)
from skryb.transcription.model_manager import ModelManager
from skryb.transcription.native_engine_dll import native_engine_dll

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

# Runs on the CPU whatever the hardware, so every hardware kind can select it.
_PHONON2_CPU_DESCRIPTOR = EngineDescriptor(
    engine_id="phonon2_cpu",
    display_name="Phonon-2 (Fermion CPU engine)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=80,
)

_CRISPERWHISPER_DESCRIPTOR = EngineDescriptor(
    engine_id="crisperwhisper",
    display_name="CrisperWhisper 2.0 (Transformers)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=40,
)


_CRISPERWHISPER_CT2_DESCRIPTOR = EngineDescriptor(
    engine_id="crisperwhisper_ct2",
    display_name="CrisperWhisper 2.0 (CTranslate2)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=45,
    requires_torch=False,
)


# Hand-written kernels compiled for gfx1200 only (the RX 9060 XT they were
# tuned and checked on), behind a DLL: no torch import.
_NATIVE_QWEN3_ASR_DESCRIPTOR = EngineDescriptor(
    engine_id="native_qwen3_asr",
    display_name="Qwen3-ASR (native HIP engine)",
    supported_hardware=("amd_rocm",),
    base_priority=95,
    requires_torch=False,
    supported_gfx=("gfx1200",),
)


# The encoder on an Intel NPU through OpenVINO, the rest of Parakeet in
# NumPy on the CPU; torch only for the one-time export. Only offered on
# machines without a GPU; available once
# the optional openvino package is installed (pip install skryb[npu]). With
# no NPU it runs the encoder on OpenVINO's CPU device.
_OPENVINO_PARAKEET_DESCRIPTOR = EngineDescriptor(
    engine_id="openvino_parakeet",
    display_name="Parakeet on the NPU (OpenVINO)",
    supported_hardware=("cpu",),
    base_priority=95,
    requires_torch=False,
)


_DEFAULT_BEAM_SIZE_BY_ENGINE: dict[EngineId, int] = {
    "nano_parakeet": 1,
    "nemotron_streaming": 1,
    "phonon2_cpu": 1,
    "crisperwhisper": 1,
    "crisperwhisper_ct2": 1,
    "native_qwen3_asr": 1,
    "openvino_parakeet": 1,
}
_FALLBACK_BEAM_SIZE = 1


def build_engine_registry() -> EngineRegistry:
    """Wire the engine adapters this build ships with."""
    return EngineRegistry(
        descriptors=[
            _NANO_PARAKEET_DESCRIPTOR,
            _NEMOTRON_STREAMING_DESCRIPTOR,
            _PHONON2_CPU_DESCRIPTOR,
            _CRISPERWHISPER_DESCRIPTOR,
            _CRISPERWHISPER_CT2_DESCRIPTOR,
            replace(_NATIVE_QWEN3_ASR_DESCRIPTOR, available=native_engine_dll() is not None),
            replace(
                _OPENVINO_PARAKEET_DESCRIPTOR,
                available=importlib.util.find_spec("openvino") is not None,
            ),
        ],
        builders={
            "nano_parakeet": _build_nano_parakeet_engine,
            "nemotron_streaming": _build_nemotron_streaming_engine,
            "phonon2_cpu": _build_phonon2_cpu_engine,
            "crisperwhisper": _build_crisperwhisper_engine,
            "crisperwhisper_ct2": _build_crisperwhisper_ct2_engine,
            "native_qwen3_asr": _build_native_qwen3_asr_engine,
            "openvino_parakeet": _build_openvino_parakeet_engine,
        },
    )


def _build_nemotron_streaming_engine(spec: EngineSpec) -> TranscriptionEngine:
    # Streaming-only: it has no batch TranscriptionEngine. The live streaming
    # engine is built directly in bootstrap (_load_streaming_engine); the batch
    # registry should never reach this model (the picker hides it in batch).
    raise ConfigurationError(
        "nemotron-speech-streaming is streaming-only; it has no batch engine."
    )


def _build_nano_parakeet_engine(spec: EngineSpec) -> TranscriptionEngine:
    from skryb.transcription.nano_parakeet_adapter import (
        NanoParakeetAdapter,
    )

    return NanoParakeetAdapter(
        repo_id=spec.repo_id,
        device=spec.device,
        model_dir=spec.model_dir,
    )


def _build_openvino_parakeet_engine(spec: EngineSpec) -> TranscriptionEngine:
    from skryb.transcription.openvino_parakeet_adapter import (
        OpenVinoParakeetAdapter,
    )

    return OpenVinoParakeetAdapter(model_dir=spec.model_dir, repo_id=spec.repo_id)


def _build_phonon2_cpu_engine(spec: EngineSpec) -> TranscriptionEngine:
    from skryb.transcription.hf_model_manager import (
        loadable_snapshot_dir,
    )
    from skryb.transcription.phonon2_cpu_adapter import Phonon2CpuAdapter

    return Phonon2CpuAdapter(
        loadable_snapshot_dir(spec.model_dir, spec.repo_id, spec.revision),
    )


def _build_native_qwen3_asr_engine(spec: EngineSpec) -> TranscriptionEngine:
    from skryb.transcription.hf_model_manager import (
        loadable_snapshot_dir,
    )
    from skryb.transcription.native_qwen3_asr_adapter import (
        NativeQwen3AsrAdapter,
    )

    return NativeQwen3AsrAdapter(
        loadable_snapshot_dir(spec.model_dir, spec.repo_id, spec.revision),
    )


def _build_crisperwhisper_engine(spec: EngineSpec) -> TranscriptionEngine:
    from skryb.transcription.crisperwhisper_adapter import (
        CrisperWhisperAdapter,
    )

    return CrisperWhisperAdapter(
        repo_id=spec.repo_id,
        revision=spec.revision,
        device=spec.device,
        model_dir=spec.model_dir,
        language=spec.language,
        mode=spec.crisperwhisper_mode,
    )


def _build_crisperwhisper_ct2_engine(spec: EngineSpec) -> TranscriptionEngine:
    from skryb.transcription.crisperwhisper_ct2_adapter import (
        CrisperWhisperCt2Adapter,
    )
    from skryb.transcription.ct2_conversion import ensure_ct2_model
    from skryb.transcription.hf_model_manager import (
        loadable_snapshot_dir,
    )

    return CrisperWhisperCt2Adapter(
        model_dir=ensure_ct2_model(
            loadable_snapshot_dir(
                spec.model_dir, spec.repo_id, spec.revision,
            ),
        ),
        device=spec.device,
        language=spec.language,
        mode=spec.crisperwhisper_mode,
    )


def selectable_engine_ids(
    config: ConfigurationStore,
    registry: EngineRegistry,
    models_dir: Path | None = None,
) -> tuple[EngineId, ...]:
    """Engine ids this config could end up running, read from config alone.

    An explicit ``transcription.engine`` names exactly one. ``"auto"``
    narrows to the configured model's ``compatible_engines``. Neither
    path touches hardware or an engine backend, which is what lets
    startup decide about torch before importing anything.

    A model name the catalogue no longer knows (a config that predates a
    migration) widens to every registered engine, so the answer stays on
    the safe side of "might need torch".
    """
    explicit = config.get("transcription.engine", "auto")
    if isinstance(explicit, str) and explicit != "auto":
        return (explicit,)  # type: ignore[return-value]
    try:
        model_info = get_model_info(
            str(config.get("transcription.model_size", DEFAULT_MODEL))
        )
    except ConfigurationError:
        return tuple(d.engine_id for d in registry.all_descriptors())
    if models_dir is None:
        return model_info.compatible_engines
    return _auto_selection(model_info, registry, models_dir)


def _auto_selection(
    model_info: ModelInfo,
    registry: EngineRegistry,
    models_dir: Path,
) -> tuple[EngineId, ...]:
    """The single engine ``auto`` will load, when that is knowable here.

    ``resolve_engine_id`` makes the same choice from the same ranking and
    the same installed set, so the torch question is answered about the
    engine that actually loads rather than about every candidate - which
    would report "needs torch" whenever any alternative did.

    Falls back to every candidate when the ranking needs a hardware
    probe, because probing is the import this is trying to avoid.
    """
    ranked = _ranking_that_ignores_hardware(model_info, registry)
    if not ranked:
        return model_info.compatible_engines
    installed = engines_with_model_on_disk(model_info, models_dir)
    for engine_id in ranked:
        if engine_id in installed:
            return (engine_id,)
    return (ranked[0],)


def startup_requires_torch(
    config: ConfigurationStore,
    registry: EngineRegistry,
    models_dir: Path | None = None,
) -> bool:
    """Whether this launch will load an engine built on torch."""
    return registry.requires_torch(
        selectable_engine_ids(config, registry, models_dir),
    )


UNPROBED_HARDWARE_KIND = "unprobed"


def probed_hardware_kind(
    config: ConfigurationStore,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    models_dir: Path,
) -> str:
    """The hardware kind for diagnostics, or ``unprobed`` if none was taken.

    Probing costs a torch import. A launch that runs a torch-free engine
    has no reason to pay it, and a diagnostics field is not reason
    enough - so it records that no probe happened rather than forcing one.
    """
    if not startup_requires_torch(config, registry, models_dir):
        return UNPROBED_HARDWARE_KIND
    return hardware_detector.detect().kind


def resolve_device(
    config: ConfigurationStore,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
    engine_ids: Iterable[EngineId],
) -> str:
    """The device string to hand an engine drawn from *engine_ids*.

    A torch-free engine gets the configured value unchanged - ``"auto"``
    included - and picks its own device from its own runtime. Asking
    torch on its behalf would import torch to answer a question about a
    GPU torch is not going to drive.
    """
    configured = config.get("transcription.device", "auto")
    if isinstance(configured, str) and configured != "auto":
        return configured
    if not registry.requires_torch(engine_ids):
        return "auto"
    return "cpu" if hardware_detector.detect().kind == "cpu" else "cuda"


def resolve_engine_id(
    config: ConfigurationStore,
    model_info: ModelInfo,
    hardware_detector: HardwareDetector,
    registry: EngineRegistry,
    *,
    installed_engines: frozenset[EngineId] = frozenset(),
) -> EngineId:
    explicit = config.get("transcription.engine", "auto")
    if explicit and explicit != "auto":
        return explicit  # type: ignore[return-value]
    ranked = _ranking_that_ignores_hardware(model_info, registry)
    if not ranked:
        ranked = _ranking_for_detected_hardware(
            model_info, registry, hardware_detector,
        )
    installed = [
        engine_id for engine_id in ranked if engine_id in installed_engines
    ]
    return installed[0] if installed else ranked[0]


# Any profile ranks the same once every candidate runs on every hardware
# kind, so this one stands in for the probe that is being skipped.
_UNPROBED_HARDWARE = HardwareProfile(
    kind="cpu", device_name="unprobed", gfx_arch=None,
)


def _ranking_that_ignores_hardware(
    model_info: ModelInfo,
    registry: EngineRegistry,
) -> tuple[EngineId, ...]:
    return tuple(
        candidate.engine_id
        for candidate in engine_candidates_ignoring_hardware(
            model_info, registry,
        )
    )


def engine_candidates_ignoring_hardware(
    model_info: ModelInfo,
    registry: EngineRegistry,
) -> tuple[EngineCandidate, ...]:
    """Ranked engine candidates when hardware cannot change the order.

    ``rank_engines`` consults hardware only to drop engines that do not
    support it. When every engine compatible with *model_info* runs on
    every hardware kind, nothing is ever dropped and the order is the
    priority order alone - so the GPU never has to be probed. Returns
    empty when hardware could matter, which sends the caller to the
    probe. The recommendation carries no reason: no hardware was probed
    to give one.
    """
    compatible = [
        descriptor for descriptor in registry.all_descriptors()
        if descriptor.engine_id in model_info.compatible_engines
    ]
    runs_anywhere = all(
        ALL_HARDWARE_KINDS <= set(descriptor.supported_hardware)
        for descriptor in compatible
    )
    if not compatible or not runs_anywhere:
        return ()
    return tuple(
        replace(candidate, reason="")
        for candidate in rank_engines(
            model_info, _UNPROBED_HARDWARE, registry.all_descriptors(),
        )
    )


def _ranking_for_detected_hardware(
    model_info: ModelInfo,
    registry: EngineRegistry,
    hardware_detector: HardwareDetector,
) -> tuple[EngineId, ...]:
    hardware = hardware_detector.detect()
    candidates = rank_engines(
        model_info, hardware, registry.all_descriptors(),
    )
    if not candidates:
        raise ConfigurationError(
            f"No engine compatible with model "
            f"'{model_info.name}' on hardware '{hardware.kind}'"
        )
    return tuple(candidate.engine_id for candidate in candidates)


def engines_with_model_on_disk(
    model_info: ModelInfo,
    models_dir: Path,
) -> frozenset[EngineId]:
    """Engines whose on-disk cache for *model_info* is actually loadable.

    Uses the same ``refs/main`` + complete-snapshot predicate as
    ``is_installed`` (see
    :func:`~skryb.transcription.hf_model_manager.repo_has_loadable_snapshot`),
    so a partial or cancelled download never biases engine selection toward
    an engine whose weights cannot actually be opened.
    """
    installed: set[EngineId] = set()
    for engine_id, repo_id in _engine_repo_ids(model_info):
        repo_dir = models_dir / f"models--{repo_id.replace('/', '--')}"
        if not repo_has_loadable_snapshot(repo_dir, model_info.revision):
            continue
        if engine_id == "crisperwhisper_ct2" and not _ctranslate2_installed():
            continue
        installed.add(engine_id)
    return frozenset(installed)


def _ctranslate2_installed() -> bool:
    """Whether the CTranslate2 runtime is importable, without importing it.

    Selection runs before startup decides about torch, so this has to
    answer from the module finder rather than by loading a backend.
    """
    return importlib.util.find_spec("ctranslate2") is not None


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
    hardware_detector: HardwareDetector,
    *,
    license_policy: LicensePolicy,
    device_override: str | None = None,
) -> TranscriptionEngine:
    model_name = config.get(
        "transcription.model_size", DEFAULT_MODEL,
    )
    model_info = get_model_info(model_name)
    license_policy.assert_accepted(model_info)
    installed = engines_with_model_on_disk(model_info, paths.models_dir)
    engine_id = resolve_engine_id(
        config, model_info, hardware_detector, registry,
        installed_engines=installed,
    )
    spec = EngineSpec(
        model_name=model_name,
        repo_id=model_info.repo_id,
        device=device_override or resolve_device(
            config, registry, hardware_detector, (engine_id,),
        ),
        language=config.get("transcription.language", "en"),
        beam_size=_resolve_beam_size(config, engine_id),
        model_dir=paths.models_dir,
        revision=model_info.revision,
        crisperwhisper_mode=_resolve_crisperwhisper_mode(config, engine_id),
    )
    return registry.build(engine_id, spec)


def _resolve_beam_size(
    config: ConfigurationStore, engine_id: EngineId,
) -> int:
    explicit = config.get("transcription.beam_size", None)
    if explicit is None:
        return _DEFAULT_BEAM_SIZE_BY_ENGINE.get(engine_id, _FALLBACK_BEAM_SIZE)
    return explicit  # type: ignore[return-value]


def _resolve_crisperwhisper_mode(
    config: ConfigurationStore,
    engine_id: EngineId,
) -> CrisperWhisperMode:
    if engine_id not in ("crisperwhisper", "crisperwhisper_ct2"):
        return "intended"
    from skryb.transcription.crisperwhisper_adapter import (
        resolve_crisperwhisper_mode,
    )

    return resolve_crisperwhisper_mode(
        config.get("transcription.crisperwhisper_mode", "intended")
    )


def create_model_manager(
    models_dir: Path,
    download_manager: DownloadManager,
    license_policy: LicensePolicy,
) -> ModelManager:
    return build_composite_model_manager(
        models_dir,
        download_manager,
        active_engine_provider=lambda: "nano_parakeet",
        license_policy=license_policy,
    )


def build_composite_model_manager(
    models_dir: Path,
    download_manager: DownloadManager,
    active_engine_provider: Callable[[], EngineId],
    license_policy: LicensePolicy,
) -> ModelManager:
    """Build a Composite ModelManager covering every engine cache.

    Each engine has its own HF repo namespace and therefore its own
    ``models--*`` directory under the shared models root. Listing aggregates
    across them; download/delete routes to the active engine.
    """
    from skryb.downloads.chunked_file_downloader import ChunkedFileDownloader
    from skryb.downloads.hf_repo_resolver import HfRepoResolver
    from skryb.downloads.requests_http_client import RequestsHttpClient
    from skryb.transcription.composite_model_manager import (
        CompositeModelManager,
    )
    from skryb.transcription.hf_model_manager import HuggingFaceModelManager

    # Both clients are built on first download. Importing requests and
    # huggingface_hub here put about a hundred modules on every launch's
    # path to the tray, for downloads most launches never make.
    http = RequestsHttpClient(session=_BuiltOnFirstUse(_requests_session))
    file_downloader = ChunkedFileDownloader(http=http)
    resolver = HfRepoResolver(api=_BuiltOnFirstUse(_hf_api))

    parakeet_models = {
        name: info for name, info in PARAKEET_MODELS.items() if info.repo_id
    }
    nemotron_streaming_models = {
        name: info for name, info in NEMOTRON_STREAMING_MODELS.items() if info.repo_id
    }
    phonon2_models = {
        name: info for name, info in PHONON2_MODELS.items() if info.repo_id
    }
    crisperwhisper_models = {
        name: info for name, info in CRISPERWHISPER_MODELS.items() if info.repo_id
    }
    qwen3_asr_models = {
        name: info for name, info in QWEN3_ASR_MODELS.items() if info.repo_id
    }

    nano_parakeet_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=parakeet_models,
        license_policy=license_policy,
    )
    nemotron_streaming_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=nemotron_streaming_models,
        license_policy=license_policy,
    )
    phonon2_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=phonon2_models,
        license_policy=license_policy,
    )
    crisperwhisper_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=crisperwhisper_models,
        license_policy=license_policy,
    )

    qwen3_asr_manager = HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=qwen3_asr_models,
        license_policy=license_policy,
    )

    return CompositeModelManager(
        by_engine={
            "nano_parakeet": nano_parakeet_manager,
            "openvino_parakeet": nano_parakeet_manager,
            "nemotron_streaming": nemotron_streaming_manager,
            "phonon2_cpu": phonon2_manager,
            "crisperwhisper": crisperwhisper_manager,
            "crisperwhisper_ct2": crisperwhisper_manager,
            "native_qwen3_asr": qwen3_asr_manager,
        },
        active_engine_provider=active_engine_provider,
    )


class _BuiltOnFirstUse:
    """Stands in for an object that is only built when first used."""

    def __init__(self, build: Callable[[], object]) -> None:
        self._build = build
        self._built: object | None = None
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> object:
        # Download chunks run on several threads, so only one may build it.
        with self._lock:
            if self._built is None:
                self._built = self._build()
        return getattr(self._built, name)


def _requests_session() -> object:
    import requests

    return requests.Session()


def _hf_api() -> object:
    from huggingface_hub import HfApi

    return HfApi()
