from pathlib import Path

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.paths import AppPaths
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.gpu_detection import detect_available_backends
from voicetotext.transcription.model_manager import ModelManager


def create_engine(
    engine_name: str,
    config: ConfigurationStore,
    paths: AppPaths,
) -> TranscriptionEngine:
    """Build a :class:`TranscriptionEngine` for *engine_name*.

    Pass ``"auto"`` to let hardware detection choose, ``"cpu"`` to force
    the CPU path, or an explicit engine name (``"faster_whisper"`` /
    ``"transformers"``).
    """
    if engine_name == "auto":
        engine_name = _auto_select_engine()

    if engine_name == "cpu":
        engine_name = "faster_whisper"

    device = _resolve_device(engine_name, config.get("transcription.device", "auto"))

    if engine_name == "transformers":
        from voicetotext.transcription.transformers_whisper_adapter import (
            TransformersWhisperAdapter,
        )

        return TransformersWhisperAdapter(
            model_size=config.get("transcription.model_size", "base"),
            device=device,
            language=config.get("transcription.language"),
            beam_size=config.get("transcription.beam_size", 5),
            download_root=str(paths.models_dir),
        )

    # Default: faster_whisper (also used for CPU fallback)
    from voicetotext.transcription.faster_whisper_adapter import FasterWhisperAdapter

    compute_type = config.get("transcription.compute_type", "float16")
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    return FasterWhisperAdapter(
        model_size=config.get("transcription.model_size", "base"),
        device=device,
        compute_type=compute_type,
        language=config.get("transcription.language"),
        beam_size=config.get("transcription.beam_size", 5),
        vad_filter=config.get("transcription.vad_filter", True),
        download_root=str(paths.models_dir),
    )


def resolve_engine_name(config: ConfigurationStore) -> str:
    """Return the concrete engine name, resolving ``"auto"``."""
    engine = config.get("transcription.engine", "auto")
    if engine == "auto":
        return _auto_select_engine()
    return engine


def create_model_manager(engine_name: str, models_dir: Path) -> ModelManager:
    """Return a :class:`ModelManager` appropriate for *engine_name*."""
    from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
    from voicetotext.transcription.model_info import (
        FASTER_WHISPER_MODELS,
        TRANSFORMERS_WHISPER_MODELS,
    )

    if engine_name == "transformers":
        return HuggingFaceModelManager(models_dir, TRANSFORMERS_WHISPER_MODELS)
    return HuggingFaceModelManager(models_dir, FASTER_WHISPER_MODELS)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _auto_select_engine() -> str:
    backends = detect_available_backends()
    return backends[0]


def _resolve_device(engine_name: str, configured_device: str) -> str:
    if configured_device != "auto":
        return configured_device

    if engine_name == "transformers":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    # faster_whisper
    try:
        import ctranslate2

        ctranslate2.get_supported_compute_types("cuda")
        return "cuda"
    except Exception:
        return "cpu"
