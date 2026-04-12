from pathlib import Path

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.paths import AppPaths
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.model_manager import ModelManager


def create_engine(
    config: ConfigurationStore,
    paths: AppPaths,
    device_override: str | None = None,
) -> TranscriptionEngine:
    """Build the transformers Whisper engine.

    Pass ``device_override`` to force a specific device (e.g. ``"cpu"`` for the
    CPU fallback path); otherwise the device is read from
    ``transcription.device`` in *config*.
    """
    from voicetotext.transcription.transformers_whisper_adapter import (
        TransformersWhisperAdapter,
    )

    if device_override:
        device = device_override
    else:
        device = _resolve_device(config.get("transcription.device", "auto"))

    return TransformersWhisperAdapter(
        model_size=config.get("transcription.model_size", "base"),
        device=device,
        language=config.get("transcription.language"),
        beam_size=config.get("transcription.beam_size", 5),
        download_root=str(paths.models_dir),
    )


def create_model_manager(
    models_dir: Path, download_manager: DownloadManager
) -> ModelManager:
    """Return the HuggingFace model manager for the transformers backend."""
    import requests
    from huggingface_hub import HfApi

    from voicetotext.downloads.chunked_file_downloader import ChunkedFileDownloader
    from voicetotext.downloads.hf_repo_resolver import HfRepoResolver
    from voicetotext.downloads.requests_http_client import RequestsHttpClient
    from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
    from voicetotext.transcription.model_info import TRANSFORMERS_WHISPER_MODELS

    session = requests.Session()
    http = RequestsHttpClient(session=session)
    file_downloader = ChunkedFileDownloader(http=http)
    resolver = HfRepoResolver(api=HfApi())

    return HuggingFaceModelManager(
        models_dir=models_dir,
        download_manager=download_manager,
        resolver=resolver,
        file_downloader=file_downloader,
        models=TRANSFORMERS_WHISPER_MODELS,
    )


def _resolve_device(configured_device: str) -> str:
    if configured_device != "auto":
        return configured_device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
