"""Ensure every (engine, repo) pair we plan to benchmark is on disk.

Uses ``huggingface_hub.snapshot_download`` directly rather than going
through the project's chunked downloader — the latter exists for async
UI download flows that we don't need here. Snapshot layout matches what
the adapters expect (each adapter passes ``local_files_only=True`` and
reads from ``models_dir``).

``HF_HUB_CACHE`` must be set to the target ``models_dir`` BEFORE this
module is imported, otherwise the snapshot lands in the system default
cache and the adapters won't find it. The performance conftest does the
env set at module-load time.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

from tests.performance._combos import BenchmarkCombo


# NB: do NOT include ``*.bin`` here even though
# ``voicetotext/downloads/hf_repo_resolver._IGNORE_PATTERNS`` does.
# CTranslate2 (faster-whisper) stores its quantized weights in
# ``model.bin`` — the project's resolver pattern is wrong for CT2
# repos and would leave them weightless on disk. For the openai/whisper
# repos we may end up with both ``model.safetensors`` and the redundant
# ``pytorch_model.bin``; that's harmless duplication for benchmarks.
_HF_IGNORE_PATTERNS = (
    "*.h5",
    "*.msgpack",
    "*.fp32-*",
    "*.index.fp32.json",
    "README*",
    ".gitattributes",
    ".gitignore",
)


def ensure_models_downloaded(
    combos: Iterable[BenchmarkCombo],
    models_dir: Path,
    *,
    log: Callable[[str], None] = print,
) -> None:
    """Download any (engine, repo) pair not already complete on disk.

    Idempotent — repos whose snapshot already has weights are skipped.
    Granite, Parakeet, and openai-whisper repos go through ``repo_id``;
    faster-whisper goes through ``ct2_repo_id``.
    """
    from huggingface_hub import snapshot_download
    from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
    from voicetotext.transcription.model_info import ALL_MODELS

    needed: set[tuple[str, str]] = set()
    for combo in combos:
        info = ALL_MODELS[combo.model_name]
        repo_id = _repo_id_for_engine(combo.engine_id, info)
        needed.add((combo.engine_id, repo_id))

    for engine_id, repo_id in sorted(needed):
        if _repo_complete(models_dir, repo_id):
            log(f"  [skip] {engine_id:<16} {repo_id} — already on disk")
            continue
        log(f"  [download] {engine_id:<16} {repo_id} — fetching...")
        try:
            snapshot_download(
                repo_id=repo_id,
                cache_dir=str(models_dir),
                ignore_patterns=list(_HF_IGNORE_PATTERNS),
                local_files_only=False,
            )
            log(f"  [download] {engine_id:<16} {repo_id} — complete")
        except Exception as exc:
            # Pre-download failures must NOT abort the sweep. The
            # affected combo will fail loudly at ``engine.load_model()``
            # time (adapters use ``local_files_only=True``) and that
            # failure becomes a ``status='error'`` entry in the JSON
            # report — which is exactly the contract the user asked for.
            log(
                f"  [download] {engine_id:<16} {repo_id} — FAILED "
                f"({type(exc).__name__}: {exc}); combo will be recorded "
                f"as an error during benchmark"
            )


def _repo_id_for_engine(engine_id: str, info) -> str:
    if engine_id == "faster_whisper":
        return info.ct2_repo_id
    return info.repo_id


def _repo_complete(models_dir: Path, repo_id: str) -> bool:
    from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager

    snapshots = models_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.exists():
        return False
    return any(
        HuggingFaceModelManager._snapshot_has_complete_model(snap)
        for snap in snapshots.iterdir()
        if snap.is_dir()
    )
