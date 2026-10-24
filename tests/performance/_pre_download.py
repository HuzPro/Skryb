"""Ensure every (engine, repo) pair we plan to benchmark is on disk.

Uses ``huggingface_hub.snapshot_download`` directly rather than going
through the project's chunked downloader - the latter exists for async
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

from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import ModelInfo

from tests.performance._combos import BenchmarkCombo


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
    license_policy: LicensePolicy,
    *,
    log: Callable[[str], None] = print,
) -> None:
    """Download any (engine, repo) pair not already complete on disk.

    Idempotent - repos whose snapshot already has weights are skipped.
    Every engine resolves its repo and pinned commit through
    ``ModelInfo``, so a pinned engine benchmarks the same commit it
    ships. ``revision or None`` because HuggingFace reads ``None`` as
    "default branch" and an empty string resolves nothing.

    ``license_policy`` is required, never defaulted: acquiring gated
    weights for a benchmark is the same acquisition the app gates, so a
    permissive default here would be a bypass. A gated model nobody has
    accepted is left undownloaded, and the runner refuses to build an
    engine for it, so it lands in the JSON as ``status='error'``.
    """
    from huggingface_hub import snapshot_download
    from skryb.transcription.model_info import ALL_MODELS

    needed: dict[tuple[str, str, str], ModelInfo] = {}
    for combo in combos:
        info = ALL_MODELS[combo.model_name]
        needed.setdefault((combo.engine_id, info.repo_id, info.revision), info)

    for (engine_id, repo_id, revision), info in sorted(needed.items()):
        if _repo_complete(models_dir, repo_id, revision):
            log(f"  [skip] {engine_id:<16} {repo_id} - already on disk")
            continue
        log(f"  [download] {engine_id:<16} {repo_id} - fetching...")
        try:
            license_policy.assert_accepted(info)
            snapshot_download(
                repo_id=repo_id,
                revision=revision or None,
                cache_dir=str(models_dir),
                ignore_patterns=list(_HF_IGNORE_PATTERNS),
                local_files_only=False,
            )
            log(f"  [download] {engine_id:<16} {repo_id} - complete")
        except Exception as exc:
            # Neither a failed fetch nor a refused licence aborts the
            # sweep. The affected combo is stopped again downstream - by
            # the runner's own licence gate, or by
            # ``local_files_only=True`` at load - and becomes a
            # ``status='error'`` entry in the JSON report, which is
            # exactly the contract the user asked for.
            log(
                f"  [download] {engine_id:<16} {repo_id} - FAILED "
                f"({type(exc).__name__}: {exc}); combo will be recorded "
                f"as an error during benchmark"
            )


def _repo_complete(models_dir: Path, repo_id: str, revision: str) -> bool:
    from skryb.transcription.hf_model_manager import HuggingFaceModelManager

    snapshots = models_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.exists():
        return False
    return any(
        HuggingFaceModelManager._snapshot_has_complete_model(snap)
        for snap in snapshots.iterdir()
        if snap.is_dir() and revision in ("", snap.name)
    )
