from collections.abc import Callable
from pathlib import Path

from voicetotext.transcription.backend_imports import (
    make_engine_dependency_error,
)


_CPU_DEVICE = "cpu"


class ParakeetModelLoader:
    """Loads a nano-parakeet ``ParakeetTDT`` model into the app's cache.

    Shared by the batch adapter and the streaming engine so the
    ``hf_hub_download`` redirect and the bf16 dtype choice live in exactly
    one place. nano-parakeet caches by ``(repo, device, dtype)``, so asking
    for the same tuple twice returns the same in-memory model - the batch
    and streaming engines never pay for two loads.
    """

    def __init__(self, repo_id: str, model_dir: Path) -> None:
        self._repo_id = repo_id
        self._model_dir = model_dir

    def load(self, device: str) -> object:
        nano_parakeet_module, torch_module = self._import_backend()
        return self._load_with_scoped_hf_patch(
            nano_parakeet_module, torch_module, device,
        )

    def _import_backend(self) -> tuple[object, object]:
        try:
            import nano_parakeet
            import torch
        except ImportError as exc:
            raise make_engine_dependency_error(
                package_name="nano_parakeet",
                install_hint="Nano-Parakeet not installed: pip install nano-parakeet",
                underlying=exc,
            ) from exc
        return nano_parakeet, torch

    def _load_with_scoped_hf_patch(
        self, nano_parakeet_module: object, torch_module: object, device: str,
    ) -> object:
        """Run ``from_pretrained`` with the hf_hub_download patch installed.

        The patch redirects nano-parakeet's internal downloads into the
        app's models dir (see :meth:`_make_patched_download`) and forces
        ``local_files_only`` to skip the freshness HEAD request. It is
        installed only for the lifetime of the ``from_pretrained`` call
        and restored in a ``finally`` block, so back-to-back loads never
        stomp each other's closure on the module-global.

        The dtype is chosen by device: bf16 on GPU (the chunked sweep
        showed it matches fp32 text on RDNA4/gfx1200 while cutting latency
        and VRAM), but fp32 on CPU. nano-parakeet casts the encoder,
        decoder, and joint to the requested dtype while leaving the
        preprocessor in fp32; on CPU the fp32 encoder output then hits the
        bf16 joint and ``F.linear`` raises "mat1 and mat2 must have the
        same dtype". CPU has no accelerated bf16 matmul anyway, so fp32 is
        both correct and faster there.
        """
        original_download = nano_parakeet_module.hf_hub_download
        nano_parakeet_module.hf_hub_download = self._make_patched_download(
            original_download,
        )
        try:
            return nano_parakeet_module.from_pretrained(
                model_name=self._repo_id,
                device=device,
                dtype=self._dtype_for(torch_module, device),
            )
        finally:
            nano_parakeet_module.hf_hub_download = original_download

    @staticmethod
    def _dtype_for(torch_module: object, device: str) -> object:
        if device == _CPU_DEVICE:
            return torch_module.float32
        return torch_module.bfloat16

    def _make_patched_download(
        self, original: Callable[..., object],
    ) -> Callable[..., object]:
        """Build the closure that redirects downloads into ``model_dir``.

        nano-parakeet calls ``hf_hub_download(repo_id, filename)`` with no
        ``cache_dir``, so it falls through to ``~/.cache/huggingface/hub``.
        The app's downloader writes to ``self._model_dir`` - without the
        redirect an already-downloaded model is invisible. The caller has
        already verified the files are on disk, so ``local_files_only``
        short-circuits the upstream HEAD round trip.
        """
        cache_dir = str(self._model_dir)

        def patched(*args: object, **kwargs: object) -> object:
            kwargs.setdefault("cache_dir", cache_dir)
            kwargs.setdefault("local_files_only", True)
            return original(*args, **kwargs)

        return patched
