from pathlib import Path

import pytest

from voicetotext.transcription.parakeet_loader import ParakeetModelLoader


class TestParakeetModelLoader:
    def test_should_load_fp32_on_cpu_and_redirect_downloads_into_model_dir(
        self, monkeypatch,
    ):
        # On CPU, bf16 weights crash the fp32 joint matmul (and CPU has no
        # fast bf16 path), so the loader must request fp32 there.
        import nano_parakeet
        import torch

        seen: dict = {}

        def recording_download(*args, **kwargs):
            seen["download_kwargs"] = kwargs
            return "local/path"

        monkeypatch.setattr(nano_parakeet, "hf_hub_download", recording_download)

        captured: dict = {}

        def fake_from_pretrained(model_name, device, dtype):
            captured.update(model_name=model_name, device=device, dtype=dtype)
            # nano-parakeet downloads through the (now patched) hook.
            nano_parakeet.hf_hub_download("repo", filename="model.bin")
            return "the-model"

        monkeypatch.setattr(nano_parakeet, "from_pretrained", fake_from_pretrained)

        loader = ParakeetModelLoader(
            repo_id="nvidia/parakeet-tdt-0.6b-v3", model_dir=Path("X:/models"),
        )
        model = loader.load("cpu")

        assert model == "the-model"
        assert captured["model_name"] == "nvidia/parakeet-tdt-0.6b-v3"
        assert captured["device"] == "cpu"
        assert captured["dtype"] is torch.float32
        assert seen["download_kwargs"]["cache_dir"] == str(Path("X:/models"))
        assert seen["download_kwargs"]["local_files_only"] is True
        # The hook is restored after loading.
        assert nano_parakeet.hf_hub_download is recording_download

    def test_should_load_bf16_on_cuda(self, monkeypatch):
        # GPU keeps bf16: the chunked sweep showed it matches fp32 text on
        # gfx1200 while cutting latency and VRAM roughly in half.
        import nano_parakeet
        import torch

        monkeypatch.setattr(
            nano_parakeet, "hf_hub_download", lambda *a, **k: "local/path",
        )

        captured: dict = {}

        def fake_from_pretrained(model_name, device, dtype):
            captured.update(device=device, dtype=dtype)
            return "the-model"

        monkeypatch.setattr(nano_parakeet, "from_pretrained", fake_from_pretrained)

        loader = ParakeetModelLoader(
            repo_id="nvidia/parakeet-tdt-0.6b-v3", model_dir=Path("X:/models"),
        )
        loader.load("cuda")

        assert captured["device"] == "cuda"
        assert captured["dtype"] is torch.bfloat16

    def test_should_restore_download_hook_when_load_fails(self, monkeypatch):
        import nano_parakeet

        sentinel = nano_parakeet.hf_hub_download

        def boom(model_name, device, dtype):
            raise RuntimeError("load failed")

        monkeypatch.setattr(nano_parakeet, "from_pretrained", boom)

        loader = ParakeetModelLoader(repo_id="r", model_dir=Path("X:/m"))
        with pytest.raises(RuntimeError, match="load failed"):
            loader.load("cpu")

        assert nano_parakeet.hf_hub_download is sentinel


def _install_snapshot(models_dir: Path, repo_id: str, *filenames: str) -> Path:
    repo_dir = models_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    for filename in filenames:
        (snapshot / filename).write_bytes(b"")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("abc123")
    return snapshot


class TestTransformersLayoutSnapshots:
    def test_a_safetensors_snapshot_without_a_nemo_loads_through_the_transformers_layout_loader(
        self, monkeypatch, tmp_path,
    ):
        import nano_parakeet
        import torch

        from voicetotext.transcription import parakeet_loader

        snapshot = _install_snapshot(
            tmp_path, "moondream/parakeet-ultra",
            "config.json", "model.safetensors",
        )
        captured: dict = {}

        def fake_load_hf(nano_parakeet_module, torch_module, path, device, dtype):
            captured.update(path=path, device=device, dtype=dtype)
            return "ultra"

        def refuse_nemo_loader(*args, **kwargs):
            raise AssertionError("nano-parakeet's .nemo loader must not run")

        monkeypatch.setattr(parakeet_loader, "load_hf_parakeet", fake_load_hf)
        monkeypatch.setattr(nano_parakeet, "from_pretrained", refuse_nemo_loader)

        model = ParakeetModelLoader(
            repo_id="moondream/parakeet-ultra", model_dir=tmp_path,
        ).load("cuda")

        assert model == "ultra"
        assert captured == {
            "path": snapshot / "model.safetensors",
            "device": "cuda",
            "dtype": torch.bfloat16,
        }

    def test_a_nemo_snapshot_keeps_using_nano_parakeets_own_loader(
        self, monkeypatch, tmp_path,
    ):
        import nano_parakeet

        from voicetotext.transcription import parakeet_loader

        _install_snapshot(
            tmp_path, "nvidia/parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3.nemo",
        )

        def refuse_hf_loader(*args, **kwargs):
            raise AssertionError("the transformers-layout loader must not run")

        monkeypatch.setattr(parakeet_loader, "load_hf_parakeet", refuse_hf_loader)
        monkeypatch.setattr(
            nano_parakeet, "from_pretrained",
            lambda model_name, device, dtype: "v3",
        )

        model = ParakeetModelLoader(
            repo_id="nvidia/parakeet-tdt-0.6b-v3", model_dir=tmp_path,
        ).load("cpu")

        assert model == "v3"
