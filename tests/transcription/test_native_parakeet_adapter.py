import json
import subprocess
import sys
from pathlib import Path

import pytest

from skryb.core.exceptions import TranscriptionError
from skryb.transcription import native_parakeet_adapter as adapter_module
from skryb.transcription.native_parakeet_adapter import NativeParakeetAdapter, packed_model_path


def test_the_packed_file_sits_in_the_snapshot_it_came_from(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "73175eb"

    assert packed_model_path(snapshot / "model.safetensors") == snapshot / ".npk" / "parakeet.bin"


def test_a_bark_model_packs_beside_its_unpacked_weights_not_inside_them(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "85db2ad"

    assert packed_model_path(snapshot / ".bark" / "model.safetensors") == snapshot / ".npk" / "parakeet.bin"


def test_loading_a_model_that_is_not_downloaded_says_so(tmp_path: Path) -> None:
    adapter = NativeParakeetAdapter(tmp_path, "moondream/parakeet-ultra", dll_path=tmp_path / "npk.dll")

    with pytest.raises(TranscriptionError, match="moondream/parakeet-ultra isn't downloaded"):
        adapter.load_model()
    assert adapter.is_model_loaded() is False


def test_transcribing_before_loading_is_refused(tmp_path: Path) -> None:
    adapter = NativeParakeetAdapter(tmp_path, "moondream/parakeet-ultra")

    with pytest.raises(TranscriptionError, match="not loaded"):
        adapter.transcribe(object())


def _phonon_download(models: Path, *, packed: bool = False) -> Path:
    revision = "da200358d638886bc96c758c672ad5568828b119"
    repo = models / "models--FermionResearch--Phonon-2.5-English"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision)
    artifact = {"filename": "phonon-2.5-en.bps.tar.zst", "profile": "five-value-25-en", "backend": "phonon2-five-value"}
    (snapshot / "config.json").write_text(json.dumps({"artifact": artifact}), encoding="utf-8")
    if packed:
        (snapshot / ".npk").mkdir()
        (snapshot / ".npk" / "parakeet.bin").write_bytes(b"NPK00001")
    return snapshot


class TestPhononOnTheNativeEngine:
    """Phonon is Parakeet v3's network: its container is expanded and packed like Parakeet's own weights."""

    @pytest.fixture
    def packs(self, monkeypatch: pytest.MonkeyPatch) -> list:
        packed = []
        monkeypatch.setattr(adapter_module, "_phonon_state", lambda snapshot: {"from": snapshot})
        monkeypatch.setattr(
            "skryb.transcription.parakeet_pack.pack_state", lambda state, out: packed.append((state, out)),
        )
        return packed

    def test_the_first_load_packs_it_from_its_own_snapshot(self, tmp_path: Path, packs: list) -> None:
        snapshot = _phonon_download(tmp_path)
        adapter = NativeParakeetAdapter(tmp_path, "FermionResearch/Phonon-2.5-English", dll_path=tmp_path / "npk.dll")

        with pytest.raises(TranscriptionError, match="native/parakeet/build.sh"):
            adapter.load_model()

        assert packs == [({"from": snapshot}, snapshot / ".npk" / "parakeet.bin")]

    def test_a_later_load_reuses_the_packed_file(self, tmp_path: Path, packs: list) -> None:
        _phonon_download(tmp_path, packed=True)
        adapter = NativeParakeetAdapter(tmp_path, "FermionResearch/Phonon-2.5-English", dll_path=tmp_path / "npk.dll")

        with pytest.raises(TranscriptionError, match="native/parakeet/build.sh"):
            adapter.load_model()

        assert packs == []

    def test_a_snapshot_with_no_phonon_config_is_reported_as_not_downloaded(self, tmp_path: Path, packs: list) -> None:
        snapshot = _phonon_download(tmp_path)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        adapter = NativeParakeetAdapter(tmp_path, "FermionResearch/Phonon-2.5-English", dll_path=tmp_path / "npk.dll")

        with pytest.raises(TranscriptionError, match="isn't downloaded"):
            adapter.load_model()
        assert packs == []


def test_the_engine_and_its_packer_never_import_torch() -> None:
    probe = (
        "import sys, skryb.transcription.native_parakeet_adapter, skryb.transcription.parakeet_pack;"
        "from skryb.transcription.parakeet_pack import _bundled_tokenizer;"
        "assert _bundled_tokenizer().is_file();"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False"
