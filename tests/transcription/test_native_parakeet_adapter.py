import subprocess
import sys
from pathlib import Path

import pytest

from skryb.core.exceptions import TranscriptionError
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


def test_the_engine_and_its_packer_never_import_torch() -> None:
    probe = (
        "import sys, skryb.transcription.native_parakeet_adapter, skryb.transcription.parakeet_pack;"
        "from skryb.transcription.parakeet_pack import _bundled_tokenizer;"
        "assert _bundled_tokenizer().is_file();"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False"
