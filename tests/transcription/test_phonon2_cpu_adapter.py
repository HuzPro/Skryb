"""Phonon on Fermion's CPU engine, against a fake ``fermion`` package.

The real engine and its 164 MB weights stay out of unit tests; the
manual real-weights check drives the adapter from a subprocess script.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from skryb.core.exceptions import (
    EngineDependencyMissingError,
    TranscriptionError,
)
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription import phonon2_cpu_adapter
from skryb.transcription.phonon2_cpu_adapter import (
    Phonon2CpuAdapter,
    unpacked_model_dir,
)

ARCHIVE_FILENAME = "phonon-2.5-en.bps.tar.zst"
# The "artifact" block of FermionResearch/Phonon-2.5-English's config.json at the pinned commit.
_ARTIFACT = {
    "profile": "five-value-25-en",
    "backend": "phonon2-five-value",
    "filename": ARCHIVE_FILENAME,
    "download_bytes": 163693770,
    "container": "model.fermion",
}


class _FakeSpeech:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.heard: list[np.ndarray] = []

    def transcribe_array(self, audio: np.ndarray) -> tuple[str, float, float]:
        self.heard.append(audio)
        return self.reply, 0.01, len(audio) / 16000


class _FakeFermion:
    """Stands in for ``fermion._speech.engine_phonon2_cpu`` and ``.fetch``."""

    def __init__(self, reply: str = "Testing one two three.") -> None:
        self.speech = _FakeSpeech(reply)
        self.unpacked: list[tuple[Path, Path]] = []
        self.loads: list[tuple[Path, dict]] = []
        self.refusal: str | None = None

    def _unpack(self, archive: Path, dest: Path) -> int:
        self.unpacked.append((archive, dest))
        # Fermion replaces whatever sits at *dest* with its finished staging folder.
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.fermion").write_bytes(b"packed")
        return 1

    def load(self, model_dir: Path, **kwargs: object) -> _FakeSpeech:
        if self.refusal is not None:
            raise SystemExit(self.refusal)
        self.loads.append((model_dir, kwargs))
        return self.speech


@pytest.fixture
def fermion(monkeypatch: pytest.MonkeyPatch) -> _FakeFermion:
    fake = _FakeFermion()
    package = ModuleType("fermion")
    speech = ModuleType("fermion._speech")
    speech.engine_phonon2_cpu = SimpleNamespace(load=fake.load)
    speech.fetch = SimpleNamespace(_unpack=fake._unpack)
    package._speech = speech
    monkeypatch.setitem(sys.modules, "fermion", package)
    monkeypatch.setitem(sys.modules, "fermion._speech", speech)
    monkeypatch.setenv("FERMION_CPU_THREADS", "6")
    return fake


def _snapshot(tmp_path: Path, artifact: dict | None = None) -> Path:
    snapshot = tmp_path / "models--FermionResearch--Phonon-2.5-English" / "snapshots" / "da200358"
    snapshot.mkdir(parents=True)
    (snapshot / ARCHIVE_FILENAME).write_bytes(b"zstd tar")
    config = {"model_id": "FermionResearch/Phonon-2.5-English", "artifact": artifact or _ARTIFACT}
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return snapshot


def _loaded(tmp_path: Path) -> Phonon2CpuAdapter:
    adapter = Phonon2CpuAdapter(_snapshot(tmp_path))
    adapter.load_model()
    return adapter


def _speech(seconds: float, *, value: float = 0.1) -> AudioSegment:
    return AudioSegment(
        data=np.full(int(seconds * 16000), value, dtype=np.float32), sample_rate=16000,
    )


class TestLoading:
    def test_the_first_load_unpacks_the_archive_inside_its_own_snapshot(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        snapshot = _snapshot(tmp_path)

        Phonon2CpuAdapter(snapshot).load_model()

        assert fermion.unpacked == [(snapshot / ARCHIVE_FILENAME, snapshot / ".phonon2")]
        assert fermion.loads == [(
            unpacked_model_dir(snapshot),
            {"profile": "five-value-25-en", "backend": "phonon2-five-value", "quiet": True},
        )]

    def test_phonon_2s_own_archive_and_profile_come_from_its_config_too(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        snapshot = _snapshot(tmp_path, {
            "profile": "five-value", "backend": "phonon2-five-value", "filename": "phonon-2.bps.tar.zst",
        })

        Phonon2CpuAdapter(snapshot).load_model()

        assert fermion.unpacked == [(snapshot / "phonon-2.bps.tar.zst", snapshot / ".phonon2")]
        assert fermion.loads[0][1]["profile"] == "five-value"

    @pytest.mark.parametrize("config", ["", "{not json", '{"model_id": "x"}', '{"artifact": {"profile": "p"}}'])
    def test_a_missing_or_broken_config_is_a_clear_error(
        self, fermion: _FakeFermion, tmp_path: Path, config: str,
    ) -> None:
        snapshot = _snapshot(tmp_path)
        (snapshot / "config.json").write_text(config, encoding="utf-8")

        with pytest.raises(TranscriptionError, match="config.json"):
            Phonon2CpuAdapter(snapshot).load_model()
        assert fermion.unpacked == []

    def test_later_loads_reuse_the_unpacked_model_without_unpacking_again(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        snapshot = _snapshot(tmp_path)

        for _ in range(3):
            Phonon2CpuAdapter(snapshot).load_model()

        assert len(fermion.unpacked) == 1
        assert len(fermion.loads) == 3

    def test_a_folder_left_without_the_model_file_is_unpacked_again(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        snapshot = _snapshot(tmp_path)
        (snapshot / ".phonon2").mkdir()
        (snapshot / ".phonon2" / "config.json").write_text("{}")

        Phonon2CpuAdapter(snapshot).load_model()

        assert len(fermion.unpacked) == 1

    def test_a_refusal_from_fermion_becomes_a_transcription_error_instead_of_ending_the_thread(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        fermion.refusal = "Phonon-2 CPU load: missing ['encoder.layers.0.weight']"
        adapter = Phonon2CpuAdapter(_snapshot(tmp_path))

        with pytest.raises(TranscriptionError, match="missing"):
            adapter.load_model()

        assert adapter.is_model_loaded() is False

    def test_a_missing_fermion_package_names_the_install_line(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setitem(sys.modules, "fermion", None)

        with pytest.raises(EngineDependencyMissingError, match="pip install fermion-research"):
            Phonon2CpuAdapter(_snapshot(tmp_path)).load_model()


class TestTranscribing:
    def test_a_clip_is_handed_over_whole_and_the_text_is_trimmed(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        fermion.speech.reply = "  Voice to text logs.\n"
        adapter = _loaded(tmp_path)
        # Ten minutes: Fermion windows long audio at pauses by itself.
        audio = _speech(600)

        result = adapter.transcribe(audio)

        assert result.text == "Voice to text logs."
        assert result.language == "en"
        assert len(fermion.speech.heard) == 1
        assert fermion.speech.heard[0] is audio.data

    def test_silence_comes_back_as_empty_text(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        fermion.speech.reply = "   "

        result = _loaded(tmp_path).transcribe(_speech(2, value=0.0))

        assert result.text == ""

    def test_non_ascii_text_passes_through_unchanged(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        fermion.speech.reply = "Café naïve, Zoë — 東京 at 5 °C"

        result = _loaded(tmp_path).transcribe(_speech(3))

        assert result.text == "Café naïve, Zoë — 東京 at 5 °C"

    def test_a_single_sample_clip_still_reaches_the_engine(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        _loaded(tmp_path).transcribe(AudioSegment(data=np.zeros(1, dtype=np.float32), sample_rate=16000))

        assert fermion.speech.heard[0].shape == (1,)

    def test_transcribing_before_loading_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TranscriptionError, match="not loaded"):
            Phonon2CpuAdapter(_snapshot(tmp_path)).transcribe(_speech(1))

    def test_disposing_lets_go_of_the_model(
        self, fermion: _FakeFermion, tmp_path: Path,
    ) -> None:
        adapter = _loaded(tmp_path)

        adapter.dispose()
        adapter.dispose()

        assert adapter.is_model_loaded() is False


class TestThreadCountOnWindows:
    """Fermion's own rule, worked out without the PowerShell call it makes."""

    @pytest.fixture(autouse=True)
    def _on_windows_with_no_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(phonon2_cpu_adapter.sys, "platform", "win32")
        # Empty rather than deleted, so monkeypatch restores whatever was there.
        monkeypatch.setenv("FERMION_CPU_THREADS", "")

    @staticmethod
    def _threads_for(
        monkeypatch: pytest.MonkeyPatch, physical: int | None, logical: int | None,
    ) -> str:
        psutil = ModuleType("psutil")
        psutil.cpu_count = lambda logical=True: physical
        monkeypatch.setitem(sys.modules, "psutil", psutil)
        monkeypatch.setattr(phonon2_cpu_adapter.os, "cpu_count", lambda: logical)
        phonon2_cpu_adapter._set_thread_count_without_powershell()
        return phonon2_cpu_adapter.os.environ["FERMION_CPU_THREADS"]

    @pytest.mark.parametrize(
        ("physical", "logical", "expected"),
        [
            (6, 6, "6"),      # Core 5 320: two P-cores and four LP E-cores
            (6, 12, "6"),     # Ryzen 5 7600: SMT siblings slow the kernels
            (5, 10, "10"),    # below six cores every logical CPU helps
            (4, 4, "4"),
            (1, 1, "1"),
            (16, 32, "16"),
            (32, 64, "16"),   # Fermion never goes past sixteen
            (None, 8, "8"),   # physical count unknown
            (None, None, "1"),
        ],
    )
    def test_threads_follow_fermions_rule(
        self, monkeypatch: pytest.MonkeyPatch, physical: int | None, logical: int | None, expected: str,
    ) -> None:
        assert self._threads_for(monkeypatch, physical, logical) == expected

    def test_a_thread_count_the_user_already_set_is_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FERMION_CPU_THREADS", "3")

        phonon2_cpu_adapter._set_thread_count_without_powershell()

        assert phonon2_cpu_adapter.os.environ["FERMION_CPU_THREADS"] == "3"


class _FakeContainerReader:
    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self.tensors, self.read = tensors, []

    def read_container(self, path: str, threads: int) -> tuple[dict[str, np.ndarray], dict]:
        self.read.append(path)
        return self.tensors, {}


class TestDenseWeights:
    """Phonon's container as dense fp32 weights, the way the GPU and NPU engines take Parakeet's."""

    @pytest.fixture
    def reader(self, monkeypatch: pytest.MonkeyPatch) -> _FakeContainerReader:
        layer = "encoder.layers.0."
        fake = _FakeContainerReader({
            layer + "conv.pointwise_conv1.weight": np.ones((2048, 1024), dtype=np.float16),
            layer + "conv.pointwise_conv2.weight": np.ones((1024, 1024), dtype=np.float16),
            layer + "conv.depthwise_conv.weight": np.ones((1024, 1, 9), dtype=np.float16),
            layer + "feed_forward1.linear1.weight": np.full((4096, 1024), 0.5, dtype=np.float16),
            layer + "conv.batch_norm.num_batches_tracked": np.array(np.inf, dtype=np.float16),
        })
        package, speech = ModuleType("fermion"), ModuleType("fermion._speech")
        speech._engine_phonon2 = SimpleNamespace(load=lambda name: fake if name == "fermion_container" else None)
        package._speech = speech
        monkeypatch.setitem(sys.modules, "fermion", package)
        monkeypatch.setitem(sys.modules, "fermion._speech", speech)
        return fake

    def test_every_tensor_comes_back_dense_in_fp32(self, reader: _FakeContainerReader, tmp_path: Path) -> None:
        dense = phonon2_cpu_adapter.dense_weights(tmp_path / "model.fermion")

        assert {array.dtype for array in dense.values()} == {np.dtype(np.float32)}
        assert dense["encoder.layers.0.feed_forward1.linear1.weight"].shape == (4096, 1024)
        assert reader.read == [str(tmp_path / "model.fermion")]

    def test_pointwise_convs_get_back_their_kernel_axis(self, reader: _FakeContainerReader, tmp_path: Path) -> None:
        dense = phonon2_cpu_adapter.dense_weights(tmp_path / "model.fermion")

        assert dense["encoder.layers.0.conv.pointwise_conv1.weight"].shape == (2048, 1024, 1)
        assert dense["encoder.layers.0.conv.pointwise_conv2.weight"].shape == (1024, 1024, 1)
        assert dense["encoder.layers.0.conv.depthwise_conv.weight"].shape == (1024, 1, 9)

    def test_the_batch_norm_counter_stored_as_inf_is_left_out(self, reader: _FakeContainerReader, tmp_path: Path) -> None:
        dense = phonon2_cpu_adapter.dense_weights(tmp_path / "model.fermion")

        assert not [name for name in dense if name.endswith("num_batches_tracked")]
