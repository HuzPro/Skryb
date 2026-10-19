"""Phonon-2 on Fermion's CPU engine, against a fake ``fermion`` package.

The real engine and its 164 MB weights stay out of unit tests; the
manual real-weights check drives the adapter from a subprocess script.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    TranscriptionError,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription import phonon2_cpu_adapter
from voicetotext.transcription.phonon2_cpu_adapter import (
    ARCHIVE_FILENAME,
    Phonon2CpuAdapter,
    unpacked_model_dir,
)


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


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "models--FermionResearch--Phonon-2" / "snapshots" / "fb09063c"
    snapshot.mkdir(parents=True)
    (snapshot / ARCHIVE_FILENAME).write_bytes(b"zstd tar")
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
            {"profile": "five-value", "backend": "phonon2-five-value", "quiet": True},
        )]

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
