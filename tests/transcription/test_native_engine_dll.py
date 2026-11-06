from pathlib import Path

import pytest

from skryb.transcription import native_engine_dll as finder


@pytest.fixture()
def places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    shipped = tmp_path / "site-packages" / "skryb" / "transcription" / "qasr.dll"
    dev = tmp_path / "checkout" / "native" / "engine" / "build" / "qasr.dll"
    monkeypatch.setattr(finder, "SHIPPED_DLL", shipped)
    monkeypatch.setattr(finder, "DEV_BUILD_DLL", dev)
    return shipped, dev


def _make(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ")
    return path


def test_an_installed_copy_uses_the_dll_shipped_in_its_wheel(places):
    shipped, dev = places
    _make(shipped)
    _make(dev)

    assert finder.native_engine_dll({}) == shipped


def test_a_source_checkout_uses_the_dll_build_sh_made(places):
    _, dev = places
    _make(dev)

    assert finder.native_engine_dll({}) == dev


def test_there_is_none_when_neither_exists(places):
    assert finder.native_engine_dll({}) is None


def test_qasr_dll_overrides_both(places, tmp_path):
    shipped, _ = places
    _make(shipped)
    chosen = _make(tmp_path / "my builds" / "qasr ü.dll")

    assert finder.native_engine_dll({"QASR_DLL": str(chosen)}) == chosen


def test_a_qasr_dll_that_does_not_exist_never_falls_back(places, tmp_path):
    shipped, _ = places
    _make(shipped)

    assert finder.native_engine_dll({"QASR_DLL": str(tmp_path / "missing.dll")}) is None


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_qasr_dll_counts_as_unset(places, blank):
    shipped, _ = places
    _make(shipped)

    assert finder.native_engine_dll({"QASR_DLL": blank}) == shipped


def test_the_parakeet_engine_is_found_the_same_way(tmp_path, monkeypatch):
    shipped = tmp_path / "site-packages" / "skryb" / "transcription" / "npk.dll"
    dev = _make(tmp_path / "checkout" / "native" / "parakeet" / "build" / "npk.dll")
    monkeypatch.setattr(finder, "SHIPPED_PARAKEET_DLL", shipped)
    monkeypatch.setattr(finder, "DEV_BUILD_PARAKEET_DLL", dev)

    assert finder.native_parakeet_dll({}) == dev
    assert finder.native_parakeet_dll({"NPK_DLL": str(tmp_path / "missing.dll")}) is None
    _make(shipped)
    assert finder.native_parakeet_dll({}) == shipped
    assert finder.native_parakeet_dll({"QASR_DLL": str(dev)}) == shipped


def test_the_diarizer_engine_is_found_the_same_way(tmp_path, monkeypatch):
    shipped = tmp_path / "site-packages" / "skryb" / "transcription" / "nd.dll"
    dev = _make(tmp_path / "checkout" / "native" / "diar" / "build" / "nd.dll")
    monkeypatch.setattr(finder, "SHIPPED_DIARIZER_DLL", shipped)
    monkeypatch.setattr(finder, "DEV_BUILD_DIARIZER_DLL", dev)

    assert finder.native_diarizer_dll({}) == dev
    assert finder.native_diarizer_dll({"ND_DLL": str(tmp_path / "missing.dll")}) is None
    _make(shipped)
    assert finder.native_diarizer_dll({}) == shipped


def test_the_nvidia_build_sits_beside_the_amd_one_and_is_asked_for_by_name(tmp_path, monkeypatch):
    shipped = tmp_path / "site-packages" / "skryb" / "transcription" / "npk.dll"
    monkeypatch.setattr(finder, "SHIPPED_PARAKEET_DLL", shipped)
    monkeypatch.setattr(finder, "DEV_BUILD_PARAKEET_DLL", tmp_path / "checkout" / "npk.dll")
    _make(shipped)

    assert finder.native_parakeet_dll({}, cuda=True) is None
    nvidia = _make(shipped.with_name("npk_cuda.dll"))
    assert finder.native_parakeet_dll({}, cuda=True) == nvidia
    assert finder.native_parakeet_dll({}) == shipped


def test_each_build_has_its_own_override(tmp_path, places):
    shipped, _ = places
    _make(shipped)
    _make(shipped.with_name("qasr_cuda.dll"))
    chosen = _make(tmp_path / "mine" / "qasr_cuda.dll")

    assert finder.native_engine_dll({"QASR_CUDA_DLL": str(chosen)}, cuda=True) == chosen
    assert finder.native_engine_dll({"QASR_CUDA_DLL": str(chosen)}) == shipped
    assert finder.native_engine_dll({"QASR_DLL": str(chosen)}, cuda=True) == shipped.with_name("qasr_cuda.dll")



def test_nvidias_cpu_diarizer_is_found_in_the_installed_wheel_then_the_fetched_build_and_the_variable_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    shipped = tmp_path / "site-packages" / "skryb" / "transcription" / "nemo-speech" / "nemo_speech_asr_c.dll"
    fetched = tmp_path / "checkout" / "native" / "nemo_speech" / "build" / "nemo_speech_asr_c.dll"
    chosen = _make(tmp_path / "Zoë's tools" / "nemo_speech_asr_c.dll")
    monkeypatch.setattr(finder, "SHIPPED_NEMO_SPEECH_DLL", shipped)
    monkeypatch.setattr(finder, "DEV_BUILD_NEMO_SPEECH_DLL", fetched)

    assert finder.nemo_speech_library({}) is None
    _make(fetched)
    assert finder.nemo_speech_library({}) == fetched
    _make(shipped)
    assert finder.nemo_speech_library({}) == shipped
    assert finder.nemo_speech_library({"NEMO_SPEECH_DLL": str(chosen)}) == chosen
    assert finder.nemo_speech_library({"NEMO_SPEECH_DLL": str(tmp_path / "gone.dll")}) is None
