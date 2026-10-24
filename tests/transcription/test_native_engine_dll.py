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
