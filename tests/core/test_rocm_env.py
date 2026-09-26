import os
import shutil
import subprocess
import sys
import types

import pytest

from voicetotext.core.rocm_env import prepare_rocm_environment, target_family_for

_ENV = "ROCM_SDK_TARGET_FAMILY"
_OFFLOAD_ARCH = r"C:\venv\Scripts\offload-arch.exe"


@pytest.fixture
def family_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setting before deleting makes monkeypatch restore the absence afterwards.
    monkeypatch.setenv(_ENV, "placeholder")
    monkeypatch.delenv(_ENV)


def _install_rocm_sdk(monkeypatch: pytest.MonkeyPatch, families: list[str]) -> None:
    dist_info = types.ModuleType("rocm_sdk._dist_info")
    dist_info.AVAILABLE_TARGET_FAMILIES = families  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rocm_sdk._dist_info", dist_info)


def _refuse_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no process may run")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "check_output", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)


def _offload_arch_prints(
    monkeypatch: pytest.MonkeyPatch, output: str,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"args": args, **kwargs})
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(
        shutil, "which", lambda name, path=None: _OFFLOAD_ARCH,
    )
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_the_only_family_this_install_ships_is_chosen_without_running_anything(
    monkeypatch: pytest.MonkeyPatch, family_unset: None,
) -> None:
    _install_rocm_sdk(monkeypatch, ["custom"])
    _refuse_processes(monkeypatch)

    prepare_rocm_environment()

    assert os.environ[_ENV] == "custom"


def test_a_family_already_in_the_environment_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV, "gfx110X-all")
    _install_rocm_sdk(monkeypatch, ["custom"])
    _refuse_processes(monkeypatch)

    prepare_rocm_environment()

    assert os.environ[_ENV] == "gfx110X-all"


def test_an_install_without_rocm_leaves_the_environment_untouched(
    monkeypatch: pytest.MonkeyPatch, family_unset: None,
) -> None:
    monkeypatch.setitem(sys.modules, "rocm_sdk._dist_info", None)
    _refuse_processes(monkeypatch)

    prepare_rocm_environment()

    assert _ENV not in os.environ


def test_a_multi_family_install_asks_offload_arch_silently_and_maps_its_answer(
    monkeypatch: pytest.MonkeyPatch, family_unset: None,
) -> None:
    _install_rocm_sdk(monkeypatch, ["gfx110X-all", "gfx120X-all", "gfx1151"])
    calls = _offload_arch_prints(monkeypatch, "gfx1201\r\ngfx1036\r\n")

    prepare_rocm_environment()

    assert os.environ[_ENV] == "gfx120X-all"
    [call] = calls
    assert call["args"] == [_OFFLOAD_ARCH]
    assert call["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert call["stdin"] is subprocess.DEVNULL
    assert call["capture_output"] is True
    assert call["timeout"] == 5.0


def test_an_arch_no_shipped_family_covers_leaves_the_choice_to_rocm_sdk(
    monkeypatch: pytest.MonkeyPatch, family_unset: None,
) -> None:
    _install_rocm_sdk(monkeypatch, ["gfx110X-all", "gfx120X-all"])
    _offload_arch_prints(monkeypatch, "gfx803\n")

    prepare_rocm_environment()

    assert _ENV not in os.environ


def test_a_failing_offload_arch_never_raises(
    monkeypatch: pytest.MonkeyPatch, family_unset: None,
) -> None:
    _install_rocm_sdk(monkeypatch, ["gfx110X-all", "gfx120X-all"])
    monkeypatch.setattr(shutil, "which", lambda name, path=None: _OFFLOAD_ARCH)

    def fail(args: list[str], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, args, output="", stderr="no GPU")

    monkeypatch.setattr(subprocess, "run", fail)

    prepare_rocm_environment()

    assert _ENV not in os.environ


def test_a_missing_offload_arch_runs_nothing(
    monkeypatch: pytest.MonkeyPatch, family_unset: None,
) -> None:
    _install_rocm_sdk(monkeypatch, ["gfx110X-all", "gfx120X-all"])
    monkeypatch.setattr(shutil, "which", lambda name, path=None: None)
    _refuse_processes(monkeypatch)

    prepare_rocm_environment()

    assert _ENV not in os.environ


@pytest.mark.parametrize(
    ("output", "available", "expected"),
    [
        ("gfx1201\n", ("gfx110X-all", "gfx120X-all"), "gfx120X-all"),
        ("gfx1100\n", ("gfx110X-dgpu", "gfx110X-all"), "gfx110X-all"),
        ("gfx1103\n", ("gfx110X-igpu",), "gfx110X-igpu"),
        ("\r\ngfx942\r\n\r\ngfx942\r\n", ("gfx94X-dcgpu",), "gfx94X-dcgpu"),
        ("gfx1151\n", ("gfx110X-all", "gfx1151"), "gfx1151"),
        ("gfx803\ngfx1201\n", ("gfx120X-all",), "gfx120X-all"),
        ("gfx803\n", ("gfx110X-all",), None),
        ("", ("gfx110X-all",), None),
    ],
)
def test_arches_map_to_families_the_way_rocm_sdk_maps_them(
    output: str, available: tuple[str, ...], expected: str | None,
) -> None:
    assert target_family_for(output, available) == expected
