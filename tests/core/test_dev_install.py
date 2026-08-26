import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voicetotext.core.dev_install import (
    _command_target,
    create_shortcut,
    disable_autostart,
    enable_autostart,
)


class FakeRegistryKey:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def __enter__(self) -> "FakeRegistryKey":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None


def _fake_winreg(key: FakeRegistryKey) -> SimpleNamespace:
    def query(_key: FakeRegistryKey, name: str) -> tuple[str, int]:
        if name not in key.values:
            raise FileNotFoundError(name)
        return key.values[name], 1

    def set_value(
        _key: FakeRegistryKey,
        name: str,
        _reserved: int,
        _kind: int,
        value: str,
    ) -> None:
        key.values[name] = value

    def delete(_key: FakeRegistryKey, name: str) -> None:
        del key.values[name]

    return SimpleNamespace(
        HKEY_CURRENT_USER=1,
        KEY_READ=1,
        KEY_SET_VALUE=2,
        REG_SZ=1,
        CreateKeyEx=lambda *_args: key,
        OpenKey=lambda *_args: key,
        QueryValueEx=query,
        SetValueEx=set_value,
        DeleteValue=delete,
    )


class TestShortcutCreation:
    def test_should_pass_paths_only_through_environment(self, tmp_path) -> None:
        launcher = tmp_path / "Scripts" / "voicetotext-gui.exe"
        launcher.parent.mkdir()
        launcher.write_bytes(b"")
        destination = tmp_path / "Programs" / "VoiceToText (Dev).lnk"
        system_root = tmp_path / "Windows"

        def run(
            command: list[str],
            *,
            capture_output: bool,
            text: bool,
            env: dict[str, str],
            check: bool,
        ) -> subprocess.CompletedProcess[str]:
            destination.write_bytes(b"shortcut")
            assert str(launcher.resolve()) not in command[-1]
            assert env["VOICETOTEXT_LAUNCHER_PATH"] == str(launcher.resolve())
            assert env["VOICETOTEXT_SHORTCUT_PATH"] == str(destination)
            return subprocess.CompletedProcess(command, 0, str(launcher.resolve()), "")

        with (
            patch.dict(os.environ, {"SystemRoot": str(system_root)}),
            patch("voicetotext.core.dev_install.subprocess.run", side_effect=run),
        ):
            succeeded, _message = create_shortcut(launcher, destination)

        assert succeeded is True

    def test_should_reject_missing_gui_launcher(self, tmp_path) -> None:
        succeeded, message = create_shortcut(
            tmp_path / "missing.exe",
            tmp_path / "shortcut.lnk",
        )

        assert succeeded is False
        assert "not found" in message


class TestAutostartOwnership:
    def test_should_round_trip_owned_registry_value(self, tmp_path) -> None:
        launcher = tmp_path / "voicetotext-gui.exe"
        launcher.write_bytes(b"")
        key = FakeRegistryKey()
        winreg = _fake_winreg(key)

        with patch.dict(sys.modules, {"winreg": winreg}):
            enabled, _message = enable_autostart(launcher)
            disabled, _message = disable_autostart(launcher)

        assert enabled is True
        assert disabled is True
        assert key.values == {}

    def test_should_refuse_to_overwrite_another_install(self, tmp_path) -> None:
        launcher = tmp_path / "current" / "voicetotext-gui.exe"
        launcher.parent.mkdir()
        launcher.write_bytes(b"")
        key = FakeRegistryKey()
        key.values["VoiceToTextDev"] = '"C:\\other\\voicetotext-gui.exe"'

        with patch.dict(sys.modules, {"winreg": _fake_winreg(key)}):
            succeeded, message = enable_autostart(launcher)

        assert succeeded is False
        assert "other" in message

    def test_should_resolve_quoted_command_target(self, tmp_path) -> None:
        target = tmp_path / "Scripts" / "voicetotext-gui.exe"

        assert _command_target(f'"{target}" --ignored') == target.resolve()
