from __future__ import annotations

import os
import subprocess
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import winreg


_SHORTCUT_NAME = "VoiceToText (Dev).lnk"
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_RUN_VALUE = "VoiceToTextDev"
_CREATE_SHORTCUT_SCRIPT = """
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($env:VOICETOTEXT_SHORTCUT_PATH)
$shortcut.TargetPath = $env:VOICETOTEXT_LAUNCHER_PATH
$shortcut.WorkingDirectory = Split-Path $env:VOICETOTEXT_LAUNCHER_PATH
$shortcut.Description = 'VoiceToText development install'
$shortcut.Save()
$check = $shell.CreateShortcut($env:VOICETOTEXT_SHORTCUT_PATH)
[Console]::Out.Write($check.TargetPath)
""".strip()


def launcher_path() -> Path:
    return Path(sysconfig.get_path("scripts")).resolve() / "voicetotext-gui.exe"


def shortcut_path() -> Path:
    return (
        Path(os.environ["APPDATA"])
        / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        / _SHORTCUT_NAME
    )


def create_shortcut(
    target: Path | None = None,
    destination: Path | None = None,
) -> tuple[bool, str]:
    resolved_target = (target or launcher_path()).resolve()
    resolved_destination = destination or shortcut_path()
    if not resolved_target.is_file():
        return False, f"GUI launcher not found: {resolved_target}"
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    system_root = Path(os.environ["SystemRoot"])
    powershell = (
        system_root / "System32" / "WindowsPowerShell" / "v1.0"
        / "powershell.exe"
    )
    environment = os.environ.copy()
    environment["VOICETOTEXT_LAUNCHER_PATH"] = str(resolved_target)
    environment["VOICETOTEXT_SHORTCUT_PATH"] = str(resolved_destination)
    completed = subprocess.run(
        [
            str(powershell), "-NoProfile", "-NonInteractive",
            "-Command", _CREATE_SHORTCUT_SCRIPT,
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "PowerShell returned no details"
        return False, f"Shortcut creation failed: {detail}"
    if not resolved_destination.is_file():
        return False, f"Shortcut was not created: {resolved_destination}"
    if Path(completed.stdout.strip()).resolve() != resolved_target:
        return False, "Shortcut target verification failed"
    return True, f"Created {resolved_destination}"


def enable_autostart(target: Path | None = None) -> tuple[bool, str]:
    import winreg

    resolved = (target or launcher_path()).resolve()
    if not resolved.is_file():
        return False, f"GUI launcher not found: {resolved}"
    command = f'"{resolved}"'
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_READ | winreg.KEY_SET_VALUE,
        ) as key:
            existing = _read_registry_value(key)
            if existing is not None and _command_target(existing) != resolved:
                return False, f"VoiceToTextDev already points to: {existing}"
            winreg.SetValueEx(key, _RUN_VALUE, 0, winreg.REG_SZ, command)
    except OSError as exc:
        return False, f"Could not enable autostart: {exc}"
    return True, f"Autostart enabled for {resolved}"


def disable_autostart(target: Path | None = None) -> tuple[bool, str]:
    import winreg

    resolved = (target or launcher_path()).resolve()
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_READ | winreg.KEY_SET_VALUE,
        ) as key:
            existing = _read_registry_value(key)
            if existing is None:
                return True, "Autostart is already disabled"
            if _command_target(existing) != resolved:
                return False, f"VoiceToTextDev belongs to another install: {existing}"
            winreg.DeleteValue(key, _RUN_VALUE)
    except FileNotFoundError:
        return True, "Autostart is already disabled"
    except OSError as exc:
        return False, f"Could not disable autostart: {exc}"
    return True, f"Autostart disabled for {resolved}"


def _read_registry_value(key: winreg.HKEYType) -> str | None:
    import winreg

    try:
        value, _value_type = winreg.QueryValueEx(key, _RUN_VALUE)
    except FileNotFoundError:
        return None
    return str(value)


def _command_target(command: str) -> Path:
    stripped = command.strip()
    if stripped.startswith('"'):
        end = stripped.find('"', 1)
        raw_target = stripped[1:end] if end > 1 else stripped.strip('"')
    else:
        raw_target = stripped.split(maxsplit=1)[0]
    return Path(raw_target).resolve()
