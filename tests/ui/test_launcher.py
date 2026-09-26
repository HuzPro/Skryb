"""Opening the UI shell: which exe, what it is told, and what the user hears when there is none."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.state_machine import AppStateMachine
from voicetotext.tray.app import SystemTrayApp
from voicetotext.ui.launcher import (
    SHELL_NOT_BUILT,
    SHELL_WOULD_NOT_START,
    UI_EXE_ENV,
    ShellLauncher,
    shell_candidates,
)

_PACKAGE = Path(__file__).resolve().parents[2] / "voicetotext"


@dataclass(frozen=True)
class _Address:
    port: int = 51_234
    token: str = "tok-_3n"


class _Shell:
    def __init__(self) -> None:
        self.exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code


class _Spawner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.shells: list[_Shell] = []

    def __call__(self, args: list[str]) -> _Shell:
        self.commands.append(args)
        self.shells.append(_Shell())
        return self.shells[-1]


def _exe(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / "voicetotext-ui.exe"
    exe.write_bytes(b"MZ")
    return exe


def test_the_shell_is_looked_for_in_the_configured_path_then_the_repo_build_then_the_package() -> None:
    repo_build = _PACKAGE.parent / "ui" / "src-tauri" / "target" / "release" / "voicetotext-ui.exe"
    packaged = _PACKAGE / "ui-bin" / "voicetotext-ui.exe"

    assert shell_candidates({UI_EXE_ENV: r"D:\Builds\UI ✓\voicetotext-ui.exe"}) == [
        Path(r"D:\Builds\UI ✓\voicetotext-ui.exe"), repo_build, packaged,
    ]
    assert shell_candidates({UI_EXE_ENV: "   "}) == [repo_build, packaged]
    assert shell_candidates({}) == [repo_build, packaged]


def test_the_first_shell_found_opens_on_the_page_with_the_servers_port_and_token(tmp_path: Path) -> None:
    spawn = _Spawner()
    found = _exe(tmp_path / "repo build")
    later = _exe(tmp_path / "packaged")
    launcher = ShellLauncher(
        _Address(), MagicMock(), spawn=spawn,
        candidates=lambda: [tmp_path / "not built" / "voicetotext-ui.exe", found, later],
    )

    assert launcher.is_running() is False
    assert launcher.open("history") is True

    assert spawn.commands == [
        [str(found), "--port", "51234", "--token", "tok-_3n", "--page", "history"],
    ]
    assert launcher.is_running() is True
    spawn.shells[0].exit_code = 0
    assert launcher.is_running() is False


def test_asking_again_while_the_shell_runs_hands_it_the_new_page(tmp_path: Path) -> None:
    spawn = _Spawner()
    exe = _exe(tmp_path)
    launcher = ShellLauncher(_Address(), MagicMock(), spawn=spawn, candidates=lambda: [exe])

    launcher.open("settings")
    launcher.open("history")

    assert [command[-1] for command in spawn.commands] == ["settings", "history"]


def test_with_no_shell_built_the_tray_says_how_to_build_it_and_nothing_starts(tmp_path: Path) -> None:
    icon = MagicMock()
    icon.run.side_effect = lambda setup: setup(icon)
    bus = SimpleEventBus()
    app = SystemTrayApp(
        use_case=MagicMock(), state_machine=AppStateMachine(bus), event_bus=bus,
        hotkey_listener=MagicMock(), tray_factory=lambda **_kwargs: icon,
    )
    app.run()
    spawn = _Spawner()
    launcher = ShellLauncher(
        _Address(), app.notify, spawn=spawn,
        candidates=lambda: [tmp_path / "a" / "voicetotext-ui.exe", tmp_path / "b" / "voicetotext-ui.exe"],
    )

    assert launcher.open("settings") is False

    icon.notify.assert_called_once_with("VoiceToText", SHELL_NOT_BUILT)
    assert spawn.commands == []
    assert SHELL_NOT_BUILT == "The settings window is not built. Run scripts\\build_ui.ps1."


def test_a_shell_that_will_not_start_is_reported_and_leaves_nothing_running(tmp_path: Path) -> None:
    notices: list[str] = []
    exe = _exe(tmp_path)

    def refuse(_args: list[str]) -> _Shell:
        raise PermissionError(13, "Access is denied", str(exe))

    launcher = ShellLauncher(_Address(), notices.append, spawn=refuse, candidates=lambda: [exe])

    assert launcher.open("settings") is False
    assert notices == [SHELL_WOULD_NOT_START]
    assert launcher.is_running() is False


def test_a_tray_balloon_before_the_icon_exists_is_dropped_rather_than_failing() -> None:
    bus = SimpleEventBus()
    app = SystemTrayApp(
        use_case=MagicMock(), state_machine=AppStateMachine(bus), event_bus=bus,
        hotkey_listener=MagicMock(), tray_factory=MagicMock(),
    )

    app.notify("The settings window is not built.")
