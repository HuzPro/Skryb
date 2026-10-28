import json
import sys
from pathlib import Path

import pytest

from skryb.core.paths import AppPaths, move_voicetotext_folders


class TestAppPathsResolution:
    def test_should_resolve_config_path(self, tmp_path):
        paths = AppPaths(base=tmp_path)

        assert paths.config_path == tmp_path / "config.json"

    def test_should_resolve_models_dir(self, tmp_path):
        paths = AppPaths(base=tmp_path)

        assert paths.models_dir == tmp_path / "models"

    def test_should_resolve_diagnostics_dir(self, tmp_path) -> None:
        paths = AppPaths(base=tmp_path)

        assert paths.diagnostics_dir == tmp_path / "diagnostics"

    def test_should_resolve_session_log_inside_active_session(
        self, tmp_path,
    ) -> None:
        paths = AppPaths(base=tmp_path)
        session_dir = tmp_path / "diagnostics" / "session"

        paths.activate_session(session_dir)

        assert paths.session_log_path == session_dir / "session.log"

    def test_should_reject_session_path_before_initialization(
        self, tmp_path,
    ) -> None:
        paths = AppPaths(base=tmp_path)

        with pytest.raises(RuntimeError, match="not been initialized"):
            _ = paths.session_log_path

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="APPDATA + backslash path semantics are Windows-only",
    )
    def test_should_default_to_appdata_skryb(self, monkeypatch):
        monkeypatch.setenv("APPDATA", "C:\\Users\\Test\\AppData\\Roaming")

        paths = AppPaths()

        assert paths.base == Path("C:\\Users\\Test\\AppData\\Roaming\\Skryb")


class TestAppPathsEnsureDirs:
    def test_should_create_base_and_models_directories(self, tmp_path):
        base = tmp_path / "Skryb"
        paths = AppPaths(base=base)

        paths.ensure_dirs()

        assert base.exists()
        assert (base / "models").exists()
        assert (base / "diagnostics").exists()

    def test_should_not_fail_if_dirs_already_exist(self, tmp_path):
        paths = AppPaths(base=tmp_path)
        paths.ensure_dirs()

        paths.ensure_dirs()


class TestAppPathsFirstRunDetection:
    def test_should_detect_first_run_when_no_config(self, tmp_path):
        paths = AppPaths(base=tmp_path)

        assert paths.is_first_run() is True

    def test_should_detect_existing_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        paths = AppPaths(base=tmp_path)

        assert paths.is_first_run() is False


class TestAppPathsDiagnosticsDirectory:
    def test_should_default_to_a_folder_under_the_app_base(
        self, tmp_path: Path,
    ) -> None:
        paths = AppPaths(base=tmp_path)

        assert paths.diagnostics_dir == tmp_path / "diagnostics"

    def test_should_use_a_configured_folder(self, tmp_path: Path) -> None:
        paths = AppPaths(base=tmp_path)

        paths.use_diagnostics_root(str(tmp_path / "elsewhere" / "logs"))

        assert paths.diagnostics_dir == tmp_path / "elsewhere" / "logs"

    @pytest.mark.parametrize("configured", ["", "   ", None, 7])
    def test_should_ignore_a_blank_or_non_text_folder(
        self, tmp_path: Path, configured: object,
    ) -> None:
        paths = AppPaths(base=tmp_path)

        paths.use_diagnostics_root(configured)

        assert paths.diagnostics_dir == tmp_path / "diagnostics"

    def test_should_create_the_configured_folder(self, tmp_path: Path) -> None:
        paths = AppPaths(base=tmp_path)
        paths.use_diagnostics_root(str(tmp_path / "elsewhere" / "logs"))

        paths.ensure_dirs()

        assert (tmp_path / "elsewhere" / "logs").is_dir()

    def test_should_fall_back_when_the_configured_folder_is_unusable(
        self, tmp_path: Path,
    ) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        paths = AppPaths(base=tmp_path)
        paths.use_diagnostics_root(str(blocker / "logs"))

        paths.ensure_dirs()

        assert paths.diagnostics_dir == tmp_path / "diagnostics"
        assert paths.diagnostics_dir.is_dir()

    def test_should_read_the_folder_from_the_saved_config(
        self, tmp_path: Path,
    ) -> None:
        paths = AppPaths(base=tmp_path)
        paths.config_path.parent.mkdir(parents=True, exist_ok=True)
        paths.config_path.write_text(
            json.dumps({"diagnostics": {"directory": str(tmp_path / "logs")}}),
            encoding="utf-8",
        )

        paths.load_configured_diagnostics_root()

        assert paths.diagnostics_dir == tmp_path / "logs"

    def test_should_default_when_no_config_is_saved(
        self, tmp_path: Path,
    ) -> None:
        paths = AppPaths(base=tmp_path)

        paths.load_configured_diagnostics_root()

        assert paths.diagnostics_dir == tmp_path / "diagnostics"


def _windows_profile(root: Path) -> dict[str, str]:
    """Roaming and Local AppData under a user name with a non-ASCII letter."""
    profile = root / "Zoë" / "AppData"
    return {"APPDATA": str(profile / "Roaming"), "LOCALAPPDATA": str(profile / "Local")}


def _fill(folder: Path, count: int) -> None:
    for index in range(count):
        session = folder / "diagnostics" / f"session-{index // 100:03d}"
        session.mkdir(parents=True, exist_ok=True)
        (session / f"clip-{index:04d}.json").write_text("{}", encoding="utf-8")
    (folder / "config.json").write_text('{"hotkey": "ctrl+shift+alt+m"}', encoding="utf-8")


class TestMoveVoiceToTextFolders:
    def test_moves_settings_and_models_with_every_file_to_the_skryb_folders(
        self, tmp_path: Path,
    ) -> None:
        environ = _windows_profile(tmp_path)
        roaming_old = Path(environ["APPDATA"]) / "VoiceToText"
        local_old = Path(environ["LOCALAPPDATA"]) / "VoiceToText"
        _fill(roaming_old, 2500)
        (local_old / "updates").mkdir(parents=True)
        (local_old / "updates" / "Skryb-Setup-0.2.0.exe").write_bytes(b"\x00" * 4096)

        move_voicetotext_folders(environ)

        roaming_new = Path(environ["APPDATA"]) / "Skryb"
        assert not roaming_old.exists() and not local_old.exists()
        assert len(list((roaming_new / "diagnostics").rglob("*.json"))) == 2500
        assert (roaming_new / "config.json").read_text(encoding="utf-8") == '{"hotkey": "ctrl+shift+alt+m"}'
        assert (Path(environ["LOCALAPPDATA"]) / "Skryb" / "updates" / "Skryb-Setup-0.2.0.exe").stat().st_size == 4096

    def test_leaves_both_folders_alone_when_skryb_already_has_one(
        self, tmp_path: Path,
    ) -> None:
        environ = _windows_profile(tmp_path)
        old = Path(environ["APPDATA"]) / "VoiceToText"
        new = Path(environ["APPDATA"]) / "Skryb"
        _fill(old, 3)
        new.mkdir(parents=True)
        (new / "config.json").write_text("{}", encoding="utf-8")

        move_voicetotext_folders(environ)

        assert (old / "config.json").exists()
        assert (new / "config.json").read_text(encoding="utf-8") == "{}"

    def test_does_nothing_on_a_computer_that_never_had_voicetotext(
        self, tmp_path: Path,
    ) -> None:
        environ = _windows_profile(tmp_path)

        move_voicetotext_folders(environ)

        assert not (Path(environ["APPDATA"]) / "Skryb").exists()

    def test_skips_appdata_variables_that_are_missing_or_blank(
        self, tmp_path: Path,
    ) -> None:
        local_old = tmp_path / "Local" / "VoiceToText"
        local_old.mkdir(parents=True)

        move_voicetotext_folders({"APPDATA": "", "LOCALAPPDATA": str(tmp_path / "Local")})

        assert (tmp_path / "Local" / "Skryb").is_dir()

    def test_raises_when_a_running_voicetotext_holds_the_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        environ = _windows_profile(tmp_path)
        _fill(Path(environ["APPDATA"]) / "VoiceToText", 1)

        def locked(self: Path, target: Path) -> Path:
            raise PermissionError(5, "Access is denied", str(self))

        monkeypatch.setattr(Path, "rename", locked)

        with pytest.raises(PermissionError):
            move_voicetotext_folders(environ)

    def test_accepts_a_folder_that_another_copy_moved_a_moment_earlier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        environ = _windows_profile(tmp_path)
        _fill(Path(environ["APPDATA"]) / "VoiceToText", 1)
        real_rename = Path.rename

        def beaten_to_it(self: Path, target: Path) -> Path:
            real_rename(self, target)
            raise FileNotFoundError(2, "The system cannot find the file specified", str(self))

        monkeypatch.setattr(Path, "rename", beaten_to_it)

        move_voicetotext_folders(environ)

        assert (Path(environ["APPDATA"]) / "Skryb" / "config.json").exists()
