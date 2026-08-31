import json
import sys
from pathlib import Path

import pytest

from voicetotext.core.paths import AppPaths


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
    def test_should_default_to_appdata_voicetotext(self, monkeypatch):
        monkeypatch.setenv("APPDATA", "C:\\Users\\Test\\AppData\\Roaming")

        paths = AppPaths()

        assert paths.base == Path("C:\\Users\\Test\\AppData\\Roaming\\VoiceToText")


class TestAppPathsEnsureDirs:
    def test_should_create_base_and_models_directories(self, tmp_path):
        base = tmp_path / "VoiceToText"
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
