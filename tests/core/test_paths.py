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
