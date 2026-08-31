import os
from pathlib import Path


class AppPaths:

    def __init__(self, base: Path | None = None) -> None:
        if base is None:
            base = Path(os.environ["APPDATA"]) / "VoiceToText"
        self.base = base
        self._session_dir: Path | None = None

    @property
    def config_path(self) -> Path:
        return self.base / "config.json"

    @property
    def models_dir(self) -> Path:
        return self.base / "models"

    @property
    def dictionary_path(self) -> Path:
        return self.base / "dictionary.json"

    @property
    def diagnostics_dir(self) -> Path:
        return self.base / "diagnostics"

    @property
    def session_dir(self) -> Path:
        if self._session_dir is None:
            raise RuntimeError("diagnostics session has not been initialized")
        return self._session_dir

    @property
    def session_log_path(self) -> Path:
        return self.session_dir / "session.log"

    def activate_session(self, session_dir: Path) -> None:
        self._session_dir = session_dir

    def ensure_dirs(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def is_first_run(self) -> bool:
        return not self.config_path.exists()
