import os
from pathlib import Path


class AppPaths:

    def __init__(self, base: Path | None = None) -> None:
        if base is None:
            base = Path(os.environ["APPDATA"]) / "VoiceToText"
        self.base = base

    @property
    def config_path(self) -> Path:
        return self.base / "config.json"

    @property
    def models_dir(self) -> Path:
        return self.base / "models"

    def ensure_dirs(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def is_first_run(self) -> bool:
        return not self.config_path.exists()
