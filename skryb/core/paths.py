import logging
import os
from collections.abc import Mapping
from pathlib import Path

_logger = logging.getLogger(__name__)


def move_voicetotext_folders(environ: Mapping[str, str] = os.environ) -> None:
    """Rename the folders the app kept as VoiceToText, the first time Skryb runs.

    Settings, learned spellings, history and gigabytes of models stay put
    instead of being set up and downloaded again. Raises ``OSError`` when a
    folder can't be moved, usually because VoiceToText is still running.
    """
    for variable in ("APPDATA", "LOCALAPPDATA"):
        root = environ.get(variable)
        if not root:
            continue
        old, new = Path(root) / "VoiceToText", Path(root) / "Skryb"
        if not old.is_dir() or new.exists():
            continue
        try:
            old.rename(new)
        except OSError:
            # A second copy launched at the same moment may have moved it first.
            if old.exists() or not new.exists():
                raise
        _logger.info("Moved %s to %s", old, new)


def local_data_dir() -> Path:
    """The Skryb folder under ``%LOCALAPPDATA%``, for files that stay on this machine."""
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "Skryb"


class AppPaths:

    def __init__(self, base: Path | None = None) -> None:
        if base is None:
            base = Path(os.environ["APPDATA"]) / "Skryb"
        self.base = base
        self._session_dir: Path | None = None
        self._diagnostics_root: Path | None = None

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
    def default_diagnostics_dir(self) -> Path:
        return self.base / "diagnostics"

    @property
    def diagnostics_dir(self) -> Path:
        if self._diagnostics_root is None:
            return self.default_diagnostics_dir
        return self._diagnostics_root

    def use_diagnostics_root(self, configured: object) -> None:
        """Point diagnostics at a user-chosen folder, if one is set."""
        if not isinstance(configured, str) or not configured.strip():
            self._diagnostics_root = None
            return
        self._diagnostics_root = Path(configured.strip()).expanduser()

    def load_configured_diagnostics_root(self) -> None:
        """Apply the saved diagnostics folder before any logging starts.

        The session log opens before the app's config store is built, so
        this reads the one key it needs straight from disk. A missing or
        unreadable config leaves diagnostics at the default location.
        """
        from skryb.core.json_config_adapter import JsonConfigAdapter

        config = JsonConfigAdapter(self.config_path)
        try:
            config.load()
        except (OSError, ValueError):
            self._diagnostics_root = None
            return
        self.use_diagnostics_root(config.get("diagnostics.directory", ""))

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
        self._ensure_diagnostics_dir()

    def _ensure_diagnostics_dir(self) -> None:
        try:
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if self._diagnostics_root is None:
                raise
            _logger.warning(
                "Diagnostics folder '%s' is unusable (%s); falling back to '%s'.",
                self._diagnostics_root, exc, self.default_diagnostics_dir,
            )
            self._diagnostics_root = None
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def is_first_run(self) -> bool:
        return not self.config_path.exists()
