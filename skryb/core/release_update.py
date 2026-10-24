"""Offer the newest Skryb release from GitHub and install it when asked.

Only a copy Skryb-Setup installed updates this way; a copy running from a
git checkout keeps updating through git (``self_update``). The installer is
downloaded in the background and checked against the SHA-256 GitHub records
for it before it is offered. Installing runs it silently: it waits for this
process to exit, installs over it and starts Skryb again.
"""

import hashlib
import json
import logging
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from skryb.downloads.chunked_file_downloader import FileDownloadPlan

_logger = logging.getLogger(__name__)

LATEST_RELEASE_URL = "https://api.github.com/repos/HuzPro/Skryb/releases/latest"
FIRST_CHECK_SECONDS = 15.0
CHECK_EVERY_SECONDS = 3600.0
# The installer is about 31 MB; anything far bigger is not one of ours.
_MAX_INSTALLER_BYTES = 500 * 1024 * 1024
_IDLE_POLL_SECONDS = 0.5
_PENDING_FILE = "pending.json"
# [0-9], not \d: \d also matches digits such as Arabic-Indic ones.
_VERSION = re.compile(r"^v?([0-9]+)\.([0-9]+)\.([0-9]+)$")
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


@dataclass(frozen=True)
class Release:
    version: str
    installer_name: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class UpdateStatus:
    """What the tray and Settings show. ``state`` is one of
    off, idle, checking, up_to_date, downloading, ready, installing, failed."""

    state: str = "idle"
    version: str = ""
    bytes_done: int = 0
    bytes_total: int = 0
    error: str = ""


class InstallerDownloader(Protocol):
    def download(
        self,
        plan: FileDownloadPlan,
        on_progress: Callable[[int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None: ...


def parse_version(text: str) -> tuple[int, int, int] | None:
    match = _VERSION.match(text.strip())
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def is_newer(candidate: str, running: str) -> bool:
    """Whether *candidate* is a later release; a version either can't parse never is."""
    new, old = parse_version(candidate), parse_version(running)
    return new is not None and old is not None and new > old


def release_from_github(document: object) -> Release | None:
    """The installer a ``releases/latest`` answer offers, or None when it offers none we trust."""
    if not isinstance(document, dict):
        return None
    tag = document.get("tag_name")
    if not isinstance(tag, str) or parse_version(tag) is None:
        return None
    version = tag.strip().removeprefix("v")
    name = f"Skryb-Setup-{version}.exe"
    for asset in document.get("assets") or []:
        if isinstance(asset, dict) and asset.get("name") == name:
            return _trusted_asset(version, name, asset)
    return None


def _trusted_asset(version: str, name: str, asset: dict[str, object]) -> Release | None:
    digest, size, url = asset.get("digest"), asset.get("size"), asset.get("browser_download_url")
    if not (isinstance(digest, str) and digest.startswith("sha256:")):
        return None
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= _MAX_INSTALLER_BYTES:
        return None
    if not (isinstance(url, str) and url.startswith("https://github.com/")):
        return None
    sha256 = digest.removeprefix("sha256:").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        return None
    return Release(version=version, installer_name=name, url=url, size=size, sha256=sha256)


def is_installed_copy(prefix: Path = Path(sys.prefix)) -> bool:
    """Whether Skryb-Setup installed this copy: its uninstaller sits beside the bundled Python."""
    return (prefix.parent / "unins000.exe").is_file()


def sha256_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_latest_release(running_version: str) -> object:
    import requests

    response = requests.get(
        LATEST_RELEASE_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": f"skryb/{running_version}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def run_installer(installer: Path, log: Path) -> None:
    """Start the installer on its own, so it outlives this process and can replace it."""
    import os

    # Setup waits a minute for this process; anything still holding Skryb's
    # files after that is stuck, so it is closed rather than asked about.
    command = [
        str(installer), "/SILENT", "/NORESTART", "/FORCECLOSEAPPLICATIONS",
        f"/WAITPID={os.getpid()}", f"/LOG={log}",
    ]
    flags = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(command, creationflags=flags | _CREATE_BREAKAWAY_FROM_JOB, close_fds=True)
    except PermissionError:
        # A job that forbids breaking away refuses the flag; the installer
        # still runs, it only shares the job.
        subprocess.Popen(command, creationflags=flags, close_fds=True)


class ReleaseUpdater:
    """Checks GitHub every hour, downloads a newer installer, and installs it when asked.

    ``on_change`` hears every status change; ``offer`` is told once per
    version that an update is ready. Checks and downloads run on the
    updater's own thread; ``install`` may be called from any thread and
    waits for a dictation to finish before ``quit_into`` closes the app.
    """

    def __init__(
        self,
        *,
        running_version: str,
        updates_dir: Path,
        enabled: Callable[[], bool],
        fetch_latest: Callable[[], object],
        downloader: InstallerDownloader,
        is_busy: Callable[[], bool],
        quit_into: Callable[[Callable[[], None]], None],
        start_installer: Callable[[Path, Path], None] = run_installer,
        on_change: Callable[[UpdateStatus], None] = lambda _status: None,
        offer: Callable[[str], None] = lambda _version: None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._running_version = running_version
        self._updates_dir = updates_dir
        self._enabled = enabled
        self._fetch_latest = fetch_latest
        self._downloader = downloader
        self._is_busy = is_busy
        self._quit_into = quit_into
        self._start_installer = start_installer
        self._on_change = on_change
        self._offer = offer
        self._sleep = sleep
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._manual = False
        self._status = UpdateStatus()
        self._ready: Release | None = None
        self._offered = ""
        self._thread: threading.Thread | None = None

    def status(self) -> UpdateStatus:
        with self._lock:
            return self._status

    def offered_version(self) -> str:
        """The version waiting to be installed, or "" when there is none."""
        with self._lock:
            return self._ready.version if self._ready is not None else ""

    def start(self, first_check_after: float = FIRST_CHECK_SECONDS) -> None:
        self._report_unfinished_update()
        self._thread = threading.Thread(
            target=self._run, args=(first_check_after,), name="release-updater", daemon=True,
        )
        self._thread.start()

    def check_now(self) -> None:
        """Check at once, even with automatic checks turned off."""
        with self._lock:
            self._manual = True
        self._wake.set()

    def dispose(self) -> None:
        self._stopped.set()
        self._wake.set()

    def install(self) -> None:
        """Install the downloaded update once no dictation is under way."""
        with self._lock:
            release = self._ready
            if release is None or self._status.state == "installing":
                return
        threading.Thread(target=self._install_when_idle, args=(release,), daemon=True).start()

    # -- checking --------------------------------------------------------------------------

    def _run(self, first_check_after: float) -> None:
        self._wake.wait(first_check_after)
        while not self._stopped.is_set():
            # Cleared before the flag is read, so a click in between still wakes the next pass.
            self._wake.clear()
            with self._lock:
                manual, self._manual = self._manual, False
            if manual or self._enabled():
                self.check_once()
            elif self.status().state not in ("ready", "installing"):
                self._set(UpdateStatus(state="off"))
            self._wake.wait(CHECK_EVERY_SECONDS)

    def check_once(self) -> None:
        if self.status().state in ("downloading", "installing"):
            return
        ready = self.offered_version()
        self._set(replace(self.status(), state="checking", error=""))
        try:
            release = self._newer_release()
        except Exception as exc:  # noqa: BLE001 - any failure is reported, never raised
            self._report_failure("Couldn't check for updates", exc, keep_ready=bool(ready))
        else:
            self._act_on(release)

    def _newer_release(self) -> Release | None:
        release = release_from_github(self._fetch_latest())
        if release is None or not is_newer(release.version, self._running_version):
            return None
        return release

    def _act_on(self, release: Release | None) -> None:
        if release is None:
            self._set(UpdateStatus(state="up_to_date"))
            return
        if self.offered_version() == release.version:
            self._set(UpdateStatus(state="ready", version=release.version))
            return
        # An older ready installer is about to be deleted for this one.
        with self._lock:
            self._ready = None
        try:
            self._download(release)
        except Exception as exc:  # noqa: BLE001
            self._report_failure(f"Couldn't download Skryb {release.version}", exc)
        else:
            self._make_ready(release)

    # -- downloading -----------------------------------------------------------------------

    def _download(self, release: Release) -> None:
        target = self._updates_dir / release.installer_name
        self._remove_other_installers(keep=target)
        if target.is_file() and sha256_of(target) == release.sha256:
            return
        target.unlink(missing_ok=True)
        self._set(UpdateStatus(state="downloading", version=release.version, bytes_total=release.size))
        progress = threading.Lock()
        done = 0
        shown_percent = 0

        # Chunks arrive on several threads; only a new whole percent is shown.
        def advance(delta: int) -> None:
            nonlocal done, shown_percent
            with progress:
                done = min(done + delta, release.size)
                percent = done * 100 // release.size
                if percent == shown_percent:
                    return
                shown_percent = percent
                status = UpdateStatus(
                    state="downloading", version=release.version,
                    bytes_done=done, bytes_total=release.size,
                )
            self._set(status)

        plan = FileDownloadPlan(url=release.url, target=target, expected_size=release.size)
        self._downloader.download(plan, advance, self._stopped, threading.Event())
        if sha256_of(target) != release.sha256:
            target.unlink(missing_ok=True)
            raise ValueError("the download doesn't match the file GitHub published")

    def _remove_other_installers(self, keep: Path) -> None:
        self._updates_dir.mkdir(parents=True, exist_ok=True)
        for old in self._updates_dir.glob("Skryb-Setup-*.exe*"):
            if old != keep and not old.name.startswith(keep.name):
                old.unlink(missing_ok=True)

    def _make_ready(self, release: Release) -> None:
        with self._lock:
            self._ready = release
            first_offer = self._offered != release.version
            self._offered = release.version
        self._set(UpdateStatus(state="ready", version=release.version))
        if first_offer:
            self._offer(release.version)

    # -- installing ------------------------------------------------------------------------

    def _install_when_idle(self, release: Release) -> None:
        self._set(UpdateStatus(state="installing", version=release.version))
        while self._is_busy():
            self._sleep(_IDLE_POLL_SECONDS)
        installer = self._updates_dir / release.installer_name
        if not installer.is_file() or sha256_of(installer) != release.sha256:
            with self._lock:
                self._ready = None
            self._set(UpdateStatus(
                state="failed", version=release.version,
                error="The downloaded installer changed on disk, so it wasn't run. Check again to fetch it fresh.",
            ))
            return
        log = self._updates_dir / f"install-{release.version}.log"
        self._remember_pending(release.version, log)
        self._quit_into(lambda: self._launch_installer(installer, log))

    def _remember_pending(self, version: str, log: Path) -> None:
        pending = {"version": version, "log": str(log)}
        (self._updates_dir / _PENDING_FILE).write_text(json.dumps(pending), encoding="utf-8")

    def _launch_installer(self, installer: Path, log: Path) -> None:
        """Start the installer; raising stops the quit, so Skryb keeps running when it can't."""
        try:
            self._start_installer(installer, log)
        except OSError as exc:
            (self._updates_dir / _PENDING_FILE).unlink(missing_ok=True)
            self._report_failure("Couldn't start the installer", exc)
            raise

    def _report_unfinished_update(self) -> None:
        marker = self._updates_dir / _PENDING_FILE
        if not marker.is_file():
            return
        try:
            pending = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pending = {}
        marker.unlink(missing_ok=True)
        version = str(pending.get("version", "")) if isinstance(pending, dict) else ""
        if version and is_newer(version, self._running_version):
            log = pending.get("log", "") if isinstance(pending, dict) else ""
            self._set(UpdateStatus(
                state="failed", version=version,
                error=f"The update to {version} didn't finish. The installer's log is {log}.",
            ))

    # -- status ----------------------------------------------------------------------------

    def _report_failure(self, what: str, exc: Exception, *, keep_ready: bool = False) -> None:
        _logger.warning("%s: %s", what, exc, exc_info=not isinstance(exc, OSError))
        if keep_ready:
            self._set(UpdateStatus(state="ready", version=self.offered_version()))
            return
        self._set(UpdateStatus(state="failed", error=f"{what}. {_plain_reason(exc)}"))

    def _set(self, status: UpdateStatus) -> None:
        with self._lock:
            if status == self._status:
                return
            self._status = status
        self._on_change(status)


def _plain_reason(exc: Exception) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text[0].upper() + text[1:] + ("" if text.endswith(".") else ".")
