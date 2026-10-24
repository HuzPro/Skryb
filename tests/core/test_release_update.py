import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from skryb.core import release_update
from skryb.core.release_update import (
    Release,
    ReleaseUpdater,
    UpdateStatus,
    is_installed_copy,
    is_newer,
    release_from_github,
    run_installer,
)
from skryb.downloads.chunked_file_downloader import FileDownloadPlan

# The installer bytes the fake GitHub serves: 31 MB in the real release,
# 300 kB here, big enough for a thousand progress callbacks.
_INSTALLER = bytes(range(256)) * 1200
_SHA = hashlib.sha256(_INSTALLER).hexdigest()


def _asset(version: str, **overrides: object) -> dict[str, object]:
    return {
        "name": f"Skryb-Setup-{version}.exe",
        "size": len(_INSTALLER),
        "digest": f"sha256:{_SHA}",
        "browser_download_url": f"https://github.com/HuzPro/Skryb/releases/download/v{version}/Skryb-Setup-{version}.exe",
        **overrides,
    }


def _github(version: str, *assets: dict[str, object]) -> dict[str, object]:
    """A ``releases/latest`` answer shaped like the real v0.1.0 one."""
    return {
        "tag_name": f"v{version}",
        "name": f"Skryb v{version}",
        "prerelease": False,
        "draft": False,
        "assets": list(assets) if assets else [_asset(version)],
    }


class _Downloader:
    """Writes the installer in 1,200 pieces from eight threads, like the chunked downloader."""

    def __init__(self, payload: bytes = _INSTALLER) -> None:
        self.payload = payload
        self.plans: list[FileDownloadPlan] = []

    def download(self, plan, on_progress, cancel_event, pause_event) -> None:
        self.plans.append(plan)
        plan.target.write_bytes(self.payload)
        pieces = [256] * (len(self.payload) // 256)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(on_progress, pieces))


class _Rig:
    def __init__(self, tmp_path: Path, *, running: str = "0.1.0", latest: object = None) -> None:
        self.updates_dir = tmp_path / "updates"
        self.latest = latest if latest is not None else _github("0.2.0")
        self.fetches = 0
        self.downloader = _Downloader()
        self.statuses: list[UpdateStatus] = []
        self.offers: list[str] = []
        self.busy_for_sleeps = 0
        self.started: list[tuple[Path, Path]] = []
        self.quits = 0
        self.enabled = True
        self.updater = ReleaseUpdater(
            running_version=running,
            updates_dir=self.updates_dir,
            enabled=lambda: self.enabled,
            fetch_latest=self._fetch,
            downloader=self.downloader,
            is_busy=lambda: self.busy_for_sleeps > 0,
            quit_into=self._quit_into,
            start_installer=lambda installer, log: self.started.append((installer, log)),
            on_change=self.statuses.append,
            offer=self.offers.append,
            sleep=self._sleep,
        )

    def _fetch(self) -> object:
        self.fetches += 1
        if isinstance(self.latest, Exception):
            raise self.latest
        return self.latest

    def _sleep(self, _seconds: float) -> None:
        self.busy_for_sleeps -= 1

    def _quit_into(self, start) -> None:
        assert self.busy_for_sleeps <= 0, "quit in the middle of a dictation"
        start()
        self.quits += 1


class TestReadingGitHubsLatestRelease:
    def test_takes_the_setup_exe_from_a_release_with_other_assets(self):
        document = _github(
            "0.2.0",
            {"name": "Source code (zip)", "size": 10, "browser_download_url": "https://github.com/x.zip"},
            _asset("0.1.0"),
            _asset("0.2.0"),
            _asset("0.2.0", digest="sha256:" + "0" * 64),
        )

        release = release_from_github(document)

        assert release == Release(
            version="0.2.0",
            installer_name="Skryb-Setup-0.2.0.exe",
            url="https://github.com/HuzPro/Skryb/releases/download/v0.2.0/Skryb-Setup-0.2.0.exe",
            size=len(_INSTALLER),
            sha256=_SHA,
        )

    @pytest.mark.parametrize("digest", [None, "", "sha256:", "sha512:" + "a" * 128, "sha256:" + "z" * 64, "sha256:abc", 42])
    def test_offers_nothing_without_a_sha256_github_recorded(self, digest):
        assert release_from_github(_github("0.2.0", _asset("0.2.0", digest=digest))) is None

    def test_reads_an_uppercase_digest(self):
        release = release_from_github(_github("0.2.0", _asset("0.2.0", digest=f"sha256:{_SHA.upper()}")))

        assert release is not None and release.sha256 == _SHA

    @pytest.mark.parametrize("tag", ["", None, "nightly", "v0.2", "v0.2.0-rc1", "0.2.0.1", "v٠.٢.٠", " ", 3])
    def test_offers_nothing_for_a_tag_that_is_not_a_plain_version(self, tag):
        document = _github("0.2.0")
        document["tag_name"] = tag

        assert release_from_github(document) is None

    @pytest.mark.parametrize("size", [0, -1, True, None, "31 MB", 500 * 1024 * 1024 + 1])
    def test_offers_nothing_for_an_installer_size_that_cannot_be_ours(self, size):
        assert release_from_github(_github("0.2.0", _asset("0.2.0", size=size))) is None

    def test_accepts_an_installer_of_exactly_the_largest_size(self):
        release = release_from_github(_github("0.2.0", _asset("0.2.0", size=500 * 1024 * 1024)))

        assert release is not None

    @pytest.mark.parametrize("url", [
        "http://github.com/HuzPro/Skryb/releases/download/v0.2.0/Skryb-Setup-0.2.0.exe",
        "https://github.com.evil.example/Skryb-Setup-0.2.0.exe",
        "https://example.com/Skryb-Setup-0.2.0.exe",
        "",
        None,
    ])
    def test_offers_nothing_downloaded_from_anywhere_but_github(self, url):
        assert release_from_github(_github("0.2.0", _asset("0.2.0", browser_download_url=url))) is None

    def test_offers_nothing_when_the_setup_exe_is_named_for_another_version(self):
        assert release_from_github(_github("0.2.0", _asset("0.1.9"))) is None

    @pytest.mark.parametrize("document", [None, [], "Not Found", {"message": "Not Found"}, {"tag_name": "v0.2.0"}, {"tag_name": "v0.2.0", "assets": None}])
    def test_offers_nothing_for_an_answer_that_is_not_a_release(self, document):
        assert release_from_github(document) is None


class TestComparingVersions:
    @pytest.mark.parametrize(("candidate", "running", "newer"), [
        ("0.2.0", "0.1.0", True),
        ("0.1.1", "0.1.0", True),
        ("0.1.10", "0.1.9", True),
        ("1.0.0", "0.99.99", True),
        ("v0.2.0", "0.1.0", True),
        ("0.1.0", "0.1.0", False),
        ("0.0.9", "0.1.0", False),
        ("0.2.0", "unknown", False),
        ("garbage", "0.1.0", False),
        ("", "", False),
    ])
    def test_compares_each_number_as_a_number(self, candidate, running, newer):
        assert is_newer(candidate, running) is newer


class TestCheckingForAnUpdate:
    def test_downloads_a_newer_installer_and_offers_it_once(self, tmp_path):
        rig = _Rig(tmp_path)

        rig.updater.check_once()

        assert rig.updater.status() == UpdateStatus(state="ready", version="0.2.0")
        assert rig.updater.offered_version() == "0.2.0"
        assert rig.offers == ["0.2.0"]
        assert (rig.updates_dir / "Skryb-Setup-0.2.0.exe").read_bytes() == _INSTALLER

    def test_shows_download_progress_a_percent_at_a_time_from_a_thousand_pieces(self, tmp_path):
        rig = _Rig(tmp_path)

        rig.updater.check_once()

        downloading = [status for status in rig.statuses if status.state == "downloading"]
        assert 90 <= len(downloading) <= 101
        assert downloading[-1].bytes_done == len(_INSTALLER)
        done = [status.bytes_done for status in downloading]
        assert done == sorted(done)

    def test_says_up_to_date_when_the_running_copy_is_the_latest(self, tmp_path):
        rig = _Rig(tmp_path, running="0.2.0")

        rig.updater.check_once()

        assert rig.updater.status().state == "up_to_date"
        assert rig.downloader.plans == []
        assert rig.offers == []

    def test_never_offers_an_older_release_than_the_running_copy(self, tmp_path):
        rig = _Rig(tmp_path, running="0.3.0", latest=_github("0.2.0"))

        rig.updater.check_once()

        assert rig.updater.status().state == "up_to_date"

    def test_throws_away_a_download_that_does_not_match_githubs_sha256(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.downloader.payload = _INSTALLER[:-1] + b"\x00"

        rig.updater.check_once()

        status = rig.updater.status()
        assert status.state == "failed"
        assert "doesn't match the file GitHub published" in status.error
        assert not (rig.updates_dir / "Skryb-Setup-0.2.0.exe").exists()
        assert rig.offers == []
        assert rig.updater.offered_version() == ""

    def test_reports_a_failed_check_in_plain_words(self, tmp_path):
        rig = _Rig(tmp_path, latest=ConnectionError("connection reset by GitHub"))

        rig.updater.check_once()

        assert rig.updater.status() == UpdateStatus(
            state="failed", error="Couldn't check for updates. Connection reset by GitHub.",
        )

    def test_keeps_a_ready_update_ready_when_a_later_check_fails(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.updater.check_once()
        rig.latest = OSError("offline")

        rig.updater.check_once()

        assert rig.updater.status() == UpdateStatus(state="ready", version="0.2.0")

    def test_a_thousand_checks_download_once_and_offer_once(self, tmp_path):
        rig = _Rig(tmp_path)

        for _ in range(1000):
            rig.updater.check_once()

        assert len(rig.downloader.plans) == 1
        assert rig.offers == ["0.2.0"]
        assert rig.fetches == 1000

    def test_reuses_an_installer_downloaded_before_a_restart(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.updates_dir.mkdir()
        (rig.updates_dir / "Skryb-Setup-0.2.0.exe").write_bytes(_INSTALLER)

        rig.updater.check_once()

        assert rig.downloader.plans == []
        assert rig.updater.status() == UpdateStatus(state="ready", version="0.2.0")

    def test_deletes_installers_for_other_versions_but_keeps_a_partial_one_for_this(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.updates_dir.mkdir()
        for name in ("Skryb-Setup-0.1.5.exe", "Skryb-Setup-0.1.9.exe.part", "Skryb-Setup-0.2.0.exe.part.meta", "notes.txt"):
            (rig.updates_dir / name).write_text("x", encoding="utf-8")
        rig.downloader.download = lambda plan, *_args: None  # stops before writing anything

        rig.updater.check_once()

        assert sorted(path.name for path in rig.updates_dir.iterdir()) == ["Skryb-Setup-0.2.0.exe.part.meta", "notes.txt"]

    def test_a_newer_release_replaces_the_one_waiting(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.updater.check_once()
        rig.latest = _github("0.3.0")

        rig.updater.check_once()

        assert rig.updater.offered_version() == "0.3.0"
        assert rig.offers == ["0.2.0", "0.3.0"]
        assert sorted(path.name for path in rig.updates_dir.iterdir()) == ["Skryb-Setup-0.3.0.exe"]


class TestInstalling:
    def test_waits_for_the_dictation_then_quits_into_the_installer(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.updater.check_once()
        rig.busy_for_sleeps = 5

        rig.updater._install_when_idle(release_from_github(rig.latest))

        assert rig.quits == 1
        assert rig.started == [(rig.updates_dir / "Skryb-Setup-0.2.0.exe", rig.updates_dir / "install-0.2.0.log")]
        pending = json.loads((rig.updates_dir / "pending.json").read_text(encoding="utf-8"))
        assert pending == {"version": "0.2.0", "log": str(rig.updates_dir / "install-0.2.0.log")}

    def test_runs_install_off_the_calling_thread(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.updater.check_once()
        quit_on: list[threading.Thread] = []
        quitting = threading.Event()

        def quit_into(_start) -> None:
            quit_on.append(threading.current_thread())
            quitting.set()

        rig.updater._quit_into = quit_into

        rig.updater.install()

        assert quitting.wait(5)
        assert quit_on[0] is not threading.current_thread()
        assert rig.updater.status() == UpdateStatus(state="installing", version="0.2.0")

    def test_does_nothing_with_no_update_downloaded(self, tmp_path):
        rig = _Rig(tmp_path, running="0.2.0")
        rig.updater.check_once()

        rig.updater.install()

        assert rig.quits == 0
        assert rig.updater.status().state == "up_to_date"

    def test_refuses_an_installer_that_changed_on_disk_after_its_check(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.updater.check_once()
        (rig.updates_dir / "Skryb-Setup-0.2.0.exe").write_bytes(b"MZ not ours")

        rig.updater._install_when_idle(release_from_github(rig.latest))

        assert rig.quits == 0
        assert rig.updater.status().state == "failed"
        assert rig.updater.offered_version() == ""

    def test_keeps_skryb_running_when_the_installer_will_not_start(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.updater.check_once()

        def blocked(_installer: Path, _log: Path) -> None:
            raise PermissionError("Access is denied")

        rig.updater._start_installer = blocked

        with pytest.raises(PermissionError):
            rig.updater._install_when_idle(release_from_github(rig.latest))

        assert rig.quits == 0
        assert not (rig.updates_dir / "pending.json").exists()
        assert rig.updater.status().error == "Couldn't start the installer. Access is denied."


class TestAfterAnUpdate:
    def _pending(self, rig: _Rig, text: str) -> None:
        rig.updates_dir.mkdir(parents=True, exist_ok=True)
        (rig.updates_dir / "pending.json").write_text(text, encoding="utf-8")

    def test_the_new_version_forgets_the_finished_update_quietly(self, tmp_path):
        rig = _Rig(tmp_path, running="0.2.0")
        self._pending(rig, json.dumps({"version": "0.2.0", "log": "C:\\Users\\Hüz\\install-0.2.0.log"}))

        rig.updater._report_unfinished_update()

        assert rig.statuses == []
        assert not (rig.updates_dir / "pending.json").exists()

    def test_the_old_version_says_the_update_did_not_finish_and_names_the_log(self, tmp_path):
        rig = _Rig(tmp_path, running="0.1.0")
        self._pending(rig, json.dumps({"version": "0.2.0", "log": "C:\\Users\\Hüz\\install-0.2.0.log"}))

        rig.updater._report_unfinished_update()

        assert rig.updater.status() == UpdateStatus(
            state="failed", version="0.2.0",
            error="The update to 0.2.0 didn't finish. The installer's log is C:\\Users\\Hüz\\install-0.2.0.log.",
        )

    @pytest.mark.parametrize("text", ["", "   ", "not json", "[]", "null", '{"version": ""}', '{"version": "ünïcode"}', "\x00\x01"])
    def test_ignores_and_removes_a_marker_it_cannot_read(self, tmp_path, text):
        rig = _Rig(tmp_path)
        self._pending(rig, text)

        rig.updater._report_unfinished_update()

        assert rig.statuses == []
        assert not (rig.updates_dir / "pending.json").exists()


class TestTheHourlyLoop:
    def _wait_for(self, condition) -> None:
        for _ in range(500):
            if condition():
                return
            threading.Event().wait(0.01)
        raise AssertionError("timed out")

    def test_checks_soon_after_start_when_automatic_checks_are_on(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.updater.start(first_check_after=0)
        try:
            self._wait_for(lambda: rig.updater.status().state == "ready")
        finally:
            rig.updater.dispose()
        assert rig.fetches == 1

    def test_does_not_ask_github_when_automatic_checks_are_off(self, tmp_path):
        rig = _Rig(tmp_path)
        rig.enabled = False
        rig.updater.start(first_check_after=0)
        try:
            self._wait_for(lambda: rig.updater.status().state == "off")
            assert rig.fetches == 0

            rig.updater.check_now()
            self._wait_for(lambda: rig.updater.status().state == "ready")
        finally:
            rig.updater.dispose()
        assert rig.fetches == 1


class TestStartingTheInstaller:
    def test_runs_it_silently_detached_and_waiting_for_this_process(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(release_update.subprocess, "Popen", lambda command, **kwargs: calls.append((command, kwargs)))
        monkeypatch.setattr("os.getpid", lambda: 4242)
        installer, log = tmp_path / "Skryb Setup 0.2.0.exe", tmp_path / "logs ü" / "install.log"

        run_installer(installer, log)

        command, kwargs = calls[0]
        assert command == [
            str(installer), "/SILENT", "/NORESTART", "/FORCECLOSEAPPLICATIONS", "/WAITPID=4242", f"/LOG={log}",
        ]
        assert kwargs["creationflags"] & 0x00000008  # detached
        assert kwargs["creationflags"] & 0x01000000  # out of any job Skryb is in

    def test_still_runs_it_when_the_job_forbids_breaking_away(self, monkeypatch, tmp_path):
        calls = []

        def popen(command, **kwargs):
            calls.append(kwargs["creationflags"])
            if kwargs["creationflags"] & 0x01000000:
                raise PermissionError("Access is denied")

        monkeypatch.setattr(release_update.subprocess, "Popen", popen)

        run_installer(tmp_path / "Skryb-Setup-0.2.0.exe", tmp_path / "install.log")

        assert len(calls) == 2 and not calls[1] & 0x01000000


class TestKnowingAnInstalledCopy:
    def test_an_installed_copy_has_the_uninstaller_beside_its_python(self, tmp_path):
        (tmp_path / "python").mkdir()
        (tmp_path / "unins000.exe").write_bytes(b"MZ")

        assert is_installed_copy(tmp_path / "python")

    def test_a_venv_or_checkout_has_no_uninstaller(self, tmp_path):
        (tmp_path / "venv").mkdir()

        assert not is_installed_copy(tmp_path / "venv")
