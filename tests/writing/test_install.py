import hashlib
import io
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fakes.writing import UnreachableDownloader
from voicetotext.downloads.chunked_file_downloader import FileDownloadPlan
from voicetotext.writing.install import (
    DOWNLOAD_CORRUPTED,
    DOWNLOAD_FAILED,
    INSTALL_FAILED,
    MODEL_GGUF,
    RUNTIME_ZIP,
    Artifact,
    InstallError,
    InstallJob,
    WritingInstaller,
    WritingPaths,
    server_files,
)
from voicetotext.writing.settings import WritingSettings


def _zip_of(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


RUNTIME_BYTES = _zip_of({
    "llama-server.exe": b"MZ fake server",
    "ggml-vulkan.dll": b"\x00" * 2048,
    "LICENSE": "Ünïcode licence ✓".encode(),
})
MODEL_BYTES = b"GGUF" + bytes(range(256)) * 40


def _artifact(name: str, data: bytes) -> Artifact:
    return Artifact(
        url=f"https://example.invalid/{name}",
        file_name=name,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


RUNTIME = _artifact("llama-test-bin-win-vulkan-x64.zip", RUNTIME_BYTES)
MODEL = _artifact("tiny.gguf", MODEL_BYTES)


class FakeDownloader:
    """Serves bytes by URL the way the chunked downloader lands a file."""

    def __init__(self, served: dict[str, bytes]) -> None:
        self.served = served
        self.fetched: list[str] = []

    def download(
        self,
        plan: FileDownloadPlan,
        on_progress: Callable[[int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        self.fetched.append(plan.url)
        data = self.served[plan.url]
        plan.target.parent.mkdir(parents=True, exist_ok=True)
        plan.target.write_bytes(data)
        on_progress(len(data))


def _run(job: InstallJob) -> list[tuple[int, int]]:
    progress: list[tuple[int, int]] = []
    job.run(lambda done, total: progress.append((done, total)), threading.Event(), threading.Event())
    return progress


def _failure_of(job: InstallJob) -> InstallError:
    with pytest.raises(InstallError) as caught:
        _run(job)
    return caught.value


class TestInstallJob:
    def test_install_verifies_size_and_hash_then_unpacks_the_runtime_and_places_the_model(
        self, tmp_path,
    ):
        paths = WritingPaths(tmp_path / "writing")
        downloader = FakeDownloader({RUNTIME.url: RUNTIME_BYTES, MODEL.url: MODEL_BYTES})

        progress = _run(InstallJob(paths, downloader, RUNTIME, MODEL))

        assert paths.runtime_exe.read_bytes() == b"MZ fake server"
        assert (paths.runtime_exe.parent / "LICENSE").read_text(encoding="utf-8") == "Ünïcode licence ✓"
        assert paths.model_file.read_bytes() == MODEL_BYTES
        assert progress[-1] == (RUNTIME.size + MODEL.size, RUNTIME.size + MODEL.size)
        assert not any(paths.downloads_dir.iterdir())

    def test_install_rejects_a_corrupt_zip_and_keeps_nothing_of_it(self, tmp_path):
        paths = WritingPaths(tmp_path / "writing")
        corrupt = bytearray(RUNTIME_BYTES)
        corrupt[len(corrupt) // 2] ^= 0xFF
        downloader = FakeDownloader({RUNTIME.url: bytes(corrupt), MODEL.url: MODEL_BYTES})

        failure = _failure_of(InstallJob(paths, downloader, RUNTIME, MODEL))

        assert str(failure) == DOWNLOAD_CORRUPTED
        assert "checksum" in failure.detail
        assert not paths.runtime_exe.exists()
        assert not (paths.downloads_dir / RUNTIME.file_name).exists()

    def test_install_rejects_a_download_of_the_wrong_size(self, tmp_path):
        paths = WritingPaths(tmp_path / "writing")
        downloader = FakeDownloader({RUNTIME.url: RUNTIME_BYTES[:-10], MODEL.url: MODEL_BYTES})

        failure = _failure_of(InstallJob(paths, downloader, RUNTIME, MODEL))

        assert str(failure) == DOWNLOAD_CORRUPTED
        assert "expected size" in failure.detail

    def test_install_rejects_a_verified_file_that_is_not_a_zip(self, tmp_path):
        paths = WritingPaths(tmp_path / "writing")
        not_a_zip = b"this is not a zip archive at all" * 10
        runtime = _artifact("llama.zip", not_a_zip)

        failure = _failure_of(
            InstallJob(paths, FakeDownloader({runtime.url: not_a_zip}), runtime, MODEL),
        )

        assert str(failure) == DOWNLOAD_CORRUPTED
        assert "unpacked" in failure.detail

    def test_install_rejects_an_archive_without_the_server(self, tmp_path):
        paths = WritingPaths(tmp_path / "writing")
        data = _zip_of({"README.md": b"no server here"})
        runtime = _artifact("llama.zip", data)

        failure = _failure_of(InstallJob(paths, FakeDownloader({runtime.url: data}), runtime, MODEL))

        assert "llama-server.exe" in failure.detail
        assert not paths.runtime_exe.parent.exists()

    def test_a_download_with_no_network_says_so_plainly_and_logs_the_details(self, tmp_path):
        paths = WritingPaths(tmp_path / "writing")

        failure = _failure_of(InstallJob(paths, UnreachableDownloader(), RUNTIME, MODEL))

        assert str(failure) == DOWNLOAD_FAILED
        assert "\n" not in str(failure) and "http" not in str(failure)
        log = paths.install_log.read_text(encoding="utf-8")
        assert RUNTIME.url in log
        assert "getaddrinfo failed" in log

    def test_an_unexpected_error_is_still_one_sentence_with_the_details_in_the_log(
        self, tmp_path,
    ):
        paths = WritingPaths(tmp_path / "writing café")
        paths.root.mkdir()
        paths.model_file.parent.write_text("a file where the models folder goes")
        downloader = FakeDownloader({RUNTIME.url: RUNTIME_BYTES, MODEL.url: MODEL_BYTES})

        failure = _failure_of(InstallJob(paths, downloader, RUNTIME, MODEL))

        assert str(failure) == INSTALL_FAILED
        assert "FileExistsError" in paths.install_log.read_text(encoding="utf-8")

    def test_a_retry_skips_the_parts_already_installed(self, tmp_path):
        paths = WritingPaths(tmp_path / "writing")
        downloader = FakeDownloader({RUNTIME.url: RUNTIME_BYTES, MODEL.url: MODEL_BYTES})
        _run(InstallJob(paths, downloader, RUNTIME, MODEL))

        again = FakeDownloader({})
        _run(InstallJob(paths, again, RUNTIME, MODEL))

        assert again.fetched == []

    def test_the_pinned_artifacts_are_the_measured_ones(self):
        assert RUNTIME_ZIP.size == 33_062_078
        assert RUNTIME_ZIP.sha256 == "670a1f9bf5272c43dbbd7b0004a12dd3d3f6f575246fd23b733f06c064f1b96b"
        assert "e87f176479d0855a907a41277aca2f8ee7a09523" in MODEL_GGUF.url
        assert MODEL_GGUF.size == 2_740_937_888


class TestInstaller:
    def test_installing_through_the_download_manager_reports_progress_then_completion(
        self, tmp_path,
    ):
        paths = WritingPaths(tmp_path / "writing")
        changes: list[str] = []
        installer = WritingInstaller(
            paths,
            lambda: changes.append(installer.progress.state),
            lambda: FakeDownloader({RUNTIME.url: RUNTIME_BYTES, MODEL.url: MODEL_BYTES}),
            RUNTIME,
            MODEL,
        )

        installer.install()
        _wait_for(lambda: installer.progress.state == "completed")
        installer.shutdown()

        assert changes[0] == "downloading"
        assert paths.model_file.exists()

    def test_a_corrupted_download_is_reported_plainly_and_the_log_names_the_file(self, tmp_path):
        paths = WritingPaths(tmp_path / "writing")
        installer = WritingInstaller(
            paths,
            lambda: None,
            lambda: FakeDownloader({RUNTIME.url: b"x" * RUNTIME.size, MODEL.url: MODEL_BYTES}),
            RUNTIME,
            MODEL,
        )

        installer.install()
        _wait_for(lambda: installer.progress.state == "failed")
        installer.shutdown()

        assert installer.progress.error == DOWNLOAD_CORRUPTED
        log = paths.install_log.read_text(encoding="utf-8")
        assert f"{RUNTIME.file_name} failed its checksum" in log

    def test_a_cancelled_install_is_not_logged_as_a_failure(self, tmp_path):
        paths = WritingPaths(tmp_path / "writing")
        cancel = threading.Event()
        cancel.set()
        downloader = FakeDownloader({RUNTIME.url: RUNTIME_BYTES, MODEL.url: MODEL_BYTES})

        with pytest.raises(InstallError):
            InstallJob(paths, downloader, RUNTIME, MODEL).run(
                lambda _done, _total: None, cancel, threading.Event(),
            )

        assert not paths.install_log.exists()


class TestServerFiles:
    def test_overrides_win_over_the_managed_install(self, tmp_path):
        runtime = tmp_path / "custom" / "llama-server.exe"
        model = tmp_path / "custom" / "mine.gguf"
        runtime.parent.mkdir()
        runtime.write_bytes(b"MZ")
        model.write_bytes(b"GGUF")
        settings = WritingSettings(runtime_path=str(runtime), model_path=str(model))

        files = server_files(WritingPaths(tmp_path / "managed"), settings)

        assert files is not None and files.runtime == runtime and files.model == model

    def test_a_partly_moved_managed_model_is_not_launched(self, tmp_path):
        paths = WritingPaths(tmp_path / "managed")
        paths.runtime_exe.parent.mkdir(parents=True)
        paths.runtime_exe.write_bytes(b"MZ")
        paths.model_file.parent.mkdir(parents=True)
        paths.model_file.write_bytes(b"GGUF too short")

        assert server_files(paths, WritingSettings()) is None


def _wait_for(condition: Callable[[], bool], seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition never became true")
        time.sleep(0.01)
