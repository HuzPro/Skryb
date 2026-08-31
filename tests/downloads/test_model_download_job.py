import threading
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from voicetotext.downloads.hf_repo_resolver import ResolvedFile
from voicetotext.downloads.model_download_job import (
    ModelDownloadJob,
    ModelDownloadError,
)


@dataclass
class _FakeDownloader:
    """Stand-in for ChunkedFileDownloader - writes a file synchronously."""

    payloads: dict[str, bytes]
    calls: list = None

    def __post_init__(self):
        self.calls = []

    def download(self, plan, on_progress, cancel_event, pause_event):
        self.calls.append(plan.url)
        plan.target.parent.mkdir(parents=True, exist_ok=True)
        data = self.payloads[plan.target.name]
        plan.target.write_bytes(data)
        on_progress(len(data))


class TestModelDownloadJobHappyPath:
    def test_should_download_every_file_and_write_refs_main(self, tmp_path):
        resolved = [
            ResolvedFile(filename="config.json", size=3, url="u/config.json"),
            ResolvedFile(filename="model.safetensors", size=4, url="u/model.safetensors"),
        ]
        payloads = {"config.json": b"{}\n", "model.safetensors": b"WGHT"}
        downloader = _FakeDownloader(payloads=payloads)
        job = ModelDownloadJob(
            job_id="parakeet-tdt-0.6b-v3",
            repo_id="nvidia/parakeet-tdt-0.6b-v3",
            commit_sha="sha123",
            files=resolved,
            models_dir=tmp_path,
            file_downloader=downloader,
        )

        job.run(
            on_progress=lambda done, total: None,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        snapshot = (
            tmp_path / "models--nvidia--parakeet-tdt-0.6b-v3"
            / "snapshots" / "sha123"
        )
        assert (snapshot / "config.json").read_bytes() == b"{}\n"
        assert (snapshot / "model.safetensors").read_bytes() == b"WGHT"
        refs_main = (
            tmp_path / "models--nvidia--parakeet-tdt-0.6b-v3"
            / "refs" / "main"
        )
        assert refs_main.read_text() == "sha123"

    def test_should_aggregate_progress_across_files(self, tmp_path):
        resolved = [
            ResolvedFile(filename="a.json", size=100, url="u/a"),
            ResolvedFile(filename="b.json", size=200, url="u/b"),
        ]
        payloads = {"a.json": b"A" * 100, "b.json": b"B" * 200}
        downloader = _FakeDownloader(payloads=payloads)
        job = ModelDownloadJob(
            job_id="parakeet-tdt-0.6b-v3",
            repo_id="r",
            commit_sha="sha",
            files=resolved,
            models_dir=tmp_path,
            file_downloader=downloader,
        )
        progress_events: list[tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            progress_events.append((done, total))

        job.run(
            on_progress=on_progress,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        assert progress_events[-1] == (300, 300)
        assert all(total == 300 for _, total in progress_events)

    def test_total_bytes_equals_sum_of_file_sizes(self, tmp_path):
        resolved = [
            ResolvedFile(filename="a", size=100, url="u/a"),
            ResolvedFile(filename="b", size=250, url="u/b"),
            ResolvedFile(filename="c", size=50, url="u/c"),
        ]
        job = ModelDownloadJob(
            job_id="x",
            repo_id="r",
            commit_sha="s",
            files=resolved,
            models_dir=tmp_path,
            file_downloader=MagicMock(),
        )

        assert job.total_bytes == 400


class TestModelDownloadJobErrors:
    def test_should_wrap_file_downloader_errors(self, tmp_path):
        resolved = [ResolvedFile(filename="a", size=1, url="u/a")]
        downloader = MagicMock()
        downloader.download.side_effect = RuntimeError("boom")
        job = ModelDownloadJob(
            job_id="x",
            repo_id="r",
            commit_sha="s",
            files=resolved,
            models_dir=tmp_path,
            file_downloader=downloader,
        )

        with pytest.raises(ModelDownloadError):
            job.run(
                on_progress=lambda d, t: None,
                cancel_event=threading.Event(),
                pause_event=threading.Event(),
            )
