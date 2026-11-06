import dataclasses
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from skryb.core.event_bus import SimpleEventBus
from skryb.downloads.download_manager import DownloadJob
from skryb.transcription.engine_types import HardwareKind, HardwareProfile
from skryb.translation import install
from skryb.translation.llama_server import Endpoint, LlamaServer, ServerError, ServerLaunch
from skryb.translation.models import AMD, CPU, NPU, NVIDIA, TranslationPath, path_for
from skryb.translation.service import TranslationService
from skryb.translation.translator import Translation
from skryb.ui.api_server import ApiError, ApiRequest, ApiResponse
from skryb.ui.event_hub import EventHub
from skryb.ui.translation_handlers import TranslationApi
from tests.fakes.llama_server import FakeLlamaServer

LINE = "زين، بنطلب عشرين كرتون تمر من المزرعة."
Devices = list[tuple[str, str, int]]


class Downloads:
    def __init__(self) -> None:
        self.jobs: list[DownloadJob] = []

    def enqueue(self, job: DownloadJob) -> object:
        self.jobs.append(job)
        return job


class Server(LlamaServer):
    """Builds its launch like the real one, then points at the fake server instead of starting a child."""

    def __init__(self, launch: Callable[[], ServerLaunch], log_path: Path, fake: FakeLlamaServer) -> None:
        super().__init__(launch, log_path)
        self.fake = fake
        self.starts = 0
        self.launches: list[ServerLaunch] = []

    def endpoint(self) -> Endpoint:
        if not self.launches:
            self.starts += 1
            self.launches.append(self._launch())
        return self.fake.endpoint


def small(path: TranslationPath) -> TranslationPath:
    return dataclasses.replace(path, model=dataclasses.replace(path.model, size=4))


@pytest.fixture(autouse=True)
def small_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each path's model is four bytes here, so a test's stand-in file never takes gigabytes of disk."""
    monkeypatch.setattr("skryb.translation.service.path_for", lambda hardware: small(path_for(hardware)))


def installed(models_dir: Path, path: TranslationPath) -> None:
    path = small(path)
    runtime = install.runtime_exe(models_dir, path)
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"MZ")
    model = install.model_file(models_dir, path.model)
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF")
    refs = model.parents[2] / "refs" / "main"
    refs.parent.mkdir(parents=True)
    refs.write_text(path.model.revision)


def service(
    tmp_path: Path,
    fake: FakeLlamaServer,
    kind: HardwareKind = "amd_rocm",
    npu: str | None = None,
    devices: Devices | None = None,
    rocm: list[Path] | None = None,
) -> tuple[TranslationService, Downloads, list[Server]]:
    downloads, servers = Downloads(), []
    rocm_folder = tmp_path / "rocm" / "bin"
    rocm_folder.mkdir(parents=True, exist_ok=True)

    def make_server(launch: Callable[[], ServerLaunch], log_path: Path) -> LlamaServer:
        servers.append(Server(launch, log_path, fake))
        return servers[-1]

    def listed(_runtime: Path, _env: Mapping[str, str]) -> Devices:
        return [("ROCm0", "AMD Radeon RX 9060 XT", 16304), ("ROCm1", "AMD Radeon(TM) Graphics", 12455)] if devices is None else devices

    return TranslationService(
        models_dir=tmp_path / "models", log_path=tmp_path / "translation-server.log",
        hardware=lambda: HardwareProfile(kind, "a test device", None, npu), downloads=downloads,
        make_downloader=lambda: None,  # type: ignore[arg-type,return-value]  # jobs are only recorded here
        rocm_libraries=lambda: [rocm_folder] if rocm is None else rocm, devices=listed, make_server=make_server,
    ), downloads, servers


@pytest.mark.parametrize(
    ("kind", "npu", "path"),
    [("amd_rocm", None, AMD), ("nvidia_cuda", None, NVIDIA), ("cpu", "FD3E", NPU), ("cpu", None, CPU), ("amd_rocm", "FD3E", AMD)],
)
def test_each_computer_gets_its_own_build_and_model(tmp_path: Path, kind: HardwareKind, npu: str | None, path: TranslationPath) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake, kind, npu)
        assert translation.path == small(path)


def test_gpus_run_the_larger_model_and_the_npu_and_processor_the_small_one() -> None:
    assert AMD.model == NVIDIA.model and NPU.model == CPU.model and AMD.model.size > CPU.model.size


def test_nothing_is_downloaded_until_install_is_asked_for_and_then_only_once_installed_it_stops(tmp_path: Path) -> None:
    with FakeLlamaServer() as fake:
        translation, downloads, _servers = service(tmp_path, fake)
        assert downloads.jobs == [] and not translation.installed()
        translation.install()
        assert [job.job_id for job in downloads.jobs] == ["translation"]
        installed(tmp_path / "models", AMD)
        translation.install()
        assert len(downloads.jobs) == 1 and translation.installed()


def test_a_line_before_the_model_is_downloaded_is_refused_with_a_sentence(tmp_path: Path) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake)
        with pytest.raises(ServerError, match="isn't downloaded yet"):
            translation.annotate(LINE)


def test_the_model_loads_with_the_first_line_and_runs_on_the_largest_gpu(tmp_path: Path) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, servers = service(tmp_path, fake)
        installed(tmp_path / "models", AMD)
        assert servers[0].launches == [] and translation.view()["device"] is None
        result = translation.annotate(LINE)
        launch = servers[0].launches[0]
    assert result.english == f"EN: {LINE}"
    assert launch.flags == (*AMD.flags, "--device", "ROCm0")
    assert "--fit" in launch.flags and "off" in launch.flags
    assert str(tmp_path / "rocm" / "bin") in launch.env["PATH"]
    assert translation.view()["device"] == "GPU" and translation.view()["device_name"] == "AMD Radeon RX 9060 XT"


def test_a_gpu_build_that_finds_no_gpu_says_so_and_never_falls_back_to_the_processor(tmp_path: Path) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake, devices=[])
        installed(tmp_path / "models", AMD)
        with pytest.raises(ServerError, match="found no AMD graphics card"):
            translation.annotate(LINE)
        assert fake.requests == [] and translation.view()["device"] is None
        assert "found no AMD graphics card" in str(translation.view()["error"])


def test_a_failed_start_is_not_retried_for_every_line_until_install_is_asked_for_again(tmp_path: Path) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, servers = service(tmp_path, fake, devices=[])
        installed(tmp_path / "models", AMD)
        for _ in range(3):
            with pytest.raises(ServerError, match="found no AMD graphics card"):
                translation.annotate(LINE)
        assert servers[0].starts == 1
        translation.install()
        assert translation.view()["error"] is None


def test_amd_without_the_rocm_libraries_is_refused(tmp_path: Path) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake, rocm=[])
        installed(tmp_path / "models", AMD)
        with pytest.raises(ServerError, match="ROCm libraries"):
            translation.annotate(LINE)


def test_the_npu_path_runs_only_on_an_npu_openvino_lists(tmp_path: Path) -> None:
    npu_listed = [("OPENVINO0", "GGML_OPENVINO_DEVICE=NPU (selected) - Intel(R) AI Boost", 18000)]
    with FakeLlamaServer() as fake:
        translation, _downloads, servers = service(tmp_path, fake, "cpu", "FD3E", npu_listed)
        installed(tmp_path / "models", NPU)
        translation.annotate(LINE)
        launch = servers[0].launches[0]
    assert launch.env == {"GGML_OPENVINO_DEVICE": "NPU"} and launch.flags[-2:] == ("--device", "OPENVINO0")
    assert translation.view()["device"] == "NPU" and translation.view()["device_name"] == "Intel(R) AI Boost"


def test_an_npu_openvino_cannot_see_is_reported_not_swapped_for_the_processor(tmp_path: Path) -> None:
    cpu_only = [("OPENVINO0", "GGML_OPENVINO_DEVICE=CPU (selected) - Some Processor", 32000)]
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake, "cpu", "FD3E", cpu_only)
        installed(tmp_path / "models", NPU)
        with pytest.raises(ServerError, match="found no NPU"):
            translation.annotate(LINE)
        assert fake.requests == []


def test_the_processor_path_needs_no_device_listing(tmp_path: Path) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, servers = service(tmp_path, fake, "cpu", None, devices=[])
        installed(tmp_path / "models", CPU)
        translation.annotate(LINE)
    assert servers[0].launches[0].flags == CPU.flags and translation.view()["device"] == "CPU"


@pytest.mark.parametrize(("kind", "npu", "path"), [("amd_rocm", None, AMD), ("cpu", None, CPU)])
def test_every_request_asks_the_model_not_to_think_first(tmp_path: Path, kind: HardwareKind, npu: str | None, path: TranslationPath) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake, kind, npu)
        installed(tmp_path / "models", path)
        translation.annotate(LINE)
    assert len(fake.requests) == 2
    assert all(request["chat_template_kwargs"] == {"reasoning_effort": "low"} for request in fake.requests)


def test_the_same_line_twice_asks_the_model_once(tmp_path: Path) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake)
        installed(tmp_path / "models", AMD)
        first, second = translation.annotate(LINE), translation.annotate(f"  {LINE} ")
        assert first == second and len(fake.requests) == 2


class Collected:
    """What came back from the translation thread, and on which thread, until *expected* lines are in."""

    def __init__(self, expected: int) -> None:
        self.english: list[str] = []
        self.threads: set[str] = set()
        self.finished = threading.Event()
        self._expected = expected

    def done(self, result: Translation) -> None:
        self.english.append(result.english)
        self.threads.add(threading.current_thread().name)
        if len(self.english) == self._expected:
            self.finished.set()


def test_meeting_lines_are_translated_in_order_on_their_own_thread(tmp_path: Path) -> None:
    lines = ["وين حطيت مفاتيح المحل؟", "", "Send the photos tonight.", "وين حطيت مفاتيح المحل؟", LINE]
    collected = Collected(len(lines))
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake)
        installed(tmp_path / "models", AMD)
        for line in lines:
            translation.submit(line, collected.done)
        assert collected.finished.wait(10)
    assert collected.english == [f"EN: {lines[0]}", "", "Send the photos tonight.", f"EN: {lines[0]}", f"EN: {LINE}"]
    assert collected.threads == {"translation"}


def test_a_line_that_fails_is_left_without_english_and_the_next_still_comes(tmp_path: Path) -> None:
    broken = "وين حطيت مفاتيح المحل؟"
    collected = Collected(1)
    with FakeLlamaServer() as fake:
        fake.cut_short.add(broken)
        translation, _downloads, _servers = service(tmp_path, fake)
        installed(tmp_path / "models", AMD)
        translation.submit(broken, collected.done)
        translation.submit(LINE, collected.done)
        assert collected.finished.wait(10)
    assert collected.english == [f"EN: {LINE}"]


def translate_route(api: TranslationApi) -> Callable[[ApiRequest], ApiResponse]:
    return next(handler for method, path, handler in api.routes() if (method, path) == ("POST", "/api/translation"))


def test_history_gets_english_for_each_arabic_line_of_a_dictation_and_null_for_the_rest(tmp_path: Path) -> None:
    text = "\n".join([LINE, "", "Call the supplier first.", LINE, "  "])
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake)
        installed(tmp_path / "models", AMD)
        route = translate_route(TranslationApi(translation, EventHub(SimpleEventBus()), enabled=lambda: True))
        answer = route(ApiRequest({}, {}, {"text": text}))
    assert answer.status == 200 and isinstance(answer.body, dict)
    lines = answer.body["lines"]
    assert [None if line is None else line["english"] for line in lines] == [f"EN: {LINE}", None, None, f"EN: {LINE}", None]
    assert lines[0]["words"][-1] == ["المزرعة.", "gloss of المزرعة."]


@pytest.mark.parametrize(
    ("enabled", "install", "text", "status"),
    [(False, True, LINE, 409), (True, False, LINE, 409), (True, True, "ز" * 60_001, 400), (True, True, None, 400)],
    ids=["off", "downloading", "too long", "no text"],
)
def test_history_is_refused_english_while_it_is_off_downloading_or_given_too_much(
    tmp_path: Path, enabled: bool, install: bool, text: str | None, status: int,
) -> None:
    with FakeLlamaServer() as fake:
        translation, _downloads, _servers = service(tmp_path, fake)
        if install:
            installed(tmp_path / "models", AMD)
        route = translate_route(TranslationApi(translation, EventHub(SimpleEventBus()), enabled=lambda: enabled))
        with pytest.raises(ApiError) as refused:
            route(ApiRequest({}, {}, {"text": text}))
        assert refused.value.status == status and fake.requests == []
