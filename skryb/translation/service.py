"""Show English: Arabic lines translated and glossed on this computer.

The model for this computer's path downloads when Show English is switched on, and loads with the
first line that needs it, never at startup. Before it starts, the build is asked which devices it
sees and is pinned to one, with every layer on it, so a GPU or NPU that is missing or too small is
reported as such instead of the model quietly running on the processor.
"""

import functools
import importlib.util
import logging
import os
import queue
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from skryb.downloads.download_manager import DownloadJob
from skryb.transcription.engine_types import HardwareProfile
from skryb.translation import install
from skryb.translation.install import FileDownloader, InstallJob
from skryb.translation.llama_server import Device, LlamaServer, ServerError, ServerLaunch, chat, list_devices
from skryb.translation.models import TranslationPath, path_for
from skryb.translation.translator import Translation, Translator, has_arabic

_logger = logging.getLogger(__name__)
_CACHED_LINES = 512
_ROCM_PACKAGES = ("_rocm_sdk_libraries_custom", "_rocm_sdk_core")
_GPU_MAKERS = {"rocm": "AMD", "cuda": "NVIDIA"}

Done = Callable[[Translation], None]
ListDevices = Callable[[Path, Mapping[str, str]], list[tuple[str, str, int]]]


class Downloads(Protocol):
    """The part of the app's download manager this needs."""

    def enqueue(self, job: DownloadJob) -> object: ...


def chunked_downloader() -> FileDownloader:
    import requests

    from skryb.downloads.chunked_file_downloader import ChunkedFileDownloader
    from skryb.downloads.requests_http_client import RequestsHttpClient

    return ChunkedFileDownloader(http=RequestsHttpClient(session=requests.Session()))


def rocm_library_folders() -> list[Path]:
    """The ``bin`` folders of the ROCm wheels Skryb's AMD install has: hipBLAS and rocBLAS for the HIP build."""
    folders = []
    for package in _ROCM_PACKAGES:
        spec = importlib.util.find_spec(package)
        if spec is not None and spec.origin is not None:
            folders.append(Path(spec.origin).parent / "bin")
    return folders


class TranslationService:
    def __init__(
        self,
        *,
        models_dir: Path,
        log_path: Path,
        hardware: Callable[[], HardwareProfile],
        downloads: Downloads,
        make_downloader: Callable[[], FileDownloader] = chunked_downloader,
        rocm_libraries: Callable[[], list[Path]] = rocm_library_folders,
        devices: ListDevices = list_devices,
        make_server: Callable[[Callable[[], ServerLaunch], Path], LlamaServer] = LlamaServer,
    ) -> None:
        self._models_dir = models_dir
        self._hardware = hardware
        self._downloads = downloads
        self._make_downloader = make_downloader
        self._rocm_libraries = rocm_libraries
        self._devices = devices
        self._server = make_server(self._launch, log_path)
        self._translator = Translator(self._complete)
        self._device: Device | None = None
        self._failure = ""
        self._lines: queue.Queue[tuple[str, Done]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    @functools.cached_property
    def path(self) -> TranslationPath:
        return path_for(self._hardware())

    def installed(self) -> bool:
        return install.installed(self._models_dir, self.path)

    def install(self) -> None:
        """Download this computer's model and build, unless they are here; also clears a failure to start."""
        self._failure = ""
        if not self.installed():
            self._downloads.enqueue(InstallJob(self._models_dir, self.path, self._make_downloader()))

    def annotate(self, line: str) -> Translation:
        """*line* in English with a gloss under each word; ``ServerError`` when that can't be done."""
        if not has_arabic(line):
            return Translation(" ".join(line.split()), [(word, word) for word in line.split()])
        return self._annotated(" ".join(line.split()))

    @functools.lru_cache(maxsize=_CACHED_LINES)
    def _annotated(self, line: str) -> Translation:
        return self._translator.annotate(line)

    def submit(self, line: str, done: Done) -> None:
        """Annotate *line* on the translation thread, one line at a time, then call *done*; a failure is logged."""
        with self._lock:
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="translation", daemon=True)
                self._worker.start()
        self._lines.put((line, done))

    def view(self) -> dict[str, object]:
        path = self.path
        device = self._device
        return {
            "name": install.INSTALL_JOB_ID,
            "label": path.model.label,
            "description": path.model.description,
            "display_size": path.display_size,
            "installed": self.installed(),
            "device": None if device is None else device.kind,
            "device_name": None if device is None else device.name or None,
            "error": self._failure or None,
        }

    def dispose(self) -> None:
        self._server.stop()

    def _work(self) -> None:
        while True:
            line, done = self._lines.get()
            try:
                translation = self.annotate(line)
            except (ServerError, ValueError) as exc:
                _logger.warning("A line was left without English: %s", exc)
                continue
            done(translation)

    def _complete(self, prompt: str, max_tokens: int, grammar: str | None) -> str:
        if not self.installed():
            raise ServerError("The translation model isn't downloaded yet. Turn on Show English to download it.")
        if self._failure:
            raise ServerError(self._failure)
        try:
            endpoint = self._server.endpoint()
        except ServerError as exc:
            self._failure = str(exc)
            raise
        return chat(endpoint, prompt, max_tokens, grammar, options=dict(self.path.model.chat_options))

    def _launch(self) -> ServerLaunch:
        path = self.path
        runtime = install.runtime_exe(self._models_dir, path)
        env = self._environment(path)
        flags, self._device = self._device_for(path, runtime, env)
        _logger.info("Translation runs %s on the %s%s.", path.model.label, self._device.kind, f" ({self._device.name})" if self._device.name else "")
        return ServerLaunch(runtime, install.model_file(self._models_dir, path.model), (*path.flags, *flags), env)

    def _environment(self, path: TranslationPath) -> dict[str, str]:
        if path.name == "openvino":
            return {"GGML_OPENVINO_DEVICE": "NPU"}
        if path.name != "rocm":
            return {}
        folders = [folder for folder in self._rocm_libraries() if folder.is_dir()]
        if not folders:
            raise ServerError("Translation on an AMD graphics card needs the ROCm libraries Skryb's AMD install comes with.")
        return {"PATH": os.pathsep.join([*map(str, folders), os.environ.get("PATH", "")])}

    def _device_for(self, path: TranslationPath, runtime: Path, env: Mapping[str, str]) -> tuple[tuple[str, ...], Device]:
        if path.name == "cpu":
            return (), Device("CPU")
        listed = self._devices(runtime, env)
        if path.name == "openvino":
            npu = next(((key, name) for key, name, _mib in listed if "GGML_OPENVINO_DEVICE=NPU" in name), None)
            if npu is None:
                raise ServerError("OpenVINO found no NPU on this computer, so translation can't run on it.")
            return ("--device", npu[0]), Device("NPU", npu[1].split(" - ", 1)[-1].strip())
        if not listed:
            raise ServerError(f"llama.cpp found no {_GPU_MAKERS[path.name]} graphics card, so translation can't run on it.")
        key, name, _mib = max(listed, key=lambda device: device[2])
        return ("--device", key), Device("GPU", name)
