"""The writing model as the rest of the app sees it.

Bootstrap builds one of these; the output chain asks it for a cleanup stage,
the use case for a time estimate, and the Settings page for status and
actions. Status listeners are called on whatever thread changed the status.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voicetotext.edit_capture.window import ForegroundWindow
from voicetotext.output.dictionary_config import DictionaryConfig, LiteralRule
from voicetotext.writing.cleanup import CleanupStage, CleanupTimings
from voicetotext.writing.client import Endpoint, WritingClient, post_completion
from voicetotext.writing.install import (
    FileDownloader,
    WritingInstaller,
    WritingPaths,
    chunked_downloader,
    server_files,
)
from voicetotext.writing.learning import Learner
from voicetotext.writing.llama_server import LlamaServer, ServerState, ServerStatus
from voicetotext.writing.prompt import Hints, cleanup_system_prompt, cleanup_user_message
from voicetotext.writing.settings import WritingSettings
from voicetotext.writing.transforms import TransformStore

_logger = logging.getLogger(__name__)

SERVER_FAILED_NOTICE = (
    "The writing model could not start. Dictation works without cleanup. "
    "See Settings > Writing."
)
_WARM_UP_DICTATION = (
    "Please send the contract to Ayesha tomorrow morning, and tell Farhan the "
    "review moved to Thursday."
)
_WARM_UP_SECONDS = 60.0
_MAX_SPELLINGS = 60
_MAX_SPELLING_CHARS = 40


@dataclass(frozen=True)
class WritingStatus:
    """One of: not_installed, downloading, install_failed, starting, ready,
    failed, stopped."""

    state: str
    detail: str = ""
    bytes_done: int = 0
    bytes_total: int = 0
    last_cleanup_seconds: float | None = None


def spellings_from(dictionary: DictionaryConfig) -> tuple[str, ...]:
    """The dictionary's written forms that look like names or terms."""
    found: dict[str, None] = {}
    for rule in dictionary.rules:
        if not isinstance(rule, LiteralRule) or not rule.enabled:
            continue
        written = rule.replacement.strip()
        if (
            written
            and len(written) <= _MAX_SPELLING_CHARS
            and "\n" not in written
            and any(character.isalpha() for character in written)
        ):
            found[written] = None
    return tuple(found)[:_MAX_SPELLINGS]


def warm_up_writing_model(
    settings: Callable[[], WritingSettings], endpoint: Endpoint,
) -> None:
    """One real request, so the cached prompt and the GPU are ready."""
    post_completion(
        endpoint,
        cleanup_system_prompt(settings()),
        cleanup_user_message(_WARM_UP_DICTATION, Hints()),
        96,
        _WARM_UP_SECONDS,
    )


def _open_with_shell(path: Path) -> None:
    os.startfile(path)  # type: ignore[attr-defined]  # Windows only, like the app


class WritingService:
    def __init__(
        self,
        settings: Callable[[], WritingSettings],
        paths: WritingPaths,
        server: LlamaServer,
        learner: Learner,
        transforms: TransformStore,
        notify: Callable[[str], None],
        run_in_background: Callable[[Callable[[], None]], None],
        read_window: Callable[[], ForegroundWindow] = ForegroundWindow,
        make_downloader: Callable[[], FileDownloader] = chunked_downloader,
        open_path: Callable[[Path], None] = _open_with_shell,
    ) -> None:
        self._settings = settings
        self._paths = paths
        self._server = server
        self.learner = learner
        self._transforms = transforms
        self._notify = notify
        self._run_in_background = run_in_background
        self._read_window = read_window
        self._open_path = open_path
        self._listeners: list[Callable[[], None]] = []
        self._failure_announced = False
        self.client = WritingClient(server.endpoint)
        self.timings = CleanupTimings()
        self._installer = WritingInstaller(paths, self._on_install_changed, make_downloader)
        server.subscribe(self._on_server_status)

    # -- app side -------------------------------------------------------

    def start(self) -> None:
        self._server.start()

    def dispose(self) -> None:
        self._installer.shutdown()
        self._server.stop()

    def cleanup_stage(self, dictionary: DictionaryConfig) -> CleanupStage:
        settings = self._settings()
        transform_name = settings.after_dictation_transform
        return CleanupStage(
            settings,
            self.client,
            self.timings,
            spellings=spellings_from(dictionary),
            learned_style=self.learner.style_sentences,
            transform=lambda: self._transforms.instruction(transform_name),
            window=self._read_window,
        )

    def estimated_seconds(self, audio_seconds: float) -> float:
        if self._settings().cleanup == "off" or not self.client.is_ready():
            return 0.0
        return self.timings.estimate_for_audio(audio_seconds)

    # -- settings page side ----------------------------------------------

    def subscribe(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def status(self) -> WritingStatus:
        install = self._installer.progress
        if install.state == "downloading":
            return WritingStatus("downloading", "", install.bytes_done, install.bytes_total)
        if install.state == "failed":
            return WritingStatus("install_failed", install.error)
        server = self._server.status
        if server.state is ServerState.READY:
            return WritingStatus("ready", last_cleanup_seconds=self.timings.last_seconds())
        if server.state in (ServerState.STOPPED, ServerState.NOT_INSTALLED) and (
            server_files(self._paths, self._settings()) is None
        ):
            return WritingStatus("not_installed")
        return WritingStatus(server.state.value, server.detail)

    def install(self) -> None:
        self._installer.install()

    def cancel_install(self) -> None:
        self._installer.cancel()

    def retry(self) -> None:
        if self._installer.progress.state == "failed":
            self._installer.install()
            return
        self.restart_server()

    def restart_server(self) -> None:
        self._run_in_background(self._server.restart)

    def open_log(self) -> None:
        log = (
            self._paths.install_log if self._installer.progress.state == "failed"
            else self._server.log_path
        )
        if log.exists():
            self._open_path(log)

    def server_paths(self) -> tuple[Path, Path]:
        """The runtime and model the server launches: each override, else the managed file."""
        settings = self._settings()
        return (
            Path(settings.runtime_path) if settings.runtime_path else self._paths.runtime_exe,
            Path(settings.model_path) if settings.model_path else self._paths.model_file,
        )

    def learned_style_summary(self) -> str:
        return self.learner.style_summary()

    def forget_learned_style(self) -> None:
        self.learner.forget_style()

    def reset_learning(self) -> None:
        self.learner.reset()

    def forget_learned_words(self, pairs: tuple[tuple[str, str], ...]) -> None:
        if pairs:
            self.learner.block(pairs)

    def transforms(self) -> dict[str, str]:
        return self._transforms.all()

    def save_transforms(self, transforms: dict[str, str]) -> None:
        self._transforms.save(transforms)

    # -- internals -------------------------------------------------------

    def _on_install_changed(self) -> None:
        if self._installer.progress.state == "completed":
            self._server.restart()
        self._tell_listeners()

    def _on_server_status(self, status: ServerStatus) -> None:
        if status.state is ServerState.FAILED and not self._failure_announced:
            self._failure_announced = True
            self._notify(SERVER_FAILED_NOTICE)
        self._tell_listeners()

    def _tell_listeners(self) -> None:
        for listener in list(self._listeners):
            listener()
