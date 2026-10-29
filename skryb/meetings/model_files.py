"""The two models a meeting needs on disk, and fetching the ones that are missing."""

from collections.abc import Callable

from skryb.meetings.asr import QWEN3_ASR
from skryb.meetings.models import DIARIZER
from skryb.transcription.model_info import get_model_info

_DISPLAY_NAMES = {DIARIZER.name: "Speaker labels", QWEN3_ASR: "Qwen3-ASR speech-to-text"}


class MeetingModelFiles:
    def __init__(
        self,
        *,
        diarizer_installed: Callable[[], bool],
        download_diarizer: Callable[[], None],
        is_installed: Callable[[str], bool],
        start_download: Callable[[str], None],
    ) -> None:
        self._diarizer_installed = diarizer_installed
        self._download_diarizer = download_diarizer
        self._is_installed = is_installed
        self._start_download = start_download

    def installed(self, name: str) -> bool:
        return self._diarizer_installed() if name == DIARIZER.name else self._is_installed(name)

    def views(self) -> list[dict[str, object]]:
        return [
            {
                "name": info.name, "label": _DISPLAY_NAMES[info.name], "display_size": info.display_size,
                "description": info.description, "installed": self.installed(info.name),
            }
            for info in (DIARIZER, get_model_info(QWEN3_ASR))
        ]

    def missing(self) -> list[str]:
        return [_DISPLAY_NAMES[name] for name in (DIARIZER.name, QWEN3_ASR) if not self.installed(name)]

    def download_missing(self) -> None:
        if not self.installed(DIARIZER.name):
            self._download_diarizer()
        if not self.installed(QWEN3_ASR):
            self._start_download(QWEN3_ASR)
