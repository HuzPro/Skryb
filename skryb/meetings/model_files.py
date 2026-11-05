"""The models a meeting needs on disk, and fetching the ones that are missing.

Every meeting needs speaker labels and Qwen3-ASR; Urdu and Auto meetings
need the Roman Urdu model too. Speaker labels are the checkpoint on a
computer with a GPU or a tuned Intel NPU, and NVIDIA's CPU build of it on one
with neither.
"""

from collections.abc import Callable

from skryb.meetings.asr import QWEN3_ASR, URDU_LANGUAGES
from skryb.meetings.models import DIARIZER, DIARIZER_CPU, ROMAN_URDU
from skryb.transcription.model_info import ModelInfo, get_model_info

_DISPLAY_NAMES = {
    DIARIZER.name: "Speaker labels", DIARIZER_CPU.name: "Speaker labels", QWEN3_ASR: "Qwen3-ASR speech-to-text",
    ROMAN_URDU.name: "Roman Urdu speech-to-text",
}
# Downloaded by the meetings' own model manager; Qwen3-ASR is one of dictation's models.
_MEETING_ONLY = (DIARIZER.name, DIARIZER_CPU.name, ROMAN_URDU.name)


class MeetingModelFiles:
    def __init__(
        self,
        *,
        language: Callable[[], str],
        meeting_only_installed: Callable[[str], bool],
        download_meeting_only: Callable[[str], None],
        is_installed: Callable[[str], bool],
        start_download: Callable[[str], None],
        diarizer: Callable[[], ModelInfo] = lambda: DIARIZER,
    ) -> None:
        self._language = language
        self._meeting_only_installed = meeting_only_installed
        self._download_meeting_only = download_meeting_only
        self._is_installed = is_installed
        self._start_download = start_download
        self._diarizer = diarizer

    def installed(self, name: str) -> bool:
        return self._meeting_only_installed(name) if name in _MEETING_ONLY else self._is_installed(name)

    def views(self) -> list[dict[str, object]]:
        return [
            {
                "name": info.name, "label": _DISPLAY_NAMES[info.name], "display_size": info.display_size,
                "description": info.description, "installed": self.installed(info.name),
            }
            for info in (self._diarizer(), get_model_info(QWEN3_ASR), ROMAN_URDU)
        ]

    def missing(self, language: str | None = None) -> list[str]:
        """What a meeting in *language* (the meeting setting when not given) still needs to download."""
        return [_DISPLAY_NAMES[name] for name in self._needed(language or self._language()) if not self.installed(name)]

    def download_missing(self) -> None:
        """What the meeting setting still needs, or else every meeting model not downloaded yet."""
        names = [name for name in self._needed(self._language()) if not self.installed(name)]
        for name in names or [name for name in self._all() if not self.installed(name)]:
            if name in _MEETING_ONLY:
                self._download_meeting_only(name)
            else:
                self._start_download(name)

    def _all(self) -> tuple[str, ...]:
        """Every meeting model this computer uses."""
        return self._diarizer().name, QWEN3_ASR, ROMAN_URDU.name

    def _needed(self, language: str) -> tuple[str, ...]:
        every = self._all()
        return every if language in URDU_LANGUAGES else every[:2]
