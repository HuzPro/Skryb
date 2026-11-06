"""The models a meeting needs on disk, and fetching the ones that are missing.

Every meeting needs speaker labels and Qwen3-ASR; Urdu and Auto meetings
need the Roman Urdu model too. Arabic meetings need Jais-ASR in Qwen3-ASR's
place, unless Qwen3-ASR is downloaded and Jais-ASR isn't: then Qwen3-ASR
hears the Arabic, and no download is asked for.
"""

from collections.abc import Callable

from skryb.meetings.asr import JAIS_ASR, QWEN3_ASR, URDU_LANGUAGES
from skryb.meetings.models import DIARIZER, ROMAN_URDU
from skryb.transcription.model_info import get_model_info

_DISPLAY_NAMES = {
    DIARIZER.name: "Speaker labels", QWEN3_ASR: "Qwen3-ASR speech-to-text", ROMAN_URDU.name: "Roman Urdu speech-to-text",
    JAIS_ASR: "Jais-ASR Arabic speech-to-text",
}
# Downloaded by the meetings' own model manager; Qwen3-ASR is one of dictation's models.
_MEETING_ONLY = (DIARIZER.name, ROMAN_URDU.name)


class MeetingModelFiles:
    def __init__(
        self,
        *,
        language: Callable[[], str],
        meeting_only_installed: Callable[[str], bool],
        download_meeting_only: Callable[[str], None],
        is_installed: Callable[[str], bool],
        start_download: Callable[[str], None],
    ) -> None:
        self._language = language
        self._meeting_only_installed = meeting_only_installed
        self._download_meeting_only = download_meeting_only
        self._is_installed = is_installed
        self._start_download = start_download

    def installed(self, name: str) -> bool:
        return self._meeting_only_installed(name) if name in _MEETING_ONLY else self._is_installed(name)

    def views(self) -> list[dict[str, object]]:
        return [
            {
                "name": info.name, "label": _DISPLAY_NAMES[info.name], "display_size": info.display_size,
                "description": info.description, "installed": self.installed(info.name),
            }
            for info in (DIARIZER, get_model_info(QWEN3_ASR), ROMAN_URDU, get_model_info(JAIS_ASR))
        ]

    def asr_model(self, language: str) -> str:
        """The speech-to-text model a meeting in *language* runs."""
        return next(name for name in self._needed(language) if name in (QWEN3_ASR, JAIS_ASR))

    def missing(self, language: str | None = None) -> list[str]:
        """What a meeting in *language* (the meeting setting when not given) still needs to download."""
        return [_DISPLAY_NAMES[name] for name in self._needed(language or self._language()) if not self.installed(name)]

    def download_missing(self) -> None:
        """What the meeting setting still needs, or else every meeting model not downloaded yet."""
        names = [name for name in self._needed(self._language()) if not self.installed(name)]
        # Jais-ASR is 11 GB of Arabic: fetched for an Arabic meeting, never as one of "the rest".
        for name in names or [name for name in _DISPLAY_NAMES if name != JAIS_ASR and not self.installed(name)]:
            if name in _MEETING_ONLY:
                self._download_meeting_only(name)
            else:
                self._start_download(name)

    def _needed(self, language: str) -> tuple[str, ...]:
        if language == "ar":
            only_qwen = self.installed(QWEN3_ASR) and not self.installed(JAIS_ASR)
            return (DIARIZER.name, QWEN3_ASR if only_qwen else JAIS_ASR)
        everyone = (DIARIZER.name, QWEN3_ASR)
        return (*everyone, ROMAN_URDU.name) if language in URDU_LANGUAGES else everyone
