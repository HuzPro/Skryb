"""Which models a meeting needs on disk, by meeting language, and what the download button fetches."""

import pytest

from skryb.meetings.model_files import MeetingModelFiles

_ALL = ("nemotron-3-diarization", "qwen3-asr-1.7b", "whisper-hindi2hinglish-apex")


class Rig:
    def __init__(self, language: str, *, missing: tuple[str, ...] = ()) -> None:
        self.language = language
        self.installed = {name: name not in missing for name in _ALL}
        self.meeting_only_downloads: list[str] = []
        self.dictation_downloads: list[str] = []
        self.files = MeetingModelFiles(
            language=lambda: self.language,
            meeting_only_installed=self._meeting_only_installed,
            download_meeting_only=self.meeting_only_downloads.append,
            is_installed=self._dictation_installed,
            start_download=self.dictation_downloads.append,
        )

    def _meeting_only_installed(self, name: str) -> bool:
        assert name != "qwen3-asr-1.7b", "Qwen3-ASR is one of dictation's models"
        return self.installed[name]

    def _dictation_installed(self, name: str) -> bool:
        assert name == "qwen3-asr-1.7b", f"{name} is a meeting-only model"
        return self.installed[name]


def test_an_english_meeting_is_ready_without_the_roman_urdu_model() -> None:
    rig = Rig("en", missing=("whisper-hindi2hinglish-apex",))

    assert rig.files.missing() == []


@pytest.mark.parametrize("language", ["ur", "auto"])
def test_urdu_and_auto_meetings_need_the_roman_urdu_model_too(language: str) -> None:
    rig = Rig(language, missing=_ALL)

    assert rig.files.missing() == ["Speaker labels", "Qwen3-ASR speech-to-text", "Roman Urdu speech-to-text"]


def test_a_language_passed_in_wins_over_the_meeting_setting() -> None:
    rig = Rig("en", missing=("whisper-hindi2hinglish-apex",))

    assert rig.files.missing("ur") == ["Roman Urdu speech-to-text"]


def test_the_models_card_lists_all_three_models_whatever_the_language() -> None:
    rig = Rig("en", missing=("whisper-hindi2hinglish-apex",))

    views = rig.files.views()

    assert [view["label"] for view in views] == [
        "Speaker labels", "Qwen3-ASR speech-to-text", "Roman Urdu speech-to-text",
    ]
    assert [view["installed"] for view in views] == [True, True, False]
    assert views[2]["display_size"] == "1.6 GB"


def test_a_fresh_english_install_downloads_only_what_english_meetings_need() -> None:
    rig = Rig("en", missing=_ALL)

    rig.files.download_missing()

    assert rig.meeting_only_downloads == ["nemotron-3-diarization"]
    assert rig.dictation_downloads == ["qwen3-asr-1.7b"]


def test_a_fresh_urdu_install_downloads_all_three() -> None:
    rig = Rig("ur", missing=_ALL)

    rig.files.download_missing()

    assert rig.meeting_only_downloads == ["nemotron-3-diarization", "whisper-hindi2hinglish-apex"]
    assert rig.dictation_downloads == ["qwen3-asr-1.7b"]


def test_with_english_ready_the_download_button_fetches_the_roman_urdu_model() -> None:
    rig = Rig("en", missing=("whisper-hindi2hinglish-apex",))

    rig.files.download_missing()

    assert rig.meeting_only_downloads == ["whisper-hindi2hinglish-apex"]
    assert rig.dictation_downloads == []


def test_with_everything_installed_the_download_button_fetches_nothing() -> None:
    rig = Rig("auto")

    rig.files.download_missing()

    assert rig.meeting_only_downloads == [] and rig.dictation_downloads == []
