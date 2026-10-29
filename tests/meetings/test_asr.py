"""What a meeting loads, and how it shares Qwen3-ASR with dictation, without torch or a GPU."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from skryb.meetings import asr
from skryb.meetings.asr import NO_GPU_WARNING, MeetingModels
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.engine_turns import EngineTurns
from skryb.transcription.result import TranscriptionResult
from tests.meetings.fakes import LevelLabeller

_CLIP = np.full(16000 * 3, 0.2, np.float32)


class FakeQwen(TranscriptionEngine):
    def __init__(self, text: str = "Haan theek hai.") -> None:
        self.text = text
        self.calls: list[tuple[float, str | None, str | None]] = []
        self.loaded = False
        self.disposed = False
        self.hold: threading.Event | None = None

    def load_model(self) -> None:
        self.loaded = True

    def is_model_loaded(self) -> bool:
        return self.loaded

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        raise AssertionError("meetings always pass their prompt")

    def transcribe_prompted(self, audio: AudioSegment, context: str | None, language: str | None) -> TranscriptionResult:
        if self.hold is not None:
            self.hold.wait(5)
        self.calls.append((audio.duration_seconds, context, language))
        return TranscriptionResult(self.text, language or "", 0.0, 0.01)

    def dispose(self) -> None:
        self.disposed = True


class FakeParakeet(TranscriptionEngine):
    def load_model(self) -> None: ...
    def is_model_loaded(self) -> bool:
        return True

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        return TranscriptionResult("dictated", "en", 1.0, 0.01)

    def dispose(self) -> None: ...


@pytest.fixture(autouse=True)
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "Vad", lambda: LevelLabeller(0.032, speech_only=True))
    monkeypatch.setattr(asr, "Diarizer", lambda snapshot: LevelLabeller(0.01))
    monkeypatch.setattr(asr, "release_gpu", lambda: None)
    monkeypatch.setattr(asr, "gpu_available", lambda: True)


def _models(dictation: TranscriptionEngine, own: FakeQwen | None = None, language: str = "en", turns: EngineTurns | None = None) -> tuple[MeetingModels, list[FakeQwen]]:
    built: list[FakeQwen] = []

    def build() -> FakeQwen:
        built.append(own or FakeQwen("own copy"))
        return built[-1]

    return MeetingModels(
        language=language, dictation_engine=lambda: dictation, turns=turns or EngineTurns(),
        build_asr=build, diarizer_snapshot=lambda: Path("snapshot"),
    ), built


def test_a_meeting_borrows_dictations_qwen3_asr_instead_of_loading_a_second_copy() -> None:
    dictation = FakeQwen("from the shared model")
    models, built = _models(dictation)

    models.speaker_labeller()
    text = models.transcribe(_CLIP, "Names and terms that may come up: Northwind.")

    assert text == "from the shared model"
    assert built == []
    assert dictation.calls == [(3.0, "Names and terms that may come up: Northwind.", "en")]
    models.release()
    assert not dictation.disposed


def test_when_dictation_uses_another_model_the_meeting_loads_its_own_and_unloads_it_at_the_end() -> None:
    models, built = _models(FakeParakeet())

    models.speaker_labeller()
    assert len(built) == 1 and built[0].loaded

    assert models.transcribe(_CLIP, None) == "own copy"
    models.release()
    assert built[0].disposed


@pytest.mark.parametrize("setting, asked", [("en", "en"), ("ur", "hi"), ("auto", None), ("klingon", "en")])
def test_the_meeting_language_setting_is_what_qwen3_asr_is_asked_for(setting: str, asked: str | None) -> None:
    dictation = FakeQwen()
    models, _built = _models(dictation, language=setting)

    models.transcribe(_CLIP, None)

    assert dictation.calls[0][2] == asked


def test_a_meeting_clip_waits_while_a_dictation_is_decoding_on_the_shared_model() -> None:
    turns = EngineTurns()
    dictation = FakeQwen()
    models, _built = _models(dictation, turns=turns)
    finished = threading.Event()

    with turns.dictation():
        clip = threading.Thread(target=lambda: (models.transcribe(_CLIP, None), finished.set()))
        clip.start()
        time.sleep(0.05)
        assert not finished.is_set()
    clip.join(5)
    assert finished.is_set()


def test_a_meeting_with_its_own_copy_never_holds_up_dictation() -> None:
    turns = EngineTurns()
    own = FakeQwen("own copy")
    own.hold = threading.Event()
    models, _built = _models(FakeParakeet(), own=own, turns=turns)
    models.speaker_labeller()
    clip = threading.Thread(target=models.transcribe, args=(_CLIP, None))
    clip.start()
    time.sleep(0.05)

    started = time.monotonic()
    with turns.dictation():
        waited = time.monotonic() - started
    own.hold.set()
    clip.join(5)

    assert waited < 0.5


def test_without_a_gpu_voices_on_the_speakers_are_not_told_apart_and_the_window_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "gpu_available", lambda: False)
    models, _built = _models(FakeQwen())

    labeller, warning = models.speaker_labeller()

    assert labeller.frame_seconds == 0.032
    assert warning == NO_GPU_WARNING
