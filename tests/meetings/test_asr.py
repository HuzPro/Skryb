"""What a meeting loads, how it shares Qwen3-ASR with dictation, and which model writes each line, without torch or a GPU."""

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


class FakeApex:
    """The Roman Urdu model: hands back its lines in order, one per utterance."""

    built: list["FakeApex"] = []
    lines: list[str] = []

    def __init__(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        self.heard: list[float] = []
        FakeApex.built.append(self)

    def transcribe(self, audio: np.ndarray) -> str:
        assert audio.dtype == np.float32 and audio.flags.c_contiguous
        self.heard.append(len(audio) / 16000)
        return FakeApex.lines.pop(0)


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
    monkeypatch.setattr(asr, "RomanUrduAsr", FakeApex)
    FakeApex.built, FakeApex.lines = [], []


def _models(dictation: TranscriptionEngine, own: FakeQwen | None = None, language: str = "en", turns: EngineTurns | None = None) -> tuple[MeetingModels, list[FakeQwen]]:
    built: list[FakeQwen] = []

    def build() -> FakeQwen:
        built.append(own or FakeQwen("own copy"))
        return built[-1]

    return MeetingModels(
        language=language, dictation_engine=lambda: dictation, turns=turns or EngineTurns(),
        build_asr=build, diarizer_snapshot=lambda: Path("snapshot"), roman_urdu_snapshot=lambda: Path("apex"),
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


@pytest.mark.parametrize("setting", ["en", "ur", "auto", "klingon"])
def test_qwen3_asr_is_always_asked_for_english_since_urdu_lines_come_from_the_roman_urdu_model(setting: str) -> None:
    dictation = FakeQwen()
    models, _built = _models(dictation, language=setting)
    FakeApex.lines = ["Let's look at the quarterly numbers."]

    models.transcribe(_CLIP, None)

    assert dictation.calls[0][2] == "en"


def test_an_english_meeting_never_loads_the_roman_urdu_model() -> None:
    dictation = FakeQwen("Let's look at the quarterly numbers.")
    models, _built = _models(dictation, language="en")

    models.speaker_labeller()
    text = models.transcribe(_CLIP, None)

    assert text == "Let's look at the quarterly numbers."
    assert FakeApex.built == []


@pytest.mark.parametrize("setting", ["ur", "auto"])
def test_an_urdu_meeting_keeps_the_roman_urdu_line_spelled_the_urdu_way(setting: str) -> None:
    dictation = FakeQwen("should never be asked")
    models, _built = _models(dictation, language=setting)
    FakeApex.lines = ["Vah kah rahe the ki yah free text box basically kaam nahin karta."]

    text = models.transcribe(_CLIP, "Names and terms that may come up: Northwind.")

    assert text == "Woh kah rahe the ki yeh free text box basically kaam nahin karta."
    assert dictation.calls == []
    assert FakeApex.built[0].snapshot == Path("apex") and FakeApex.built[0].heard == [3.0]


@pytest.mark.parametrize("apex_line", [
    "So the estimator connects to LedgerLine first.",
    "nan",
    " NaN ",
    "",
    "haan haan haan haan haan haan haan",
    "Theek hai.",
])
def test_an_urdu_meeting_hands_english_empty_and_runaway_lines_to_qwen3_asr_with_the_vocabulary(apex_line: str) -> None:
    dictation = FakeQwen("So the estimator connects to LedgerLine first.")
    models, _built = _models(dictation, language="ur")
    FakeApex.lines = [apex_line]

    text = models.transcribe(_CLIP, "Names and terms that may come up: LedgerLine.")

    assert text == "So the estimator connects to LedgerLine first."
    assert dictation.calls == [(3.0, "Names and terms that may come up: LedgerLine.", "en")]


def test_an_urdu_meeting_loads_the_roman_urdu_model_once_at_the_start_and_lets_go_of_it_at_the_end() -> None:
    models, _built = _models(FakeQwen(), language="ur")
    FakeApex.lines = ["Main kal tak woh sheet bhej dunga, theek hai?"] * 3

    models.speaker_labeller()
    for _ in range(3):
        models.transcribe(_CLIP, None)
    assert len(FakeApex.built) == 1
    models.release()

    assert models._roman_urdu is None


@pytest.mark.parametrize(("line", "urdu"), [
    ("Haan theek hai, main kal tak woh sheet bhej dunga.", True),
    ("Aap ka experience kya raha?", True),
    ("Yah sahi hai.", True),  # two Urdu words: the least that counts
    ("Theek hai.", False),  # one is not enough
    ("The main point is that we ship on Friday.", False),
    ("Bachon ko pata hai ke unhein kya karna hai, you know, in the first place.", True),
    ("Pakistan versus wahan par?", False),
    ("nan", False),
    ("NaN", False),
    ("", False),
    ("   ", False),
    ("hai hai hai hai hai hai", False),  # six in a row is a runaway loop
    ("hai hai hai hai hai", True),  # five is still speech
    ("Café naïve résumé, Zoë's ka matlab hai", True),
    ("وہاں پر کیا ہو رہا ہے", False),  # Urdu script has no Roman words to count
])
def test_a_roman_urdu_line_is_kept_only_when_it_reads_as_urdu(line: str, urdu: bool) -> None:
    assert asr.reads_as_urdu(line) is urdu


def test_judging_a_runaway_line_thousands_of_words_long_stays_fast() -> None:
    line = " ".join(["main", "kal", "tak", "woh", "sheet", "bhej", "dunga"] * 700)
    started = time.monotonic()

    assert asr.reads_as_urdu(line)
    assert not asr.reads_as_urdu(line + " theek" * 3000)
    assert time.monotonic() - started < 1.0


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


def test_loading_the_speech_detector_leaves_torch_on_every_thread_it_had() -> None:
    import subprocess
    import sys

    probe = (
        "import torch; torch.set_num_threads(4);"
        "from skryb.meetings.models import Vad; Vad();"
        "print(torch.get_num_threads())"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip().splitlines()[-1] == "4"


class _FakeNativeDiarizer(LevelLabeller):
    def __init__(self, snapshot: Path, dll: Path | None = None, fail: bool = False) -> None:
        if fail:
            raise asr.TranscriptionError("hipMalloc: out of memory")
        super().__init__(0.01)
        self.snapshot, self.dll, self.disposed = snapshot, dll, False

    def dispose(self) -> None:
        self.disposed = True


def test_on_an_rx_9060_xt_the_native_diarizer_tells_the_speakers_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "NativeDiarizer", _FakeNativeDiarizer)
    monkeypatch.setattr(asr, "gpu_available", lambda: pytest.fail("the GPU check imports torch"))
    models, _built = _models(FakeQwen())
    models._native_diarizer = lambda: Path("nd.dll")

    labeller, warning = models.speaker_labeller()

    assert isinstance(labeller, _FakeNativeDiarizer) and labeller.snapshot == Path("snapshot")
    assert labeller.dll == Path("nd.dll")
    assert warning is None


def test_a_native_diarizer_that_fails_to_load_leaves_the_pytorch_one_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "NativeDiarizer", lambda snapshot, dll: _FakeNativeDiarizer(snapshot, dll, fail=True))
    models, _built = _models(FakeQwen())
    models._native_diarizer = lambda: Path("nd.dll")

    labeller, warning = models.speaker_labeller()

    assert not isinstance(labeller, _FakeNativeDiarizer) and labeller.frame_seconds == 0.01
    assert warning is None


def test_the_native_diarizer_lets_go_of_the_gpu_when_the_meeting_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "NativeDiarizer", _FakeNativeDiarizer)
    models, _built = _models(FakeQwen())
    models._native_diarizer = lambda: Path("nd.dll")
    labeller, _ = models.speaker_labeller()

    models.release()

    assert labeller.disposed


@pytest.mark.parametrize(("kind", "gfx_arch", "installed", "chosen"), [
    ("amd_rocm", "gfx1200", True, "nd.dll"),
    ("amd_rocm", "gfx1100", True, None),
    ("amd_rocm", "gfx1200", False, None),
    ("nvidia_cuda", "sm_120", True, "nd_cuda.dll"),
    ("nvidia_cuda", "sm_86", True, "nd_cuda.dll"),
    ("nvidia_cuda", "sm_75", True, None),
    ("nvidia_cuda", "sm_120", False, None),
])
def test_the_native_diarizer_runs_on_an_rx_9060_xt_or_an_rtx_30_to_50_card_with_its_dll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, gfx_arch: str, installed: bool, chosen: str | None,
) -> None:
    from skryb.transcription.engine_types import HardwareProfile
    from tests.fakes.hardware import FakeHardwareDetector

    def installed_dll(cuda: bool = False) -> Path | None:
        return tmp_path / ("nd_cuda.dll" if cuda else "nd.dll") if installed else None

    monkeypatch.setattr(asr, "native_diarizer_dll", installed_dll)
    detector = FakeHardwareDetector(HardwareProfile(kind=kind, device_name="GPU", gfx_arch=gfx_arch))

    assert asr.native_diarizer_for(detector) == (None if chosen is None else tmp_path / chosen)
