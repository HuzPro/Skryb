"""Everything a meeting loads when it starts and lets go of when it stops.

Speech-to-text is Qwen3-ASR. When dictation already has it loaded, the
meeting borrows that instance between dictations; otherwise it loads its
own, on the engine Skryb picks for this computer, with dictation's engine
drafting its text when it can use a draft (the NPU one). Urdu and Auto
meetings hear each line with Whisper Hindi2Hinglish Apex first, since
Qwen3-ASR has no Urdu, and keep Qwen3-ASR for the lines that come out English.
Arabic meetings run Jais-ASR when it is downloaded, else ask Qwen3-ASR for Arabic.
"""

import importlib.util
import logging
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.core.paths import AppPaths
from skryb.meetings.capture import RATE
from skryb.meetings.diarization import NativeDiarizer
from skryb.meetings.openvino_diarizer import OpenVinoDiarizer
from skryb.meetings.models import (
    DIARIZER, ROMAN_URDU, Diarizer, Labeller, NpuRomanUrduAsr, RomanUrduAsr, Vad, gpu_available, release_gpu,
)
from skryb.people.hints import transcribe_with_hint
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.engine_registry import EngineRegistry, EngineSpec
from skryb.transcription.engine_selector import rank_engines
from skryb.transcription.engine_turns import EngineTurns
from skryb.transcription.hardware_detector import HardwareDetector
from skryb.transcription.hf_model_manager import loadable_snapshot_dir
from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import get_model_info
from skryb.transcription.native_engine_dll import NATIVE_GPU_ARCHS, native_diarizer_dll
from skryb.transcription.qwen3_asr_prompt import PromptedTranscriber
from skryb.transcription.roman_urdu import urdu_spelling

QWEN3_ASR = "qwen3-asr-1.7b"
JAIS_ASR = "jais-asr"
# The meeting languages that hear each line with the Roman Urdu model first.
URDU_LANGUAGES = ("ur", "auto")
# Roman Urdu words English doesn't use; "to", "na" and "ho" are English words too, so they don't count.
_URDU_WORDS = re.compile(
    r"\b(hai|hain|nahin|nahi|kya|ke|ki|ka|aur|main|mein|tha|thi|hum|ham|aap|aapka|yah|vah|yeh|woh|kar|raha|rahi|rahe"
    r"|bhi|se|ko|par)\b", re.IGNORECASE)
_LOOP = re.compile(r"\b(\w+)( \1\b){5,}")
NO_GPU_WARNING = "No GPU or NPU found, so speakers are not told apart: everyone on the call is Speaker 1."

_logger = logging.getLogger(__name__)


def build_meeting_asr(
    paths: AppPaths, registry: EngineRegistry, hardware_detector: HardwareDetector, license_policy: LicensePolicy,
    model: str = QWEN3_ASR,
) -> TranscriptionEngine:
    """*model* on the best engine for this computer: the native one on an RX 9060 XT or an RTX 30-50 card."""
    info = get_model_info(model)
    license_policy.assert_accepted(info)
    hardware = hardware_detector.detect()
    engine_id = rank_engines(info, hardware, registry.all_descriptors())[0].engine_id
    return registry.build(engine_id, EngineSpec(
        model_name=info.name, repo_id=info.repo_id, device="cpu" if hardware.kind == "cpu" else "cuda",
        language="en", beam_size=1, model_dir=paths.models_dir, revision=info.revision, hardware_kind=hardware.kind,
    ))


def diarizer_snapshot(models_dir: Path) -> Path:
    return loadable_snapshot_dir(models_dir, DIARIZER.repo_id, DIARIZER.revision)


def roman_urdu_snapshot(models_dir: Path) -> Path:
    return loadable_snapshot_dir(models_dir, ROMAN_URDU.repo_id, ROMAN_URDU.revision)


def reads_as_urdu(line: str) -> bool:
    """Whether a line from the Roman Urdu model is Urdu worth keeping.

    English lines go to Qwen3-ASR, which writes English far better. So do a
    line that repeats one word over and over and the "nan" the model writes
    for some short English clips.
    """
    return line.strip().lower() != "nan" and not _LOOP.search(line) and len(_URDU_WORDS.findall(line)) >= 2


def native_diarizer_for(hardware_detector: HardwareDetector) -> Path | None:
    """The native diarizer's DLL for this GPU: an RX 9060 XT, or the NVIDIA build on an RTX 30-50 card. None when
    the GPU is another one or the DLL isn't installed."""
    hardware = hardware_detector.detect()
    if hardware.gfx_arch not in NATIVE_GPU_ARCHS:
        return None
    return native_diarizer_dll(cuda=hardware.kind == "nvidia_cuda")


class MeetingModels:
    """Loaded at meeting start by ``speaker_labeller``, released by ``release`` at stop."""

    def __init__(
        self,
        *,
        language: str,
        dictation_engine: Callable[[], TranscriptionEngine],
        turns: EngineTurns,
        build_asr: Callable[[], TranscriptionEngine],
        diarizer_snapshot: Callable[[], Path],
        roman_urdu_snapshot: Callable[[], Path],
        native_diarizer: Callable[[], Path | None] = lambda: None,
        asr_model: str = QWEN3_ASR,
    ) -> None:
        self._urdu = language in URDU_LANGUAGES
        self._asr_language = "ar" if language == "ar" else "en"
        self._jais = asr_model == JAIS_ASR
        self._dictation_engine = dictation_engine
        self._turns = turns
        self._build_asr = build_asr
        self._diarizer_snapshot = diarizer_snapshot
        self._roman_urdu_snapshot = roman_urdu_snapshot
        self._native_diarizer = native_diarizer
        self._own: TranscriptionEngine | None = None
        self._diarizer: Labeller | None = None
        self._roman_urdu: RomanUrduAsr | NpuRomanUrduAsr | None = None

    def voice_detector(self) -> Labeller:
        return Vad()

    def speaker_labeller(self) -> tuple[Labeller, str | None]:
        """Who is talking on the speakers, and a warning when that cannot be told."""
        if not self._borrowed_engine_is_prompted():
            self._own_engine()
        if self._urdu:
            self._roman_urdu_engine()
        native_dll = self._native_diarizer()
        if native_dll is not None:
            try:
                self._diarizer = NativeDiarizer(self._diarizer_snapshot(), native_dll)
                return self._diarizer, None
            except TranscriptionError as error:
                _logger.warning("The native diarizer didn't load, so the PyTorch one runs: %s", error)
        if not gpu_available():
            return self._npu_diarizer()
        self._diarizer = Diarizer(self._diarizer_snapshot())
        return self._diarizer, None

    def _npu_diarizer(self) -> tuple[Labeller, str | None]:
        """Without a GPU the model step runs on an Intel NPU through OpenVINO; on the CPU alone it falls far
        behind live audio, so then everyone on the call is one speaker."""
        if importlib.util.find_spec("openvino") is None:
            return Vad(), NO_GPU_WARNING
        try:
            self._diarizer = OpenVinoDiarizer(self._diarizer_snapshot())
        except Exception:
            _logger.warning("The NPU diarizer didn't load, so speakers are not told apart", exc_info=True)
            return Vad(), NO_GPU_WARNING
        if self._diarizer.device != "NPU":
            self._diarizer = None
            return Vad(), NO_GPU_WARNING
        return self._diarizer, None

    def transcribe(self, audio: np.ndarray, context: str | None) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if self._urdu:
            line = self._roman_urdu_engine().transcribe(audio)
            if reads_as_urdu(line):
                return urdu_spelling(line)
        clip = AudioSegment(audio, RATE)
        with self._turns.meeting():
            engine = self._dictation_engine()
            if self._can_borrow(engine):
                return transcribe_with_hint(engine, clip, context, self._asr_language).text
        # ponytail: if dictation switches to Qwen3-ASR mid-meeting, the meeting keeps its own copy until it ends.
        return transcribe_with_hint(self._own_engine(), clip, context, self._asr_language).text

    def release(self) -> None:
        if isinstance(self._diarizer, (NativeDiarizer, OpenVinoDiarizer)):
            self._diarizer.dispose()
        self._diarizer = None
        if isinstance(self._roman_urdu, NpuRomanUrduAsr):
            self._roman_urdu.dispose()
        self._roman_urdu = None
        if self._own is not None:
            self._own.dispose()
            self._own = None
        release_gpu()

    def _borrowed_engine_is_prompted(self) -> bool:
        return self._can_borrow(self._dictation_engine())

    def _can_borrow(self, engine: TranscriptionEngine) -> bool:
        """Whether dictation's engine is the model this meeting runs: Jais-ASR, or else Qwen3-ASR."""
        return isinstance(engine, PromptedTranscriber) and _is_jais(engine) == self._jais

    def _own_engine(self) -> PromptedTranscriber:
        if self._own is None:
            engine = self._build_asr()
            engine.load_model()
            # The NPU engine checks a draft a pass at a time; dictation's engine writes one in a fraction of that.
            if hasattr(engine, "drafter"):
                engine.drafter = self._dictation_draft
            self._own = engine
            _logger.info("Loaded %s for the meeting: dictation uses another model.", "Jais-ASR" if self._jais else "Qwen3-ASR")
        assert isinstance(self._own, PromptedTranscriber)
        return self._own

    def _dictation_draft(self, clip: AudioSegment) -> str:
        with self._turns.meeting():
            return self._dictation_engine().transcribe(clip).text

    def _roman_urdu_engine(self) -> RomanUrduAsr | NpuRomanUrduAsr:
        if self._roman_urdu is None:
            self._roman_urdu = self._load_roman_urdu(self._roman_urdu_snapshot())
            _logger.info("Loaded Whisper Hindi2Hinglish Apex for an Urdu meeting.")
        return self._roman_urdu

    @staticmethod
    def _load_roman_urdu(snapshot: Path) -> RomanUrduAsr | NpuRomanUrduAsr:
        """On the GPU through PyTorch; without one, its encoder on the NPU, 4-6x faster than PyTorch on the CPU."""
        if gpu_available() or importlib.util.find_spec("openvino") is None:
            return RomanUrduAsr(snapshot)
        try:
            return NpuRomanUrduAsr(snapshot)
        except Exception:
            _logger.warning("Whisper Hindi2Hinglish Apex didn't load on the NPU; it runs on PyTorch", exc_info=True)
            return RomanUrduAsr(snapshot)


def _is_jais(engine: TranscriptionEngine) -> bool:
    from skryb.transcription.native_jais_asr_adapter import NativeJaisAsrAdapter
    from skryb.transcription.openvino_jais_asr_adapter import OpenVinoJaisAsrAdapter

    return isinstance(engine, (NativeJaisAsrAdapter, OpenVinoJaisAsrAdapter))
