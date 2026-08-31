from pathlib import Path

import pytest

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_registry import EngineRegistry, EngineSpec
from voicetotext.transcription.engine_types import EngineDescriptor
from voicetotext.transcription.result import TranscriptionResult

_DESCRIPTOR = EngineDescriptor(
    engine_id="nano_parakeet",
    display_name="nano-parakeet",
    supported_hardware=("cpu",),
    base_priority=50,
)


class _FakeEngine(TranscriptionEngine):

    def __init__(self, spec: EngineSpec) -> None:
        self.spec = spec

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        return TranscriptionResult(
            text="", language="en", language_probability=0.0, transcription_seconds=0.0,
        )

    def load_model(self) -> None: ...
    def is_model_loaded(self) -> bool: return False
    def dispose(self) -> None: ...


def _spec(*, repo_id: str = "nvidia/parakeet-tdt-0.6b-v3") -> EngineSpec:
    return EngineSpec(
        model_name="parakeet-tdt-0.6b-v3", repo_id=repo_id,
        device="cpu", language="en",
        beam_size=5, model_dir=Path("/tmp/models"),
    )


class TestEngineRegistry:
    def test_should_build_engine_via_registered_builder(self):
        registry = EngineRegistry(
            descriptors=[_DESCRIPTOR],
            builders={"nano_parakeet": _FakeEngine},
        )
        spec = _spec()

        engine = registry.build("nano_parakeet", spec)

        assert isinstance(engine, _FakeEngine)
        assert engine.spec is spec

    def test_should_raise_on_unknown_engine_id(self):
        registry = EngineRegistry(
            descriptors=[_DESCRIPTOR],
            builders={"nano_parakeet": _FakeEngine},
        )

        with pytest.raises(KeyError):
            registry.build("nemotron_streaming", _spec())

    def test_should_return_descriptor_by_id(self):
        registry = EngineRegistry(
            descriptors=[_DESCRIPTOR],
            builders={"nano_parakeet": _FakeEngine},
        )

        assert registry.descriptor("nano_parakeet") is _DESCRIPTOR

    def test_all_descriptors_should_return_tuple(self):
        registry = EngineRegistry(
            descriptors=[_DESCRIPTOR],
            builders={"nano_parakeet": _FakeEngine},
        )

        assert registry.all_descriptors() == (_DESCRIPTOR,)


class TestEngineSpec:
    def test_should_carry_engine_specific_repo_id(self):
        spec = EngineSpec(
            model_name="parakeet-tdt-0.6b-v3",
            repo_id="nvidia/parakeet-tdt-0.6b-v3",
            device="cuda", language="en", beam_size=5, model_dir=Path("/m"),
        )

        assert spec.repo_id == "nvidia/parakeet-tdt-0.6b-v3"

    def test_revision_should_default_to_tracking_main(self):
        assert _spec().revision == ""

    def test_should_carry_a_pinned_revision(self):
        spec = EngineSpec(
            model_name="crisperwhisper-2.0-turbo",
            repo_id="nyralabs/CrisperWhisper2.0_turbo",
            device="cuda", language="en", beam_size=1, model_dir=Path("/m"),
            revision="de0369c8a68025b7f6e86387b6eb5a3b369787c8",
        )

        assert spec.revision == "de0369c8a68025b7f6e86387b6eb5a3b369787c8"
