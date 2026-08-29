from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_types import (
    CrisperWhisperMode,
    EngineDescriptor,
    EngineId,
)


@dataclass(frozen=True)
class EngineSpec:
    """Inputs every engine needs to build itself.

    Threaded through :class:`EngineBuilder` so the registry can stay
    ignorant of any engine-specific config: dtype overrides,
    compute-type heuristics, and similar live inside the adapters and
    pull from :class:`ConfigurationStore` directly. ``repo_id`` and
    ``revision`` come from ``ModelInfo`` so each adapter receives the HF
    repo it knows how to load, at the same commit the downloader fetched.
    An empty ``revision`` tracks the repo's ``main`` branch.
    """
    model_name: str
    repo_id: str
    device: str
    language: str
    beam_size: int
    model_dir: Path
    revision: str = ""
    crisperwhisper_mode: CrisperWhisperMode = "intended"


class EngineBuilder(Protocol):
    def __call__(self, spec: EngineSpec) -> TranscriptionEngine: ...


class EngineRegistry:
    """Registry + Factory Method for transcription engines.

    Engines register a descriptor (metadata for the selection layer)
    and a builder (callable that turns an :class:`EngineSpec` into a
    :class:`TranscriptionEngine`). The registry has no global state -
    instances are wired in :func:`voicetotext.bootstrap.build_app` and
    handed down explicitly.
    """

    def __init__(
        self,
        descriptors: Sequence[EngineDescriptor],
        builders: Mapping[EngineId, EngineBuilder],
    ) -> None:
        self._descriptors: dict[EngineId, EngineDescriptor] = {
            d.engine_id: d for d in descriptors
        }
        self._builders: dict[EngineId, EngineBuilder] = dict(builders)

    def descriptor(self, engine_id: EngineId) -> EngineDescriptor:
        return self._descriptors[engine_id]

    def all_descriptors(self) -> tuple[EngineDescriptor, ...]:
        return tuple(self._descriptors.values())

    def build(
        self, engine_id: EngineId, spec: EngineSpec,
    ) -> TranscriptionEngine:
        builder = self._builders[engine_id]
        return builder(spec)
