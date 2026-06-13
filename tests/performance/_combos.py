"""Build the list of (model, engine, device) benchmark combinations.

Per user constraint: GPU-only — no CPU benchmarks for any model. Each
combo pairs a model with every engine in its ``compatible_engines``;
any engine that cannot run on this hardware is captured in the JSON as
``status='error'`` rather than skipped, so the report still reflects
what's possible on this hardware.

The list is computed once at module load. Importing :mod:`model_info`
is safe — it pulls only ABC/Literal definitions, no torch or HF.
"""

from dataclasses import dataclass

from voicetotext.transcription.engine_types import EngineId
from voicetotext.transcription.model_info import ALL_MODELS


@dataclass(frozen=True, slots=True)
class BenchmarkCombo:
    model_name: str
    engine_id: EngineId
    device: str  # always "cuda" — see module docstring

    @property
    def test_id(self) -> str:
        return f"{self.model_name}-{self.engine_id}-{self.device}"


def build_benchmark_combinations() -> list[BenchmarkCombo]:
    combos: list[BenchmarkCombo] = []
    for model_name, info in ALL_MODELS.items():
        for engine_id in info.compatible_engines:
            combos.append(BenchmarkCombo(model_name, engine_id, "cuda"))
    return _ordered_for_fragile_models_first(combos)


# Models that historically corrupt process state on this hardware
# when loaded AFTER other large models. We schedule them at the start
# of the sweep where the process is cleanest. Granite's multi-shard
# safetensors load via ``spawn_materialize`` has been observed to
# segfault when called after Parakeet on this AMD ROCm Windows host.
_FRAGILE_MODELS = ("granite-speech-4.1-2b",)


def _ordered_for_fragile_models_first(
    combos: list[BenchmarkCombo],
) -> list[BenchmarkCombo]:
    fragile = [c for c in combos if c.model_name in _FRAGILE_MODELS]
    other = [c for c in combos if c.model_name not in _FRAGILE_MODELS]
    return fragile + other


BENCHMARK_COMBOS: list[BenchmarkCombo] = build_benchmark_combinations()
