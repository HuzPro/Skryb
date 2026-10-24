"""Build the list of (model, engine, device) benchmark combinations.

Per user constraint: GPU-only - no CPU benchmarks for any model. Each
combo pairs a model with every engine in its ``compatible_engines``;
any engine that cannot run on this hardware is captured in the JSON as
``status='error'`` rather than skipped, so the report still reflects
what's possible on this hardware.

The matrix stays the full catalog, licence-gated entries included. The
gate belongs on the two paths that actually acquire and run weights
(``_pre_download`` and ``_runner``), which both refuse a gated model
nobody has accepted; filtering here instead would either abort every
sweep at import time or drop a model from the report in silence.

The list is computed once at module load. Importing :mod:`model_info`
is safe - it pulls only ABC/Literal definitions, no torch or HF.
"""

from dataclasses import dataclass

from skryb.transcription.engine_types import EngineId
from skryb.transcription.model_info import ALL_MODELS


@dataclass(frozen=True, slots=True)
class BenchmarkCombo:
    model_name: str
    engine_id: EngineId
    device: str  # always "cuda" - see module docstring

    @property
    def test_id(self) -> str:
        return f"{self.model_name}-{self.engine_id}-{self.device}"


def build_benchmark_combinations() -> list[BenchmarkCombo]:
    combos: list[BenchmarkCombo] = []
    for model_name, info in ALL_MODELS.items():
        for engine_id in info.compatible_engines:
            combos.append(BenchmarkCombo(model_name, engine_id, "cuda"))
    return combos


BENCHMARK_COMBOS: list[BenchmarkCombo] = build_benchmark_combinations()
