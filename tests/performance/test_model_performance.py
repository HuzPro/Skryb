"""Parametrized performance benchmarks across every supported
(model, engine, device) combination.

Failures during engine build/load/transcribe are *recorded*, not
raised, so the full sweep always completes regardless of individual
broken combinations. The pytest layer's only job is to drive the
runner and persist results.
"""

import pytest

from voicetotext.recording.audio_segment import AudioSegment

from tests.performance._combos import BENCHMARK_COMBOS, BenchmarkCombo
from tests.performance._identity import BenchmarkIdentity
from tests.performance._runner import run_benchmark
from tests.performance.conftest import PerformanceContext


_SHORT_AUDIO_FIXTURE_NAME = "Testing_1_2_3.mp3"
_LONG_AUDIO_FIXTURE_NAME = "LongStory.mp3"

_TIMED_RUNS_SHORT = 3
_TIMED_RUNS_LONG = 1

# Models whose load is observed to corrupt process state on second
# invocation — the safetensors multi-thread materializer in transformers
# segfaults on this AMD ROCm Windows host when called twice. To preserve
# the rest of the sweep, we benchmark the SHORT-audio cycle for these
# models and record a synthetic skip entry for LONG-audio rather than
# attempt a second load.
_FRAGILE_ON_RELOAD = frozenset({"granite-speech-4.1-2b"})


# pyproject.toml sets ``filterwarnings = ["error"]`` which makes every
# benign UserWarning / DeprecationWarning / ResourceWarning emitted by
# huggingface_hub, transformers, torch, and the Windows tempfile GC
# finalizer abort the run. Markers take precedence over ini filters AND
# cover fixture setup (session fixtures are lazily set up inside the
# first test's warning context).
pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


def _skip_if_device_unavailable(combo: BenchmarkCombo, hardware_kind: str) -> None:
    if combo.device == "cuda" and hardware_kind == "cpu":
        pytest.skip(
            f"no GPU available on this machine; cannot benchmark {combo.test_id}",
        )


@pytest.mark.parametrize(
    "combo",
    BENCHMARK_COMBOS,
    ids=lambda c: c.test_id,
)
class TestShortAudioBenchmark:

    def test_should_record_benchmark_when_transcribing_short_audio(
        self,
        combo: BenchmarkCombo,
        perf_context: PerformanceContext,
        short_audio: AudioSegment,
    ) -> None:
        _skip_if_device_unavailable(combo, perf_context.hardware.kind)

        identity = BenchmarkIdentity(
            model=combo.model_name,
            engine=combo.engine_id,
            device=combo.device,  # type: ignore[arg-type]
            audio=_SHORT_AUDIO_FIXTURE_NAME,
        )
        result = run_benchmark(
            combo=combo,
            audio=short_audio,
            n_timed_runs=_TIMED_RUNS_SHORT,
            registry=perf_context.registry,
            models_dir=perf_context.models_dir,
        )
        perf_context.recorder.record(
            identity=identity,
            result=result,
            hardware=perf_context.hardware,
            audio_duration_seconds=short_audio.duration_seconds,
        )
        # Save after every test so a future crash mid-sweep preserves
        # earlier results. Disk write per test is ~16 KB — negligible.
        perf_context.recorder.save()
        assert result.status in ("ok", "error")
        if result.status == "ok":
            assert result.runs_seconds, "ok status requires at least one timed run"


def _last_short_audio_result_for(
    perf_context: PerformanceContext,
    combo: BenchmarkCombo,
) -> str | None:
    """Return the status of the most-recent short-audio entry for this
    combo in the current session, or None if none recorded yet.

    Used by the long-audio test to short-circuit when the short-audio
    test for the same combo already failed: re-attempting the engine
    load can leave the process in an unrecoverable state (Granite has
    aborted the entire sweep this way), so we record a synthetic error
    instead of retrying.
    """
    short_identity = BenchmarkIdentity(
        model=combo.model_name,
        engine=combo.engine_id,
        device=combo.device,  # type: ignore[arg-type]
        audio=_SHORT_AUDIO_FIXTURE_NAME,
    )
    for entry in reversed(perf_context.recorder.entries_in_current_run()):
        if entry.identity == short_identity:
            return entry.result.status
    return None


@pytest.mark.parametrize(
    "combo",
    BENCHMARK_COMBOS,
    ids=lambda c: c.test_id,
)
class TestLongAudioBenchmark:

    def test_should_record_benchmark_when_transcribing_long_audio(
        self,
        combo: BenchmarkCombo,
        perf_context: PerformanceContext,
        long_audio: AudioSegment,
    ) -> None:
        _skip_if_device_unavailable(combo, perf_context.hardware.kind)

        identity = BenchmarkIdentity(
            model=combo.model_name,
            engine=combo.engine_id,
            device=combo.device,  # type: ignore[arg-type]
            audio=_LONG_AUDIO_FIXTURE_NAME,
        )

        short_status = _last_short_audio_result_for(perf_context, combo)
        if short_status == "error":
            from tests.performance._runner import BenchmarkResult

            derived = BenchmarkResult(
                status="error",
                error_type="ShortAudioErroredSkipped",
                error_message=(
                    "Short-audio benchmark for this combo errored - "
                    "skipping long-audio attempt to avoid re-triggering "
                    "the same engine load failure (which has been "
                    "observed to crash the process for some models)."
                ),
            )
            perf_context.recorder.record(
                identity=identity,
                result=derived,
                hardware=perf_context.hardware,
                audio_duration_seconds=long_audio.duration_seconds,
            )
            perf_context.recorder.save()
            return

        if combo.model_name in _FRAGILE_ON_RELOAD and short_status == "ok":
            from tests.performance._runner import BenchmarkResult

            derived = BenchmarkResult(
                status="error",
                error_type="FragileReloadSkipped",
                error_message=(
                    "This model's load path corrupts process state on "
                    "second invocation in the same Python process "
                    "(transformers spawn_materialize segfault on AMD "
                    "ROCm Windows). Short-audio benchmark already "
                    "succeeded - skipping long-audio reload to keep "
                    "the rest of the sweep alive. Run long-audio "
                    "manually with `-k LongStory and granite` to "
                    "benchmark it in a fresh process."
                ),
            )
            perf_context.recorder.record(
                identity=identity,
                result=derived,
                hardware=perf_context.hardware,
                audio_duration_seconds=long_audio.duration_seconds,
            )
            perf_context.recorder.save()
            return

        result = run_benchmark(
            combo=combo,
            audio=long_audio,
            n_timed_runs=_TIMED_RUNS_LONG,
            registry=perf_context.registry,
            models_dir=perf_context.models_dir,
        )
        perf_context.recorder.record(
            identity=identity,
            result=result,
            hardware=perf_context.hardware,
            audio_duration_seconds=long_audio.duration_seconds,
        )
        perf_context.recorder.save()
        assert result.status in ("ok", "error")
