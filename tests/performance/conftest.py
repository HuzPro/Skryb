"""Performance-benchmark fixtures, hooks, and gating.

This conftest sets ``HF_HUB_CACHE`` to the app's models directory at
module-load time - BEFORE any HuggingFace import runs inside the
project tree - so that ``snapshot_download`` (in ``_pre_download``) and
each adapter's ``from_pretrained(..., local_files_only=True)`` both
agree on where weights live.

Gating is two-layered:

* Env var ``VOICETOTEXT_RUN_PERFORMANCE_TESTS`` must be set to a
  non-empty value, otherwise every test in this directory is skipped
  with a clear reason during collection.
* The ``@pytest.mark.performance`` + ``@pytest.mark.slow`` markers on
  the tests themselves mean ``pytest`` default ``addopts`` (which
  excludes ``slow``) keeps them out of normal runs.

Run with::

    set VOICETOTEXT_RUN_PERFORMANCE_TESTS=1
    venv\\Scripts\\pytest -m performance -s tests/performance/
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

# ──────────────────────────────────────────────────────────────────────
# Env setup BEFORE any HF import.
# ``AppPaths`` reads ``%APPDATA%`` and computes the same models dir
# the production app uses, so the perf suite and the GUI share weights.
# ──────────────────────────────────────────────────────────────────────
from voicetotext.core.paths import AppPaths

_paths = AppPaths()
os.environ.setdefault("HF_HUB_CACHE", str(_paths.models_dir))
# Without dev-mode/admin, Windows can't make HF's expected symlinks and
# the Hub emits a noisy UserWarning. Suppress at source so it never
# turns into an error under pyproject's ``filterwarnings = ["error"]``.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# The benchmark cycles several engines through the same process, any of
# which may pull a second OpenMP runtime alongside PyTorch's. Intel's
# runtime aborts on duplicate OpenMP init by default; the documented
# workaround for testing scenarios is ``KMP_DUPLICATE_LIB_OK``.
# Production never hits this because it picks one engine and sticks
# with it for the process lifetime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Late imports - these go through huggingface_hub / torch under the hood
# and must see HF_HUB_CACHE.
from voicetotext.transcription.engine_registry import EngineRegistry  # noqa: E402
from voicetotext.transcription.engine_types import HardwareProfile  # noqa: E402
from voicetotext.transcription.license_policy import LicensePolicy  # noqa: E402

from tests.helpers.license_gate import user_license_policy  # noqa: E402
from tests.performance._recorder import PerformanceRecorder  # noqa: E402


_GATE_ENV_VAR = "VOICETOTEXT_RUN_PERFORMANCE_TESTS"
_RESULTS_DIR = Path(__file__).parent / ".results"
_RESULTS_PATH = _RESULTS_DIR / "benchmark_results.json"


@dataclass(frozen=True, slots=True)
class PerformanceContext:
    models_dir: Path
    hardware: HardwareProfile
    registry: EngineRegistry
    recorder: PerformanceRecorder
    license_policy: LicensePolicy


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get(_GATE_ENV_VAR):
        return
    skip_marker = pytest.mark.skip(
        reason=f"set {_GATE_ENV_VAR}=1 to run performance benchmarks",
    )
    perf_root = str(Path(__file__).parent)
    for item in items:
        if str(item.fspath).startswith(perf_root):
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def perf_context(request: pytest.FixtureRequest) -> Iterator[PerformanceContext]:
    if hasattr(request.config, "workerinput"):
        raise RuntimeError(
            "performance benchmarks cannot run under pytest-xdist - "
            "VRAM measurements and engine state would corrupt across workers"
        )

    from voicetotext.transcription.engine_factory import build_engine_registry
    from voicetotext.transcription.torch_hardware_detector import TorchHardwareDetector

    from tests.performance._combos import BENCHMARK_COMBOS
    from tests.performance._pre_download import ensure_models_downloaded

    print("\n=== Performance benchmark setup ===", flush=True)
    hardware = TorchHardwareDetector().detect()
    print(
        f"Hardware: kind={hardware.kind} "
        f"device={hardware.device_name} "
        f"gfx_arch={hardware.gfx_arch}",
        flush=True,
    )
    print(f"Models dir: {_paths.models_dir}", flush=True)
    print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}", flush=True)

    # The sweep downloads and runs the very weights the app gates, so it
    # reads the same acceptance record the app writes. A gated model the
    # user has not accepted is never fetched and never benchmarked.
    license_policy = user_license_policy()

    relevant_combos = _filter_combos_for_hardware(BENCHMARK_COMBOS, hardware.kind)
    print(
        f"Pre-downloading models for {len(relevant_combos)} (engine, repo) pairs...",
        flush=True,
    )
    ensure_models_downloaded(relevant_combos, _paths.models_dir, license_policy)
    print("Pre-download complete.", flush=True)

    recorder = PerformanceRecorder(_RESULTS_PATH)
    recorder.load()
    print(
        f"Loaded {len(recorder.all_entries())} historic entries from "
        f"{_RESULTS_PATH}",
        flush=True,
    )

    registry = build_engine_registry()
    ctx = PerformanceContext(
        models_dir=_paths.models_dir,
        hardware=hardware,
        registry=registry,
        recorder=recorder,
        license_policy=license_policy,
    )
    print("=== Setup complete; starting benchmark sweep ===", flush=True)
    yield ctx
    # Fixture teardown fires BEFORE ``pytest_unconfigure`` in the
    # parent conftest (which calls ``os._exit``), so the JSON save
    # always lands.
    print("\n=== Saving benchmark results ===", flush=True)
    recorder.save()
    print(f"  -> {_RESULTS_PATH}", flush=True)


def _filter_combos_for_hardware(combos, hardware_kind: str):
    if hardware_kind == "cpu":
        return [c for c in combos if c.device == "cpu"]
    return list(combos)


@pytest.fixture(scope="session")
def short_audio():
    from tests.fixtures import require_short_fixture
    from tests.helpers.audio_file_loader import load_audio_file

    return load_audio_file(require_short_fixture())


@pytest.fixture(scope="session")
def long_audio():
    from tests.fixtures import require_long_fixture
    from tests.helpers.audio_file_loader import load_audio_file

    return load_audio_file(require_long_fixture())


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not os.environ.get(_GATE_ENV_VAR):
        return
    if not _RESULTS_PATH.exists():
        return
    try:
        from tests.performance._summary import format_summary_table

        recorder = PerformanceRecorder(_RESULTS_PATH)
        recorder.load()
        entries = recorder.entries_in_current_run() or recorder.all_entries()
        terminalreporter.write_sep("=", "Benchmark Results")
        terminalreporter.write_line(format_summary_table(entries))
    except Exception as exc:  # diagnostic only
        terminalreporter.write_line(f"summary table render failed: {exc}")
