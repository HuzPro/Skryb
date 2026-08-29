"""Prove the declared ``transformers`` floor is exercised, not just declared.

Every real-engine integration test skips itself when a GPU, an audio
fixture, or a downloaded model is missing, so a green pytest exit says
nothing about whether a single engine ever ran. This driver pins the
exact floor, runs the three engine tests, and fails unless every engine
leaves behind at least one result that actually executed.

    venv\\Scripts\\python.exe scripts/validate_engine_floor.py

Preconditions: a working GPU, the pinned snapshots downloaded through the
app, and the CrisperWhisper licence accepted in the app. Not for CI - it
needs real hardware and tens of gigabytes of weights.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree


_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPECTED_TRANSFORMERS = "5.16.0"
_OPT_IN = "VOICETOTEXT_RUN_REAL_MODEL_TESTS"
_ENGINE_TEST_MODULES: dict[str, str] = {
    "ark_asr": "tests.integration.test_ark_asr_transcribes_audio_file",
    "qwen3_asr": "tests.integration.test_qwen3_asr_transcribes_audio_file",
    "crisperwhisper": (
        "tests.integration.test_crisperwhisper_transcribes_audio_file"
    ),
}
# A testcase carrying any of these did not run the engine.
_NON_RESULT_TAGS = ("skipped", "failure", "error")


def main() -> int:
    _assert_exact_transformers_floor()
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "engine-floor.xml"
        completed = _run_engine_tests(report)
        if not report.exists():
            print("pytest wrote no report; nothing was validated")
            return 1
        executed = _engines_that_executed(report)
    return _verdict(executed, completed.returncode)


def _assert_exact_transformers_floor() -> None:
    import transformers

    if transformers.__version__ != _EXPECTED_TRANSFORMERS:
        raise SystemExit(
            f"transformers {transformers.__version__} is installed, so this "
            f"run would not exercise the declared floor. Install exactly "
            f"{_EXPECTED_TRANSFORMERS} and try again."
        )


def _run_engine_tests(report: Path) -> "subprocess.CompletedProcess[bytes]":
    command = [
        sys.executable,
        "-m",
        "pytest",
        *(_test_path(module) for module in _ENGINE_TEST_MODULES.values()),
        # The engine tests are marked slow, which the project's addopts
        # deselects by default.
        "-m",
        "slow",
        f"--junit-xml={report}",
    ]
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(
        command, cwd=str(_REPO_ROOT), env=os.environ | {_OPT_IN: "1"},
    )


def _test_path(module: str) -> str:
    return f"{module.replace('.', '/')}.py"


def _engines_that_executed(report: Path) -> set[str]:
    executed: set[str] = set()
    for case in ElementTree.parse(report).getroot().iter("testcase"):
        if any(case.find(tag) is not None for tag in _NON_RESULT_TAGS):
            continue
        engine = _engine_of(case.get("classname", ""))
        if engine:
            executed.add(engine)
    return executed


def _engine_of(classname: str) -> str:
    for engine, module in _ENGINE_TEST_MODULES.items():
        if classname == module or classname.startswith(f"{module}."):
            return engine
    return ""


def _verdict(executed: set[str], pytest_returncode: int) -> int:
    missing = sorted(set(_ENGINE_TEST_MODULES) - executed)
    for engine in missing:
        print(
            f"FAIL {engine}: nothing ran - every test skipped, failed, or "
            f"was never collected"
        )
    if missing or pytest_returncode != 0:
        return 1
    print(
        f"OK: transformers {_EXPECTED_TRANSFORMERS} exercised by "
        f"{', '.join(sorted(executed))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
