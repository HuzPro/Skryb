"""The engine-floor validator must not mistake a skip for a result.

A skipped engine test is the exact failure the validator exists to
catch, so its reading of pytest's report is pinned against a synthetic
one rather than trusted to a run nobody can reproduce on CI.
"""

from pathlib import Path

from scripts.validate_engine_floor import (
    _ENGINE_TEST_MODULES,
    _engines_that_executed,
    _verdict,
)

_MODULE = "tests.integration"
_REPORT = f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="{_MODULE}.test_crisperwhisper_transcribes_audio_file"
          name="test_ran"/>
</testsuite></testsuites>
"""


def test_should_count_only_engines_whose_tests_really_ran(
    tmp_path: Path,
) -> None:
    report = tmp_path / "engine-floor.xml"
    report.write_text(_REPORT, encoding="utf-8")

    assert _engines_that_executed(report) == {"crisperwhisper"}


def test_should_fail_when_an_engine_produced_no_result() -> None:
    assert _verdict(set(), 0) == 1


def test_should_fail_when_pytest_itself_failed() -> None:
    assert _verdict(set(_ENGINE_TEST_MODULES), 1) == 1


def test_should_pass_when_every_engine_ran_green() -> None:
    assert _verdict(set(_ENGINE_TEST_MODULES), 0) == 0
