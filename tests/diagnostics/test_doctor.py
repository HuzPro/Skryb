"""Tests for the ``voicetotext --doctor`` diagnostic probe.

The doctor exists so a user whose transcription silently fails can run
one command and get back a multi-line report listing which engine
backends are installed, what PyTorch sees, and where the HuggingFace
cache lives. The whole point is that it must never crash - every probe
is wrapped so a missing or broken module degrades to an
"ERROR" / "MISSING" line, not a stack trace.
"""

import sys
from unittest.mock import patch

from voicetotext.diagnostics.doctor import run_doctor


class TestDoctorReportShape:
    def test_should_return_a_multi_line_string(self) -> None:
        report = run_doctor()

        assert isinstance(report, str)
        assert "\n" in report

    def test_should_include_python_section(self) -> None:
        report = run_doctor()

        assert "[Python]" in report
        assert sys.version.split()[0] in report

    def test_should_include_engine_backends_section(self) -> None:
        report = run_doctor()

        assert "[Engine backends]" in report

    def test_should_include_torch_section(self) -> None:
        report = run_doctor()

        assert "[torch]" in report.lower() or "[Torch]" in report

    def test_should_include_huggingface_cache_section(self) -> None:
        report = run_doctor()

        assert "HuggingFace" in report or "HF_HUB_CACHE" in report


class TestDoctorReportsMissingBackends:
    """A missing optional backend is the friend's symptom; the doctor
    must call it out specifically (with the pip-install string) instead
    of crashing or just omitting the line.
    """

    def test_should_report_missing_nano_parakeet(self) -> None:
        with patch.dict("sys.modules", {"nano_parakeet": None}):
            report = run_doctor()

        assert "nano_parakeet" in report or "nano-parakeet" in report
        assert "MISSING" in report

    def test_missing_backend_line_should_include_install_hint(self) -> None:
        with patch.dict("sys.modules", {"nano_parakeet": None}):
            report = run_doctor()

        assert "pip install" in report


class TestDoctorReportsInstalledVersions:
    def test_should_report_torch_cuda_availability(self) -> None:
        report = run_doctor()

        # Either True or False string must be present; the doctor's
        # job is to surface this regardless of host hardware.
        assert "cuda available" in report.lower()

    def test_should_not_crash_when_torch_import_fails(self) -> None:
        # If torch itself blows up on import, the doctor must still
        # produce a usable report. The friend's machine could be in
        # this exact state.
        with patch.dict("sys.modules", {"torch": None}):
            report = run_doctor()

        assert "[torch]" in report.lower() or "[Torch]" in report
        # No stack trace leaked
        assert "Traceback" not in report
