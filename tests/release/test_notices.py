"""Tests for THIRD-PARTY-NOTICES generation.

The shipped binary bundles permissively-licensed third-party libraries
(torch, requests, Pillow, ...), so the distribution must reproduce
their license texts. This is the one legal to-do from the licensing
audit. Build and test tooling that never ends up inside the binary
(PyInstaller, pytest, the app itself) is excluded; model weights are
downloaded at runtime and are out of scope for this file.

Only the pure selection/summary/rendering logic is covered here. The
adapter that reads license files from the live environment via
``importlib.metadata`` is a thin I/O boundary, exercised manually.
"""

import pytest

from release.notices import (
    PackageLicense,
    RawLicenseMetadata,
    is_license_file,
    render_document,
    render_notice,
    should_include,
    summarize_license,
)


class TestDecidingWhichPackagesToInclude:
    def test_includes_a_bundled_runtime_dependency(self):
        assert should_include("torch")
        assert should_include("requests")
        assert should_include("Pillow")

    def test_excludes_the_application_itself(self):
        assert not should_include("voicetotext")

    def test_excludes_build_and_test_tooling(self):
        assert not should_include("pyinstaller")
        assert not should_include("pytest")
        assert not should_include("setuptools")

    def test_matches_excluded_names_irrespective_of_case_or_separator(self):
        assert not should_include("PyInstaller")
        assert not should_include("pytest_cov")
        assert not should_include("pytest-cov")


class TestRecognisingLicenseFiles:
    def test_recognises_common_license_filenames(self):
        assert is_license_file("LICENSE")
        assert is_license_file("LICENSE.txt")
        assert is_license_file("licence.md")
        assert is_license_file("COPYING")
        assert is_license_file("NOTICE")
        assert is_license_file("LICENSE.APACHE")

    def test_ignores_non_license_files(self):
        assert not is_license_file("__init__.py")
        assert not is_license_file("RECORD")
        assert not is_license_file("metadata.json")


class TestSummarisingTheLicenseIdentifier:
    def test_prefers_an_explicit_spdx_expression(self):
        raw = RawLicenseMetadata(
            expression="Apache-2.0",
            declared="whatever",
            classifiers=("License :: OSI Approved :: MIT License",),
        )

        assert summarize_license(raw) == "Apache-2.0"

    def test_falls_back_to_the_osi_classifier(self):
        raw = RawLicenseMetadata(
            expression="",
            declared="",
            classifiers=("License :: OSI Approved :: MIT License",),
        )

        assert summarize_license(raw) == "MIT License"

    def test_uses_a_short_declared_field_when_no_classifier(self):
        raw = RawLicenseMetadata(
            expression="", declared="BSD-3-Clause", classifiers=()
        )

        assert summarize_license(raw) == "BSD-3-Clause"

    def test_ignores_a_declared_field_that_is_full_license_text(self):
        raw = RawLicenseMetadata(
            expression="",
            declared="Permission is hereby granted, free of charge, "
            "to any person\nobtaining a copy of this software...",
            classifiers=(),
        )

        assert summarize_license(raw) == "Unknown"

    def test_reports_unknown_when_nothing_is_declared(self):
        raw = RawLicenseMetadata(expression="", declared="", classifiers=())

        assert summarize_license(raw) == "Unknown"


class TestRenderingOneNotice:
    def test_includes_the_package_identity_and_license_text(self):
        pkg = PackageLicense(
            name="requests",
            version="2.34.2",
            license_id="Apache-2.0",
            license_text="Apache License, Version 2.0 ...",
        )

        notice = render_notice(pkg)

        assert "requests" in notice
        assert "2.34.2" in notice
        assert "Apache-2.0" in notice
        assert "Apache License, Version 2.0 ..." in notice


class TestRenderingTheWholeDocument:
    def _package(self, name: str) -> PackageLicense:
        return PackageLicense(
            name=name,
            version="1.0.0",
            license_id="MIT",
            license_text=f"{name} license body",
        )

    def test_lists_components_sorted_case_insensitively(self):
        document = render_document(
            [self._package("torch"), self._package("Pillow")]
        )

        assert document.index("Pillow") < document.index("torch")

    def test_states_the_product_and_the_runtime_model_disclaimer(self):
        document = render_document([self._package("torch")])

        assert "VoiceToText" in document
        assert "model" in document.lower()

    def test_reproduces_every_component_license(self):
        document = render_document(
            [self._package("torch"), self._package("requests")]
        )

        assert "torch license body" in document
        assert "requests license body" in document
