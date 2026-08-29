"""Generate the bundled THIRD-PARTY-NOTICES document.

The installer pip-installs the app's permissively-licensed third-party
libraries on the user's machine; this module ships inside the install
payload and is run there afterwards (``python -m release.notices
<path>``), reproducing the license texts of the environment that was
actually installed so the distribution discharges those attribution
duties. The licensing audit flagged this as the one legal to-do for
shipping VoiceToText installers, and it keeps that duty discharged for
any downstream fork - including proprietary ones, which the app's
Apache-2.0 license deliberately permits.

Selection, license-id summarisation, and document rendering are pure
functions (unit-tested). Reading the actual license files from the
installed environment is isolated in a thin ``importlib.metadata``
adapter at the bottom of the file.
"""

import importlib.metadata as metadata
import sys
from dataclasses import dataclass
from pathlib import Path

from release import APP_NAME


# Distributions that may be present in the environment the tool runs in
# (the user's install, or a dev venv) without being shipped app
# components: the app itself, the packaging toolchain, and the test
# stack. Names are compared normalised (lower-case, dashes).
_NOT_BUNDLED = frozenset(
    {
        "voicetotext",
        "pip",
        "setuptools",
        "wheel",
        "pyinstaller",
        "pyinstaller-hooks-contrib",
        "altgraph",
        "pefile",
        "pywin32-ctypes",
        "pytest",
        "pytest-mock",
        "pytest-cov",
        "pytest-qt",
        "pluggy",
        "iniconfig",
        "coverage",
    }
)

_LICENSE_FILENAME_TOKENS = ("license", "licence", "copying", "notice")

# A ``License:`` metadata field longer than this (or spanning lines) is
# almost certainly the full license text crammed into the field rather
# than an identifier, so it is not used as the short id.
_MAX_DECLARED_ID_LENGTH = 64

_SECTION_RULE = "=" * 70


@dataclass(frozen=True)
class RawLicenseMetadata:
    """The license-related metadata fields as found on a distribution."""

    expression: str
    declared: str
    classifiers: tuple[str, ...]


@dataclass(frozen=True)
class PackageLicense:
    """A bundled component and the license text that must travel with it."""

    name: str
    version: str
    license_id: str
    license_text: str


def should_include(name: str) -> bool:
    """Return whether ``name`` is a component shipped inside the binary."""
    return _normalise(name) not in _NOT_BUNDLED


def is_license_file(filename: str) -> bool:
    """Return whether ``filename`` looks like a license or notice file."""
    base = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(token in base for token in _LICENSE_FILENAME_TOKENS)


def summarize_license(raw: RawLicenseMetadata) -> str:
    """Return the best short license identifier from raw metadata."""
    if raw.expression.strip():
        return raw.expression.strip()
    from_classifiers = _license_from_classifiers(raw.classifiers)
    if from_classifiers:
        return from_classifiers
    declared = raw.declared.strip()
    if declared and "\n" not in declared and len(declared) <= _MAX_DECLARED_ID_LENGTH:
        return declared
    return "Unknown"


def render_notice(package: PackageLicense) -> str:
    """Render one component's heading and license text."""
    return "\n".join(
        [
            _SECTION_RULE,
            f"{package.name} {package.version}",
            f"License: {package.license_id}",
            _SECTION_RULE,
            "",
            package.license_text.strip(),
            "",
        ]
    )


def render_document(packages: list[PackageLicense]) -> str:
    """Render the full THIRD-PARTY-NOTICES document, components sorted."""
    ordered = sorted(packages, key=lambda package: package.name.lower())
    sections = [_render_header(ordered)]
    sections += [render_notice(package) for package in ordered]
    return "\n".join(sections)


def _render_header(packages: list[PackageLicense]) -> str:
    summary = [f"  - {package.name} {package.version} ({package.license_id})" for package in packages]
    return "\n".join(
        [
            f"{APP_NAME}: Third-Party Software Notices",
            _SECTION_RULE,
            "",
            f"This distribution of {APP_NAME} bundles the third-party",
            "open-source components listed below. Each component is the",
            "property of its respective authors and is provided under the",
            "license reproduced with it.",
            "",
            "Machine-learning model weights (Parakeet, Nemotron, ARK-ASR,",
            "Qwen3-ASR, CrisperWhisper) are NOT included in this distribution;",
            "they are downloaded from their original sources at runtime, so",
            "their model licenses are not reproduced here.",
            "",
            f"Bundled components ({len(packages)}):",
            *summary,
            "",
            "",
        ]
    )


def _license_from_classifiers(classifiers: tuple[str, ...]) -> str:
    names: list[str] = []
    for classifier in classifiers:
        if not classifier.startswith("License") or "::" not in classifier:
            continue
        tail = classifier.split("::")[-1].strip()
        if tail and tail not in ("OSI Approved", "License"):
            names.append(tail)
    return " / ".join(dict.fromkeys(names))


def _normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-")


# --- importlib.metadata adapter (thin I/O boundary) ----------------------


def render_installed_notices() -> str:
    """Render the notices document for the current environment."""
    return render_document(installed_package_licenses())


def installed_package_licenses() -> list[PackageLicense]:
    """Collect license info for every bundled distribution installed."""
    collected: dict[str, PackageLicense] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"] or ""
        if not name or not should_include(name):
            continue
        key = _normalise(name)
        if key not in collected:
            collected[key] = _to_package_license(distribution)
    return list(collected.values())


def _to_package_license(distribution: "metadata.Distribution") -> PackageLicense:
    name = distribution.metadata["Name"] or ""
    raw = RawLicenseMetadata(
        expression=distribution.metadata.get("License-Expression", "") or "",
        declared=distribution.metadata.get("License", "") or "",
        classifiers=tuple(distribution.metadata.get_all("Classifier") or ()),
    )
    return PackageLicense(
        name=name,
        version=distribution.version or "",
        license_id=summarize_license(raw),
        license_text=_read_license_text(distribution, name),
    )


def _read_license_text(distribution: "metadata.Distribution", name: str) -> str:
    texts: list[str] = []
    for package_path in distribution.files or []:
        if not is_license_file(package_path.name):
            continue
        content = _safe_read(distribution, package_path)
        if content:
            texts.append(f"--- {package_path.name} ---\n{content.strip()}")
    if texts:
        return "\n\n".join(texts)
    return (
        f"(No license file was found in the {name} distribution. "
        "Refer to the project's homepage for its license terms.)"
    )


def _safe_read(
    distribution: "metadata.Distribution", package_path: "metadata.PackagePath"
) -> str:
    # ``Distribution.read_text`` only resolves files relative to the
    # ``.dist-info`` dir, so it cannot read a ``licenses/LICENSE`` entry
    # from ``distribution.files``. Locate the real file and read it.
    try:
        location = distribution.locate_file(package_path)
        return Path(location).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return ""


def main(argv: list[str]) -> int:
    """Write the notices to a file argument, or stdout when omitted."""
    document = render_installed_notices()
    if argv:
        Path(argv[0]).write_text(document, encoding="utf-8")
    else:
        print(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
