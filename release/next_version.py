"""Work out the next release version from the commits since the last one.

Skryb uses semantic versioning, MAJOR.MINOR.PATCH, read for an app:

- PATCH (0.1.0 -> 0.1.1): only fixes and upkeep since the last release
  (``fix:``, ``perf:``, ``docs:``, ``test:``, ``chore:``...).
- MINOR (0.1.1 -> 0.2.0): at least one ``feat:``, something a user can see
  or turn on.
- MAJOR (1.4.2 -> 2.0.0): a breaking change, marked ``feat!:`` / ``fix!:``
  or with a ``BREAKING CHANGE:`` line: settings or data that don't carry
  over, a dropped GPU or Windows version. Before 1.0.0 a breaking change
  bumps MINOR instead; 1.0.0 is cut by hand once Skryb is stable.

``python -m release.next_version`` prints the suggestion and why;
``--apply`` also writes it into ``pyproject.toml`` for the release commit.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from release.version import read_pyproject_version

_TYPE = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]*\))?(?P<breaking>!)?:", re.IGNORECASE)
_PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"[^"]*"', re.MULTILINE)


def bump_for(messages: list[str]) -> str:
    """"major", "minor" or "patch" for these commit messages (subject plus body)."""
    bump = "patch"
    for message in messages:
        subject = message.splitlines()[0] if message.strip() else ""
        match = _TYPE.match(subject.strip())
        if (match and match["breaking"]) or re.search(r"^BREAKING[ -]CHANGE:", message, re.MULTILINE):
            return "major"
        if match and match["type"].lower() == "feat":
            bump = "minor"
    return bump


def next_version(current: str, bump: str) -> str:
    major, minor, patch = (int(part) for part in current.split("-")[0].split("+")[0].split("."))
    if bump == "major" and major == 0:
        bump = "minor"
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def with_version(pyproject_text: str, version: str) -> str:
    if len(_PYPROJECT_VERSION.findall(pyproject_text)) != 1:
        raise ValueError("pyproject.toml must declare exactly one version line")
    return _PYPROJECT_VERSION.sub(f'version = "{version}"', pyproject_text)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout


def _messages_since(tag: str) -> list[str]:
    log = _git("log", "--format=%B%x00", f"{tag}..HEAD")
    return [message.strip() for message in log.split("\x00") if message.strip()]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Suggest the next Skryb version.")
    parser.add_argument("--apply", action="store_true", help="write it into pyproject.toml")
    args = parser.parse_args(argv)
    pyproject = Path("pyproject.toml")
    current = read_pyproject_version(pyproject)
    tag = f"v{current}"
    try:
        messages = _messages_since(tag)
    except subprocess.CalledProcessError:
        print(f"No tag {tag} here. Run `git fetch --tags` first.", file=sys.stderr)
        return 1
    if not messages:
        print(f"Nothing has changed since {tag}.", file=sys.stderr)
        return 1
    bump = bump_for(messages)
    version = next_version(current, bump)
    commits = f"{len(messages)} commit{'' if len(messages) == 1 else 's'}"
    print(f"{version} ({bump}: {commits} since {tag})")
    if args.apply:
        pyproject.write_text(with_version(pyproject.read_text(encoding="utf-8"), version), encoding="utf-8")
        print(f'pyproject.toml now says {version}. Commit it as "release: v{version}".')
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
