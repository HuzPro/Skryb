"""Decide whether a commit message requests a release, and which version.

The release workflow runs on every push to ``main`` but must publish a
release only when the human who merged explicitly asked for one. The
agreed convention is a head-commit message whose first line opens with
``release:`` followed by a semantic version, e.g.::

    release: v0.2.0

The leading ``v`` is optional, the colon is optional, a pre-release or
build suffix is preserved, and a trailing squash-merge PR number such as
``(#42)`` is ignored. Matching is case-insensitive and considers only
the first line so a version mentioned in the body never triggers a
release.
"""

import re
import sys

from release.errors import NotAReleaseCommitError


_RELEASE_PATTERN = re.compile(
    r"^\s*release\b\s*:?\s*v?"
    r"(?P<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?)",
    re.IGNORECASE,
)


def is_release_request(message: str) -> bool:
    """Return whether ``message`` opens with a valid release marker."""
    return _match_first_line(message) is not None


def release_version(message: str) -> str:
    """Return the version declared by a release commit message.

    Raises :class:`NotAReleaseCommitError` when ``message`` is an
    ordinary commit, so the caller never has to reason about ``None``.
    """
    match = _match_first_line(message)
    if match is None:
        raise NotAReleaseCommitError(
            f"commit message does not request a release: {message!r}"
        )
    return match.group("version")


def _match_first_line(message: str) -> re.Match[str] | None:
    first_line = message.splitlines()[0] if message.splitlines() else ""
    return _RELEASE_PATTERN.match(first_line)


def main(argv: list[str]) -> int:
    """CLI bridge for the workflow: print the version, or exit non-zero.

    ``python -m release.release_request "<message>"`` prints the release
    version to stdout and exits 0 for a release commit; for any other
    commit it prints nothing and exits 1. This lets the workflow gate on
    the exit code without parsing commit messages in shell.
    """
    message = argv[0] if argv else ""
    try:
        print(release_version(message))
    except NotAReleaseCommitError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
