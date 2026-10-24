"""Exception hierarchy for the release tooling.

Mirrors the application's own ``SkrybError`` pattern (one base
class so callers can catch every release-tooling failure with a single
``except``) but stays in this package to keep the build tooling
independent of the application domain.
"""


class ReleaseError(Exception):
    """Base class for every error raised by the release tooling."""


class NotAReleaseCommitError(ReleaseError):
    """The commit message does not request a release."""


class VersionResolutionError(ReleaseError):
    """No release version could be determined from any source."""


class InstallerBuildError(ReleaseError):
    """The installer could not be assembled or compiled."""
