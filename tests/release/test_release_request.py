"""Tests for the commit-message release gate.

A push to ``main`` should publish a release only when the head commit
message opens with a ``release:`` marker carrying a version. Ordinary
commits and merges must be ignored. These behaviours decide whether the
GitHub Actions workflow builds and tags a release at all, so they are
specified here rather than left to an untested shell regex.
"""

import pytest

from release.errors import NotAReleaseCommitError
from release.release_request import is_release_request, release_version


class TestRecognisingAReleaseCommit:
    @pytest.mark.parametrize(
        "message",
        [
            "release: v1.2.3",
            "release: 1.2.3",
            "release v1.2.3",
            "release:v1.2.3",
            "Release: V1.2.3",
            "  release: v1.2.3  ",
            "release: v1.2.3 (#42)",
            "release: v1.2.3\n\nLong body explaining the release.",
        ],
    )
    def test_recognises_a_release_marker(self, message: str):
        assert is_release_request(message)

    @pytest.mark.parametrize(
        "message",
        [
            "fix: correct off-by-one in chunk planner",
            "Add release notes to the README",
            "released v1.2.3",
            "release the streaming engine",
            "release: v1.2",
            "",
            "   ",
        ],
    )
    def test_ignores_non_release_commits(self, message: str):
        assert not is_release_request(message)


class TestExtractingTheReleaseVersion:
    def test_strips_the_leading_v(self):
        assert release_version("release: v1.2.3") == "1.2.3"

    def test_accepts_a_bare_version_without_v(self):
        assert release_version("release: 1.2.3") == "1.2.3"

    def test_keeps_a_prerelease_suffix(self):
        assert release_version("release: v1.2.3-rc.1") == "1.2.3-rc.1"

    def test_ignores_a_trailing_squash_merge_pr_number(self):
        assert release_version("release: v0.2.0 (#42)") == "0.2.0"

    def test_reads_the_version_from_the_first_line_only(self):
        message = "release: v2.0.0\n\nrelease: v9.9.9 mentioned in the body"

        assert release_version(message) == "2.0.0"

    def test_raises_when_the_commit_is_not_a_release(self):
        with pytest.raises(NotAReleaseCommitError):
            release_version("fix: nothing to release here")
