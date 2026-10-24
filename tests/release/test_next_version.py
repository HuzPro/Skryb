import pytest

from release.next_version import bump_for, next_version, with_version

_PYPROJECT = '''[project]
name = "skryb"
version = "0.1.0"
description = "Local dictation"
dependencies = ["requests>=2.31.0"]

[tool.ruff]
target-version = "py312"
'''


class TestChoosingTheBump:
    def test_only_fixes_and_upkeep_are_a_patch(self):
        messages = [
            "fix: stop reporting pastes as lost (#43)",
            "perf: load the model while the tray is built",
            "docs: clean up the README",
            "chore: sort the renamed package in Cargo.lock",
            "test: stage the licence tests",
            "merge: bring in the other four tracks",
            "release: v0.1.0",
        ]

        assert bump_for(messages) == "patch"

    @pytest.mark.parametrize("feature", [
        "feat: check GitHub for updates",
        "feat(ui): an Updates page",
        "Feat: capitalised by hand",
        "feat: type spoken Urdu as Roman Urdu instead of Hindi script",
    ])
    def test_a_single_feature_among_thousands_of_fixes_is_a_minor(self, feature):
        messages = [f"fix: case {n}" for n in range(5000)]
        messages.insert(2500, feature)

        assert bump_for(messages) == "minor"

    @pytest.mark.parametrize("breaking", [
        "feat!: move settings to a new format",
        "fix(installer)!: install per user only",
        "feat: new settings format\n\nBREAKING CHANGE: old settings are not read",
        "refactor: drop CUDA 11\n\nBREAKING-CHANGE: needs a newer NVIDIA driver",
    ])
    def test_a_breaking_change_is_a_major(self, breaking):
        assert bump_for(["fix: one", breaking, "feat: two"]) == "major"

    @pytest.mark.parametrize("message", ["", "   ", "\n\n", "featuring a typo", "feature: not the feat type", "WIP", "Ünïcödé commit"])
    def test_messages_without_a_known_type_count_as_patches(self, message):
        assert bump_for([message]) == "patch"

    def test_mentioning_breaking_change_mid_sentence_is_not_breaking(self):
        assert bump_for(["fix: avoid a BREAKING CHANGE: later"]) == "patch"

    def test_no_commits_is_a_patch(self):
        assert bump_for([]) == "patch"


class TestTheNextNumber:
    @pytest.mark.parametrize(("current", "bump", "expected"), [
        ("0.1.0", "patch", "0.1.1"),
        ("0.1.9", "patch", "0.1.10"),
        ("0.1.1", "minor", "0.2.0"),
        ("0.9.4", "minor", "0.10.0"),
        ("1.4.2", "major", "2.0.0"),
        ("1.4.2", "minor", "1.5.0"),
        ("0.4.2", "major", "0.5.0"),
        ("0.2.0-rc1", "patch", "0.2.1"),
    ])
    def test_bumps_one_number_and_zeroes_the_ones_after_it(self, current, bump, expected):
        assert next_version(current, bump) == expected


class TestWritingPyproject:
    def test_changes_only_the_project_version_line(self):
        updated = with_version(_PYPROJECT, "0.2.0")

        assert updated == _PYPROJECT.replace('version = "0.1.0"', 'version = "0.2.0"')
        assert 'target-version = "py312"' in updated

    def test_refuses_a_file_with_two_version_lines(self):
        with pytest.raises(ValueError):
            with_version(_PYPROJECT + '\n[tool.other]\nversion = "9"\n', "0.2.0")

    def test_refuses_a_file_with_no_version_line(self):
        with pytest.raises(ValueError):
            with_version('[project]\nname = "skryb"\n', "0.2.0")
