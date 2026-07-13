"""Tests for the hardware acceleration targets.

A single installer ships all three requirements files and pip-installs
one at setup time based on the user's detected hardware. This enum is the
single source of truth for the target -> requirements-file mapping the
installer build feeds into the Inno Setup script.
"""

import pytest

from release.build_target import BuildTarget


class TestBuildTargetIdentity:
    def test_exposes_the_three_supported_targets(self):
        assert {target.slug for target in BuildTarget} == {"cpu", "cuda", "rocm"}

    @pytest.mark.parametrize(
        ("target", "requirements_file"),
        [
            (BuildTarget.CPU, "requirements-cpu.txt"),
            (BuildTarget.CUDA, "requirements-nvidia.txt"),
            (BuildTarget.ROCM, "requirements-amd.txt"),
        ],
    )
    def test_maps_to_its_requirements_file(
        self, target: BuildTarget, requirements_file: str
    ):
        assert target.requirements_file == requirements_file
