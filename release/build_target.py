"""The hardware acceleration targets and their requirements files.

Skryb supports CPU, NVIDIA CUDA, and AMD ROCm PyTorch builds, each
installed from a different wheel index (see the ``requirements-*.txt``
files). The installer ships all three requirements files and pip-installs
exactly one at setup time, chosen from the user's detected hardware.

This enum is the single source of truth for that mapping: the installer
build (:mod:`release.installer`) derives the Inno Setup defines from it so
the ``.iss`` script never hard-codes a requirements filename.
"""

from enum import Enum


class BuildTarget(Enum):
    """A hardware flavour the user can install dependencies for.

    The value pair is ``(slug, requirements_file)``.
    """

    CPU = ("cpu", "requirements-cpu.txt")
    CUDA = ("cuda", "requirements-nvidia.txt")
    ROCM = ("rocm", "requirements-amd.txt")

    @property
    def slug(self) -> str:
        return self.value[0]

    @property
    def requirements_file(self) -> str:
        return self.value[1]
