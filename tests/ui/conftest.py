from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.ui.rig import Rig


@pytest.fixture()
def make_rig(tmp_path: Path) -> Iterator[Callable[..., Rig]]:
    built: list[Rig] = []

    def make(**options: Any) -> Rig:
        rig = Rig(tmp_path / f"rig{len(built)}", **options)
        built.append(rig)
        return rig

    yield make
    for rig in built:
        rig.close()


@pytest.fixture()
def rig(make_rig: Callable[..., Rig]) -> Rig:
    return make_rig()
