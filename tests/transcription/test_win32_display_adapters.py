import sys

import pytest

from voicetotext.transcription import win32_display_adapters
from voicetotext.transcription.win32_display_adapters import (
    display_adapter_fingerprint,
)


def test_each_adapter_is_named_once_in_a_stable_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        win32_display_adapters,
        "_display_adapter_names",
        lambda: [
            "NVIDIA GeForce RTX 4090",
            "AMD Radeon(TM) Graphics",
            "NVIDIA GeForce RTX 4090",
            "Écran Radeon™ intégré",
            "NVIDIA GeForce RTX 4090",
        ],
    )

    assert display_adapter_fingerprint() == (
        "AMD Radeon(TM) Graphics|NVIDIA GeForce RTX 4090|Écran Radeon™ intégré"
    )


def test_adapters_that_cannot_be_read_give_an_empty_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable() -> list[str]:
        raise OSError("EnumDisplayDevicesW unavailable")

    monkeypatch.setattr(win32_display_adapters, "_display_adapter_names", unreadable)

    assert display_adapter_fingerprint() == ""


@pytest.mark.skipif(sys.platform != "win32", reason="reads Win32 display adapters")
def test_this_machine_reports_the_same_fingerprint_every_time() -> None:
    fingerprint = display_adapter_fingerprint()

    assert fingerprint
    assert all(display_adapter_fingerprint() == fingerprint for _ in range(50))
