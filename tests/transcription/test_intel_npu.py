"""Finding a tuned Intel NPU among the PCI devices Windows has enumerated."""

import sys

import pytest

from skryb.transcription import intel_npu


@pytest.fixture(autouse=True)
def _forget_the_last_answer():
    intel_npu.tuned_intel_npu.cache_clear()
    yield
    intel_npu.tuned_intel_npu.cache_clear()


@pytest.fixture
def pci(monkeypatch):
    def install(device_ids: list[str] | Exception) -> None:
        def listed() -> list[str]:
            if isinstance(device_ids, Exception):
                raise device_ids
            return device_ids

        monkeypatch.setattr(intel_npu.sys, "platform", "win32")
        monkeypatch.setattr(intel_npu, "_intel_pci_device_ids", listed)

    return install


def test_a_core_series_3_npu_is_found_among_the_other_intel_devices(pci) -> None:
    pci(["7D67", "FD3E", "A0EF", "7E7F"])

    assert intel_npu.tuned_intel_npu() == "FD3E"


def test_an_npu_the_engine_was_not_measured_on_does_not_count(pci) -> None:
    pci(["643E", "7D1D", "B03E"])  # Lunar Lake, Meteor Lake, Panther Lake

    assert intel_npu.tuned_intel_npu() is None


def test_an_unreadable_registry_means_no_npu(pci) -> None:
    pci(PermissionError("access denied"))

    assert intel_npu.tuned_intel_npu() is None


def test_off_windows_there_is_never_an_npu(monkeypatch) -> None:
    monkeypatch.setattr(intel_npu.sys, "platform", "linux")

    assert intel_npu.tuned_intel_npu() is None


@pytest.mark.skipif(sys.platform != "win32", reason="reads the Windows registry")
def test_the_registry_lists_device_ids_as_four_upper_case_hex_digits() -> None:
    ids = intel_npu._intel_pci_device_ids()

    assert all(len(device) == 4 and device == device.upper() for device in ids)
