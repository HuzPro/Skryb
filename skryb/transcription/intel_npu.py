"""Which Intel NPU this computer has, if it is one the NPU engine was tuned and measured on.

Read from the PCI devices Windows has enumerated, in the registry: no driver call and no OpenVINO
import, so it is cheap enough for every engine ranking.
"""

import sys
from functools import cache

# PCI device IDs of the NPUs the OpenVINO engine has been measured on, where it beats Fermion's
# CPU engine. Core Series 3 (Wildcat Lake), NPU 5, after a pause as between dictations: Phonon
# types 3 s in 63 ms against 421 and 10 s in 190 against 878, with a thirtieth of the CPU time.
TUNED_NPU_DEVICE_IDS = frozenset({"FD3E"})
_PCI_KEY = r"SYSTEM\CurrentControlSet\Enum\PCI"
_INTEL_VENDOR = "VEN_8086&DEV_"


@cache
def tuned_intel_npu() -> str | None:
    """The PCI device ID of a tuned Intel NPU in this computer, such as ``"FD3E"``, else None."""
    if sys.platform != "win32":
        return None
    try:
        return next((device for device in _intel_pci_device_ids() if device in TUNED_NPU_DEVICE_IDS), None)
    except OSError:
        return None


def _intel_pci_device_ids() -> list[str]:
    import winreg

    ids = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _PCI_KEY) as pci:
        index = 0
        while True:
            try:
                name = winreg.EnumKey(pci, index)
            except OSError:
                return ids
            if name.upper().startswith(_INTEL_VENDOR):
                ids.append(name[len(_INTEL_VENDOR):len(_INTEL_VENDOR) + 4].upper())
            index += 1
