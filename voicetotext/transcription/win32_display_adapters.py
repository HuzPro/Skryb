import ctypes


class _DisplayDevice(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("DeviceName", ctypes.c_wchar * 32),
        ("DeviceString", ctypes.c_wchar * 128),
        ("StateFlags", ctypes.c_uint32),
        ("DeviceID", ctypes.c_wchar * 128),
        ("DeviceKey", ctypes.c_wchar * 128),
    ]


def display_adapter_fingerprint() -> str:
    """The display adapters' names, each once, sorted and joined with ``|``.

    It changes when a GPU is added, removed or swapped, which is when a
    remembered GPU probe stops being true. Empty when the adapters cannot
    be read, including off Windows.
    """
    try:
        names = _display_adapter_names()
    except Exception:
        return ""
    return "|".join(sorted(set(names)))


def _display_adapter_names() -> list[str]:
    enum_display_devices = ctypes.windll.user32.EnumDisplayDevicesW
    device = _DisplayDevice()
    device.cb = ctypes.sizeof(_DisplayDevice)
    names: list[str] = []
    index = 0
    while enum_display_devices(None, index, ctypes.byref(device), 0):
        if device.DeviceString:
            names.append(device.DeviceString)
        index += 1
    return names
