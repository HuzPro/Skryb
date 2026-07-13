import ctypes

# DWM window attribute that paints the native title bar dark. Index 20 on
# Windows 10 build 19041+ and Windows 11; index 19 on the earlier insider
# builds where the attribute first appeared.
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_LEGACY = 19


def set_titlebar_dark(window: object, *, dark: bool) -> None:
    """Best-effort: tint a Tk window's native title bar (Windows 10+).

    Tkinter never themes the OS title bar, so a dark-themed window would
    keep a white caption. This sets the DWM immersive-dark-mode attribute
    on the window's frame. It silently no-ops off Windows, on older builds,
    or if the handle cannot be resolved.
    """
    try:
        _set_immersive_dark_mode(window.winfo_id(), dark)
    except Exception:
        pass


def _set_immersive_dark_mode(child_hwnd: int, dark: bool) -> None:
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi
    user32.GetParent.argtypes = [wt.HWND]
    user32.GetParent.restype = wt.HWND
    dwmapi.DwmSetWindowAttribute.argtypes = [
        wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD,
    ]
    # Tk's ``winfo_id`` returns the client window; the framed (title-bar)
    # window DWM themes is its parent.
    hwnd = user32.GetParent(child_hwnd) or child_hwnd
    value = ctypes.c_int(1 if dark else 0)
    for attribute in (
        _DWMWA_USE_IMMERSIVE_DARK_MODE,
        _DWMWA_USE_IMMERSIVE_DARK_MODE_LEGACY,
    ):
        result = dwmapi.DwmSetWindowAttribute(
            hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value),
        )
        if result == 0:
            return
