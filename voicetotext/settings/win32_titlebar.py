import ctypes
import tkinter

# DWM window attribute that paints the native title bar dark. Index 20 on
# Windows 10 build 19041+ and Windows 11; index 19 on the earlier insider
# builds where the attribute first appeared.
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_LEGACY = 19
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SWP_FRAMECHANGED = 0x0020
_REPAINT_FRAME = (
    _SWP_NOSIZE | _SWP_NOMOVE | _SWP_NOZORDER | _SWP_NOACTIVATE | _SWP_FRAMECHANGED
)


def show_with_titlebar(window: tkinter.Wm, *, dark: bool) -> None:
    """Map *window* with its caption already the right colour.

    The frame window whose caption Windows paints only exists once Tk has
    mapped the window, so the window is mapped fully transparent, themed,
    then made visible; otherwise the caption flashes white first.
    """
    window.attributes("-alpha", 0.0)
    window.deiconify()
    window.update_idletasks()
    set_titlebar_dark(window, dark=dark)
    window.attributes("-alpha", 1.0)


def set_titlebar_dark(window: object, *, dark: bool) -> None:
    """Best-effort: tint a Tk window's native title bar (Windows 10+).

    Tkinter never themes the OS title bar, so a dark-themed window would
    keep a white caption. This sets the DWM immersive-dark-mode attribute
    on the window's frame and makes Windows repaint the caption at once;
    without the repaint the active caption stays white until the window
    next loses focus. Call it once the window is mapped: before that Tk
    has no frame window to theme. It silently no-ops off Windows, on older
    builds, or if the handle cannot be resolved.
    """
    try:
        _set_immersive_dark_mode(int(window.wm_frame(), 16), dark)
    except Exception:
        pass


def _set_immersive_dark_mode(hwnd: int, dark: bool) -> None:
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi
    dwmapi.DwmSetWindowAttribute.argtypes = [
        wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD,
    ]
    user32.SetWindowPos.argtypes = [
        wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint,
    ]
    value = ctypes.c_int(1 if dark else 0)
    for attribute in (
        _DWMWA_USE_IMMERSIVE_DARK_MODE,
        _DWMWA_USE_IMMERSIVE_DARK_MODE_LEGACY,
    ):
        result = dwmapi.DwmSetWindowAttribute(
            hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value),
        )
        if result == 0:
            break
    user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, _REPAINT_FRAME)
