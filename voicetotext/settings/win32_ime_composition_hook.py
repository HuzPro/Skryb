"""Win32 window-procedure hook that delivers IME results Tk 8.6 drops."""

import ctypes
import functools
import tkinter
from tkinter import ttk

_WM_IME_COMPOSITION = 0x010F
_GCS_RESULTSTR = 0x0800
_GWLP_WNDPROC = -4
_MESSAGE_HANDLED = 0
_HIGHEST_BMP_CODEPOINT = 0xFFFF
_UTF16_ENCODING = "utf-16-le"
_WINDOW_PROC_SIGNATURE = (
    ctypes.c_ssize_t,
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_size_t,
    ctypes.c_ssize_t,
)


def insert_as_typed(entry: ttk.Entry, text: str) -> None:
    """Insert text at the caret, replacing any selection, as a keypress would."""
    entry.tk.call("ttk::entry::Insert", entry, text)


def needs_surrogate_pairs(text: str) -> bool:
    return any(ord(char) > _HIGHEST_BMP_CODEPOINT for char in text)


class Win32ImeCompositionHook:
    """Inserts IME results containing surrogate pairs into the focused entry."""

    # Tk 8.6 HandleIMEComposition (tkWinX.c) exits at the first UTF-16
    # surrogate, so emoji from the Windows emoji panel never reach the widget.

    def __init__(self, window: tkinter.Toplevel) -> None:
        self._window = window
        window_proc_type = ctypes.WINFUNCTYPE(*_WINDOW_PROC_SIGNATURE)
        self._proc = window_proc_type(self._window_proc)
        self._original_proc = _user32().SetWindowLongPtrW(
            window.winfo_id(),
            _GWLP_WNDPROC,
            ctypes.cast(self._proc, ctypes.c_void_p),
        )
        if not self._original_proc:
            raise ctypes.WinError()

    def _window_proc(
        self, hwnd: int, message: int, w_param: int, l_param: int,
    ) -> int:
        if message == _WM_IME_COMPOSITION and l_param & _GCS_RESULTSTR:
            text = _composition_result(hwnd)
            focused = self._window.focus_get()
            if needs_surrogate_pairs(text) and isinstance(focused, ttk.Entry):
                insert_as_typed(focused, text)
                return _MESSAGE_HANDLED
        return _user32().CallWindowProcW(
            self._original_proc, hwnd, message, w_param, l_param,
        )


def _composition_result(hwnd: int) -> str:
    imm32 = _imm32()
    context = imm32.ImmGetContext(hwnd)
    if not context:
        return ""
    try:
        size = imm32.ImmGetCompositionStringW(context, _GCS_RESULTSTR, None, 0)
        if size <= 0:
            return ""
        buffer = ctypes.create_string_buffer(size)
        imm32.ImmGetCompositionStringW(context, _GCS_RESULTSTR, buffer, size)
        return buffer.raw.decode(_UTF16_ENCODING)
    finally:
        imm32.ImmReleaseContext(hwnd, context)


@functools.cache
def _user32() -> ctypes.WinDLL:
    user32 = ctypes.windll.user32
    user32.SetWindowLongPtrW.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.CallWindowProcW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_ssize_t,
    ]
    user32.CallWindowProcW.restype = ctypes.c_ssize_t
    return user32


@functools.cache
def _imm32() -> ctypes.WinDLL:
    imm32 = ctypes.windll.imm32
    imm32.ImmGetContext.argtypes = [ctypes.c_void_p]
    imm32.ImmGetContext.restype = ctypes.c_void_p
    imm32.ImmReleaseContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    imm32.ImmReleaseContext.restype = ctypes.c_int
    imm32.ImmGetCompositionStringW.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint,
    ]
    imm32.ImmGetCompositionStringW.restype = ctypes.c_long
    return imm32
