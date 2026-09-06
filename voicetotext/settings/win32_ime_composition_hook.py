"""Win32 window-procedure hook that delivers IME results Tk drops."""

import ctypes
import functools
import logging
import tkinter
from tkinter import ttk

_logger = logging.getLogger(__name__)

_WM_IME_COMPOSITION = 0x010F
_GCS_RESULTSTR = 0x0800
_GWLP_WNDPROC = -4
_MESSAGE_HANDLED = 0
_HIGHEST_BMP_CODEPOINT = 0xFFFF
_UTF16_ENCODING = "utf-16-le"
_UTF16_CODE_UNIT_BYTES = 2
_LRESULT = ctypes.c_ssize_t
_ULONG_PTR = ctypes.c_size_t


def insert_as_typed(entry: ttk.Entry, text: str) -> None:
    """Insert text at the caret, replacing any selection, as a keypress would."""
    entry.tk.call("ttk::entry::Insert", entry, text)


def contains_non_bmp_characters(text: str) -> bool:
    return any(ord(char) > _HIGHEST_BMP_CODEPOINT for char in text)


class Win32ImeCompositionHook:
    """Inserts emoji IME results; keep the instance alive to pin the window proc."""

    # Tk 8.6 HandleIMEComposition (tkWinX.c) exits at the first UTF-16
    # surrogate, so emoji from the Windows emoji panel never reach the widget.

    def __init__(self, window: tkinter.Toplevel) -> None:
        self._window = window
        self._hwnd = window.winfo_id()
        self._restored = False
        user32 = _user32()
        self._call_original = user32.CallWindowProcW
        self._proc = ctypes.WINFUNCTYPE(
            _LRESULT, ctypes.c_void_p, ctypes.c_uint, _ULONG_PTR, _LRESULT,
        )(self._window_proc)
        self._original_proc = user32.SetWindowLongPtrW(
            self._hwnd, _GWLP_WNDPROC, ctypes.cast(self._proc, ctypes.c_void_p),
        )
        if not self._original_proc:
            raise OSError(
                f"SetWindowLongPtrW could not hook window {self._hwnd}",
            )
        window.bind("<Destroy>", self._on_destroy, add="+")

    def _window_proc(
        self, hwnd: int, message: int, w_param: int, l_param: int,
    ) -> int:
        if (
            message == _WM_IME_COMPOSITION
            and l_param & _GCS_RESULTSTR
            and self._inserted_composition_result(hwnd)
        ):
            return _MESSAGE_HANDLED
        return int(
            self._call_original(
                self._original_proc, hwnd, message, w_param, l_param,
            ),
        )

    def _inserted_composition_result(self, hwnd: int) -> bool:
        try:
            return self._insert_non_bmp_result(hwnd)
        except Exception:
            _logger.exception("IME composition result could not be inserted")
            return False

    def _insert_non_bmp_result(self, hwnd: int) -> bool:
        text = _composition_result(hwnd)
        if not contains_non_bmp_characters(text):
            return False
        focused = self._window.focus_get()
        if not isinstance(focused, ttk.Entry):
            return False
        insert_as_typed(focused, text)
        return True

    def _on_destroy(self, event: "tkinter.Event[tkinter.Misc]") -> None:
        if event.widget is self._window:
            self._restore_original_proc()

    def _restore_original_proc(self) -> None:
        if self._restored:
            return
        self._restored = True
        _user32().SetWindowLongPtrW(
            self._hwnd, _GWLP_WNDPROC, self._original_proc,
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
        whole_units = size - size % _UTF16_CODE_UNIT_BYTES
        return buffer.raw[:whole_units].decode(
            _UTF16_ENCODING, errors="replace",
        )
    finally:
        imm32.ImmReleaseContext(hwnd, context)


@functools.cache
def _user32() -> ctypes.CDLL:
    user32 = ctypes.windll.user32
    user32.SetWindowLongPtrW.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.CallWindowProcW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        _ULONG_PTR,
        _LRESULT,
    ]
    user32.CallWindowProcW.restype = _LRESULT
    return user32


@functools.cache
def _imm32() -> ctypes.CDLL:
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
