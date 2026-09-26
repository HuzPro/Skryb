"""Win32 window-procedure hook that delivers IME results Tk drops."""

import ctypes
import functools
import logging
import tkinter

_logger = logging.getLogger(__name__)

_WM_IME_COMPOSITION = 0x010F
_WM_UNICHAR = 0x0109
_GCS_RESULTSTR = 0x0800
_GWLP_WNDPROC = -4
_MESSAGE_HANDLED = 0
_HIGHEST_BMP_CODEPOINT = 0xFFFF
_UTF16_ENCODING = "utf-16-le"
_UTF16_CODE_UNIT_BYTES = 2
_LRESULT = ctypes.c_ssize_t
_ULONG_PTR = ctypes.c_size_t


def contains_non_bmp_characters(text: str) -> bool:
    return any(ord(char) > _HIGHEST_BMP_CODEPOINT for char in text)


def unichar_reposts(text: str) -> tuple[int, ...]:
    """Code points to repost as WM_UNICHAR, empty when Tk handles the text itself."""
    if not contains_non_bmp_characters(text):
        return ()
    return tuple(ord(char) for char in text)


class Win32ImeCompositionHook:
    """Reposts emoji IME results as keys; keep the instance alive to pin the window proc."""

    # Tk 8.6 HandleIMEComposition (tkWinX.c) exits at the first UTF-16
    # surrogate, so emoji from the Windows emoji panel never reach the widget.

    def __init__(self, window: tkinter.Toplevel) -> None:
        self._window = window
        self._hwnd = window.winfo_id()
        self._restored = False
        self._failure_logged = False
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

    # This runs inside Tcl_DoOneEvent, so it must not touch tkinter: _tkinter keeps
    # one global interpreter state that a nested call clears, aborting the next callback.
    def _window_proc(
        self, hwnd: int, message: int, w_param: int, l_param: int,
    ) -> int:
        try:
            return self._answer(hwnd, message, w_param, l_param)
        except Exception:
            self._log_first_failure()
            return int(_user32().DefWindowProcW(hwnd, message, w_param, l_param))

    def _answer(self, hwnd: int, message: int, w_param: int, l_param: int) -> int:
        if (
            message == _WM_IME_COMPOSITION
            and l_param & _GCS_RESULTSTR
            and self._reposted_composition_result(hwnd)
        ):
            return _MESSAGE_HANDLED
        return int(
            self._call_original(
                self._original_proc, hwnd, message, w_param, l_param,
            ),
        )

    def _log_first_failure(self) -> None:
        if self._failure_logged:
            return
        self._failure_logged = True
        _logger.exception("The IME window hook failed on a message; passing messages on.")

    def _reposted_composition_result(self, hwnd: int) -> bool:
        try:
            return _repost_as_unichars(hwnd, _composition_result(hwnd))
        except Exception:
            _logger.exception("IME composition result could not be reposted")
            return False

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


def _repost_as_unichars(hwnd: int, text: str) -> bool:
    code_points = unichar_reposts(text)
    post_message = _user32().PostMessageW
    for code_point in code_points:
        post_message(hwnd, _WM_UNICHAR, code_point, 0)
    return bool(code_points)


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
    user32.DefWindowProcW.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, _ULONG_PTR, _LRESULT,
    ]
    user32.DefWindowProcW.restype = _LRESULT
    user32.PostMessageW.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, _ULONG_PTR, _LRESULT,
    ]
    user32.PostMessageW.restype = ctypes.c_int
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
