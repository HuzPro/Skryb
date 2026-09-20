"""Read the text around the caret straight from a Win32 edit control.

The classic ``EDIT`` and ``RichEdit`` windows answer selection and text
queries themselves, which reaches controls that expose no UI Automation text
pattern. WinForms ``TextBox`` is the one that matters: .NET wraps a plain
``EDIT`` window and publishes no text pattern for it.

Read-only throughout. Only ``EM_GETSEL``, ``WM_GETTEXTLENGTH`` and
``WM_GETTEXT`` are sent, so no window is activated and no input is injected.
"""

import ctypes
import logging
from functools import cache

from voicetotext.output.caret_context import (
    CONTEXT_CHARS,
    NO_CONTEXT,
    CaretContext,
)


_logger = logging.getLogger(__name__)

_EM_GETSEL = 0x00B0
_WM_GETTEXT = 0x000D
_WM_GETTEXTLENGTH = 0x000E

_SMTO_BLOCK = 0x0001
_SMTO_ABORTIFHUNG = 0x0002
_MESSAGE_FLAGS = _SMTO_BLOCK | _SMTO_ABORTIFHUNG
_MESSAGE_TIMEOUT_MS = 200

_CLASS_NAME_BUFFER_CHARS = 256
_EDIT_CLASS_MARKER = "EDIT"

_LOGGED_TEXT_CHARS = 20

_NO_WPARAM = 0
_NO_LPARAM = 0


class _GuiThreadInfo(ctypes.Structure):
    """``GUITHREADINFO``, built from plain ctypes so Linux can import this."""

    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("flags", ctypes.c_ulong),
        ("hwndActive", ctypes.c_void_p),
        ("hwndFocus", ctypes.c_void_p),
        ("hwndCapture", ctypes.c_void_p),
        ("hwndMenuOwner", ctypes.c_void_p),
        ("hwndMoveSize", ctypes.c_void_p),
        ("hwndCaret", ctypes.c_void_p),
        ("rcCaret", ctypes.c_long * 4),
    ]


def read_edit_control_context() -> CaretContext:
    """Text around the caret in the focused edit control, or ``NO_CONTEXT``.

    Never raises. No keyboard focus, a window that is not an edit control, a
    hung or unresponsive target and a failed Win32 call all read as "no
    context", which callers treat as ordinary data.
    """
    try:
        return edit_control_context_of(_focused_hwnd())
    except Exception:
        _logger.debug("Reading the edit control context failed.", exc_info=True)
        return NO_CONTEXT


def edit_control_context_of(hwnd: int) -> CaretContext:
    """Same reading, aimed at one window handle instead of the focused one."""
    try:
        return _context_of(hwnd)
    except Exception:
        _logger.debug(
            "Reading the edit control context of %s failed.", hwnd, exc_info=True,
        )
        return NO_CONTEXT


def _context_of(hwnd: int) -> CaretContext:
    if not hwnd or not _is_edit_control(_class_name(hwnd)):
        return NO_CONTEXT
    start, end = _selection(hwnd)
    text = _text(hwnd)
    context = CaretContext(
        before=text[max(0, start - CONTEXT_CHARS):start],
        selected=text[start:end],
        after=text[end:end + CONTEXT_CHARS],
    )
    _log_context(context)
    return context


def _is_edit_control(class_name: str) -> bool:
    """Whether the window answers the edit-control messages.

    ``WM_GETTEXT`` on any other window hands back its title, which must never
    be mistaken for document text. Every edit class contains ``EDIT`` once
    upper-cased: ``Edit``, ``RichEdit20W``, ``RICHEDIT50W`` and the WinForms
    wrappers ``WindowsForms10.EDIT.app.0.x`` and
    ``WindowsForms10.RichEdit20W.app.0.x``.
    """
    return _EDIT_CLASS_MARKER in class_name.upper()


def _focused_hwnd() -> int:
    """The control with the keyboard focus, not just the foreground window.

    ``GetFocus`` only answers for the calling thread, so the focused child has
    to come from the foreground thread's own GUI state.
    """
    user32 = _user32()
    thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    info = _GuiThreadInfo()
    info.cbSize = ctypes.sizeof(_GuiThreadInfo)
    if not user32.GetGUIThreadInfo(thread, ctypes.byref(info)):
        return 0
    return int(info.hwndFocus or 0)


def _class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(_CLASS_NAME_BUFFER_CHARS)
    _user32().GetClassNameW(hwnd, buffer, _CLASS_NAME_BUFFER_CHARS)
    return buffer.value


def _selection(hwnd: int) -> tuple[int, int]:
    start = ctypes.c_ulong(0)
    end = ctypes.c_ulong(0)
    _send(hwnd, _EM_GETSEL, ctypes.byref(start), ctypes.byref(end))
    return start.value, end.value


def _text(hwnd: int) -> str:
    length = _send(hwnd, _WM_GETTEXTLENGTH, _NO_WPARAM, _NO_LPARAM)
    buffer = ctypes.create_unicode_buffer(length + 1)
    _send(hwnd, _WM_GETTEXT, length + 1, buffer)
    return buffer.value


def _send(hwnd: int, message: int, wparam: object, lparam: object) -> int:
    result = ctypes.c_size_t(0)
    delivered = _user32().SendMessageTimeoutW(
        hwnd,
        message,
        wparam,
        lparam,
        _MESSAGE_FLAGS,
        _MESSAGE_TIMEOUT_MS,
        ctypes.byref(result),
    )
    if not delivered:
        raise OSError(f"window {hwnd:#x} did not answer message {message:#x}")
    return int(result.value)


def _log_context(context: CaretContext) -> None:
    _logger.debug(
        "Edit control context: before=%r selected=%r after=%r",
        context.before[-_LOGGED_TEXT_CHARS:],
        context.selected[:_LOGGED_TEXT_CHARS],
        context.after[:_LOGGED_TEXT_CHARS],
    )


@cache
def _user32() -> ctypes.CDLL:
    """``user32`` with the signatures this module needs.

    Resolved lazily because ``ctypes.windll`` does not exist off Windows, and
    :mod:`voicetotext.output.win32_caret_context` imports this module on every
    platform.
    """
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    user32.GetWindowThreadProcessId.restype = ctypes.c_ulong
    user32.GetGUIThreadInfo.argtypes = [
        ctypes.c_ulong, ctypes.POINTER(_GuiThreadInfo),
    ]
    user32.GetGUIThreadInfo.restype = ctypes.c_int
    user32.GetClassNameW.argtypes = [
        ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int,
    ]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.SendMessageTimeoutW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    user32.SendMessageTimeoutW.restype = ctypes.c_ssize_t
    return user32
