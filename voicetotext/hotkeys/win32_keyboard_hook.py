import ctypes
import ctypes.wintypes as _wt
import threading
from collections.abc import Callable

from voicetotext.core.exceptions import HotkeyError


_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32

_WH_KEYBOARD_LL = 13
_HC_ACTION = 0
_WM_KEYDOWN = 0x0100
_WM_KEYUP = 0x0101
_WM_SYSKEYDOWN = 0x0104
_WM_SYSKEYUP = 0x0105
_WM_QUIT = 0x0012
_SUPPRESS = 1

_KEY_DOWN_MESSAGES = frozenset({_WM_KEYDOWN, _WM_SYSKEYDOWN})
_KEY_UP_MESSAGES = frozenset({_WM_KEYUP, _WM_SYSKEYUP})

_READY_TIMEOUT_SECONDS = 2.0
_JOIN_TIMEOUT_SECONDS = 2.0

_ULONG_PTR = ctypes.c_size_t
_LRESULT = ctypes.c_ssize_t


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", _wt.DWORD),
        ("scanCode", _wt.DWORD),
        ("flags", _wt.DWORD),
        ("time", _wt.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


_HOOKPROC = ctypes.WINFUNCTYPE(_LRESULT, ctypes.c_int, _wt.WPARAM, _wt.LPARAM)

_user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int, _HOOKPROC, _wt.HMODULE, _wt.DWORD,
]
_user32.SetWindowsHookExW.restype = ctypes.c_void_p
_user32.CallNextHookEx.argtypes = [
    ctypes.c_void_p, ctypes.c_int, _wt.WPARAM, _wt.LPARAM,
]
_user32.CallNextHookEx.restype = _LRESULT
_user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
_user32.UnhookWindowsHookEx.restype = _wt.BOOL
_user32.GetMessageW.argtypes = [
    ctypes.POINTER(_wt.MSG), _wt.HWND, _wt.UINT, _wt.UINT,
]
_user32.GetMessageW.restype = ctypes.c_int
_user32.TranslateMessage.argtypes = [ctypes.POINTER(_wt.MSG)]
_user32.TranslateMessage.restype = _wt.BOOL
_user32.DispatchMessageW.argtypes = [ctypes.POINTER(_wt.MSG)]
_user32.DispatchMessageW.restype = _LRESULT
_user32.PostThreadMessageW.argtypes = [
    _wt.DWORD, _wt.UINT, _wt.WPARAM, _wt.LPARAM,
]
_user32.PostThreadMessageW.restype = _wt.BOOL
_kernel32.GetModuleHandleW.argtypes = [_wt.LPCWSTR]
_kernel32.GetModuleHandleW.restype = _wt.HMODULE
_kernel32.GetCurrentThreadId.restype = _wt.DWORD


# Decides what to do with each key event. Receives ``(is_key_down, vk)`` and
# returns True to SUPPRESS the key (stop it reaching other applications).
KeyEventHandler = Callable[[bool, int], bool]


class Win32KeyboardHook:
    """A low-level keyboard hook running on its own message-pump thread.

    A ``WH_KEYBOARD_LL`` hook only fires on a thread that pumps messages, so
    ``start`` spawns a daemon thread that installs the hook and runs a
    ``GetMessage`` loop; ``stop`` posts ``WM_QUIT`` to break it and unhooks.

    ``start`` blocks on a readiness event until the worker has published its
    thread id AND created its message queue (via ``SetWindowsHookExW``), so
    ``stop`` can never race ahead and drop the ``WM_QUIT`` that frees the
    thread. The hook proc is pinned on the instance - a ``WINFUNCTYPE`` the
    GC collects mid-flight would crash the interpreter.
    """

    def __init__(self, handler: KeyEventHandler) -> None:
        self._handler = handler
        self._proc = _HOOKPROC(self._hook_proc)
        self._thread: threading.Thread | None = None
        self._thread_id: int = 0
        self._hook: int | None = None
        self._ready = threading.Event()
        self._start_error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=_READY_TIMEOUT_SECONDS)
        self._raise_if_start_failed()

    def _raise_if_start_failed(self) -> None:
        if self._start_error is None:
            return
        error = self._start_error
        self._thread = None
        self._thread_id = 0
        self._start_error = None
        raise error

    def stop(self) -> None:
        if self._thread is None:
            return
        _user32.PostThreadMessageW(self._thread_id, _WM_QUIT, 0, 0)
        self._thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise HotkeyError(
                "Keyboard hook thread did not exit; the hook may still be installed."
            )
        self._thread = None
        self._thread_id = 0

    def _run(self) -> None:
        self._thread_id = _kernel32.GetCurrentThreadId()
        if not self._install_hook_and_signal_ready():
            return
        try:
            self._pump()
        finally:
            self._uninstall_hook()

    def _install_hook_and_signal_ready(self) -> bool:
        # Publish readiness in a finally so a failed install still releases
        # ``start``; the stored error is then re-raised there.
        try:
            self._install_hook()
            return True
        except Exception as exc:  # surfaced to start() via _start_error
            self._start_error = exc
            return False
        finally:
            self._ready.set()

    def _install_hook(self) -> None:
        self._hook = _user32.SetWindowsHookExW(
            _WH_KEYBOARD_LL, self._proc, _kernel32.GetModuleHandleW(None), 0,
        )
        if not self._hook:
            raise HotkeyError(
                "Could not install the global keyboard hook "
                "(SetWindowsHookExW returned NULL)."
            )

    def _uninstall_hook(self) -> None:
        if self._hook:
            _user32.UnhookWindowsHookEx(self._hook)
            self._hook = None

    def _pump(self) -> None:
        message = _wt.MSG()
        while _user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            _user32.TranslateMessage(ctypes.byref(message))
            _user32.DispatchMessageW(ctypes.byref(message))

    def _hook_proc(self, n_code: int, w_param: int, l_param: int) -> int:
        if n_code == _HC_ACTION and self._dispatch(w_param, l_param):
            return _SUPPRESS
        return _user32.CallNextHookEx(None, n_code, w_param, l_param)

    def _dispatch(self, message: int, l_param: int) -> bool:
        info = ctypes.cast(
            l_param, ctypes.POINTER(_KBDLLHOOKSTRUCT),
        ).contents
        if message in _KEY_DOWN_MESSAGES:
            return self._handler(True, info.vkCode)
        if message in _KEY_UP_MESSAGES:
            return self._handler(False, info.vkCode)
        return False
