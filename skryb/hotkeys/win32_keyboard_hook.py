import ctypes
import ctypes.wintypes as _wt
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from skryb.core.exceptions import HotkeyError


_logger = logging.getLogger(__name__)

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
_KEY_MESSAGES = _KEY_DOWN_MESSAGES | _KEY_UP_MESSAGES

# Codes no key on a keyboard produces: 0 and 0xFF come from scan codes with
# no virtual key, and VK_PACKET carries a character injected as text. None
# can be part of a shortcut, so they only ever go past the handler.
_VK_PACKET = 0xE7
_VK_NONE = 0xFF
_NOT_KEYS = frozenset({0, _VK_PACKET, _VK_NONE})

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


# Decides what to do with each key or mouse-button event. Receives
# ``(is_key_down, vk)`` and returns True to SUPPRESS it (stop it reaching
# other applications).
KeyEventHandler = Callable[[bool, int], bool]

# Watches key events without a say in suppression. Receives
# ``(is_key_down, vk, flags, extra)``, the last two straight off
# ``KBDLLHOOKSTRUCT``. A second consumer rides this hook rather than
# installing its own: two Python hook procs per keystroke double the odds
# Windows silently drops one for exceeding ``LowLevelHooksTimeout``.
KeyObserver = Callable[[bool, int, int, int], None]


def ignore_key_event(is_key_down: bool, vk: int, flags: int, extra: int) -> None:
    return None


class Win32LowLevelHook(ABC):
    """A low-level input hook running on its own message-pump thread.

    A ``WH_KEYBOARD_LL``/``WH_MOUSE_LL`` hook only fires on a thread that
    pumps messages, so ``start`` spawns a daemon thread that installs the
    hook and runs a ``GetMessage`` loop; ``stop`` posts ``WM_QUIT`` to break
    it and unhooks.

    ``start`` blocks on a readiness event until the worker has published its
    thread id AND created its message queue (via ``SetWindowsHookExW``), so
    ``stop`` can never race ahead and drop the ``WM_QUIT`` that frees the
    thread. The hook proc is pinned on the instance - a ``WINFUNCTYPE`` the
    GC collects mid-flight would crash the interpreter.

    Nothing raised while handling an event leaves the hook proc: the first
    failure is logged and every failing event is passed on unchanged.
    """

    _hook_type: int
    _device: str

    def __init__(self, handler: KeyEventHandler) -> None:
        self._handler = handler
        self._proc = _HOOKPROC(self._hook_proc)
        self._thread: threading.Thread | None = None
        self._thread_id: int = 0
        self._hook: int | None = None
        self._ready = threading.Event()
        self._start_error: Exception | None = None
        self._failure_logged = False

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
                f"The {self._device} hook thread did not exit; "
                "the hook may still be installed."
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
            self._hook_type, self._proc, _kernel32.GetModuleHandleW(None), 0,
        )
        if not self._hook:
            raise HotkeyError(
                f"Could not install the global {self._device} hook "
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
        if n_code == _HC_ACTION and l_param and self._swallows(w_param, l_param):
            return _SUPPRESS
        return _user32.CallNextHookEx(None, n_code, w_param, l_param)

    def _swallows(self, message: int, l_param: int) -> bool:
        try:
            return self._dispatch(message, l_param)
        except Exception:
            self._log_first_failure()
            return False

    def _log_first_failure(self) -> None:
        if self._failure_logged:
            return
        self._failure_logged = True
        _logger.exception(
            "The %s hook failed on an event; passing events on.", self._device,
        )

    @abstractmethod
    def _dispatch(self, message: int, l_param: int) -> bool:
        """Hand one hook message to the handler; True to swallow it."""


class Win32KeyboardHook(Win32LowLevelHook):
    """Every key down and up, for a handler that may swallow it and an
    observer that watches whatever the handler let through."""

    _hook_type = _WH_KEYBOARD_LL
    _device = "keyboard"

    def __init__(
        self,
        handler: KeyEventHandler,
        observer: KeyObserver = ignore_key_event,
    ) -> None:
        super().__init__(handler)
        self._observer = observer

    def _dispatch(self, message: int, l_param: int) -> bool:
        if message not in _KEY_MESSAGES:
            return False
        info = ctypes.cast(
            l_param, ctypes.POINTER(_KBDLLHOOKSTRUCT),
        ).contents
        return self._handle_then_observe(message in _KEY_DOWN_MESSAGES, info)

    def _handle_then_observe(
        self, is_key_down: bool, info: _KBDLLHOOKSTRUCT,
    ) -> bool:
        suppress = info.vkCode not in _NOT_KEYS and self._handler(
            is_key_down, info.vkCode,
        )
        # A swallowed key never reached the focused app, so watchers of
        # what the user typed there must not see it either.
        if not suppress:
            self._observe(is_key_down, info)
        return suppress

    def _observe(self, is_key_down: bool, info: _KBDLLHOOKSTRUCT) -> None:
        # An observer fault must not reach the hook proc: an exception there
        # escapes into a Windows callback, and the hotkeys it breaks are the
        # only way to start recording.
        try:
            self._notify_observer(is_key_down, info)
        except Exception:
            _logger.exception("Keyboard hook observer raised; ignored.")

    def _notify_observer(
        self, is_key_down: bool, info: _KBDLLHOOKSTRUCT,
    ) -> None:
        self._observer(is_key_down, info.vkCode, info.flags, info.dwExtraInfo)
