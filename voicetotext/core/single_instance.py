import ctypes
import ctypes.wintypes as wintypes
from typing import Protocol


_ERROR_ALREADY_EXISTS = 183
_MUTEX_NAME = "Local\\VoiceToText"


class KernelMutexApi(Protocol):
    def create_mutex(self, name: str) -> int: ...
    def last_error(self) -> int: ...
    def close_handle(self, handle: int) -> None: ...


class Win32KernelMutexApi:
    def __init__(self) -> None:
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR,
        ]
        self._kernel32.CreateMutexW.restype = wintypes.HANDLE
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def create_mutex(self, name: str) -> int:
        handle = self._kernel32.CreateMutexW(None, False, name)
        return int(handle or 0)

    def last_error(self) -> int:
        return int(ctypes.get_last_error())

    def close_handle(self, handle: int) -> None:
        self._kernel32.CloseHandle(handle)


class SingleInstanceMutex:
    def __init__(self, api: KernelMutexApi | None = None) -> None:
        self._api = api or Win32KernelMutexApi()
        self._handle = 0

    def acquire(self) -> bool:
        self._handle = self._api.create_mutex(_MUTEX_NAME)
        if not self._handle:
            raise OSError("CreateMutexW failed")
        if self._api.last_error() == _ERROR_ALREADY_EXISTS:
            self.release()
            return False
        return True

    def release(self) -> None:
        if self._handle:
            self._api.close_handle(self._handle)
            self._handle = 0
