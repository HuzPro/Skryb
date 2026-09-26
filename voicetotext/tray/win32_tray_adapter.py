import ctypes
import ctypes.wintypes as _wt
import logging
import tkinter
from collections.abc import Callable, Sequence
from typing import Protocol

from voicetotext.core.state_machine import AppState
from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.tray.duration_format import format_duration
from voicetotext.tray.win32_icons import create_icon_image

_logger = logging.getLogger(__name__)

_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32
_shell32 = ctypes.windll.shell32
_kernel32 = ctypes.windll.kernel32

_TITLE_PREFIX = "VoiceToText - "
_ELAPSED_REFRESH_INTERVAL_MS = 1000
_ICON_SIZE = 32

# Shell_NotifyIcon messages / flags.
_NIM_ADD = 0x0
_NIM_MODIFY = 0x1
_NIM_DELETE = 0x2
_NIF_MESSAGE = 0x1
_NIF_ICON = 0x2
_NIF_TIP = 0x4
_NIF_INFO = 0x10
_NIIF_INFO = 0x1

# Window / menu plumbing.
_WM_APP_TRAY = 0x0400 + 1  # WM_APP + 1: our tray callback message
_WM_RBUTTONUP = 0x0205
_WM_CONTEXTMENU = 0x007B
_WM_LBUTTONDBLCLK = 0x0203
_WM_DEVICECHANGE = 0x0219
_DBT_DEVNODES_CHANGED = 0x0007
_WS_EX_TOOLWINDOW = 0x80
_MF_STRING = 0x0
_MF_SEPARATOR = 0x800
_MF_GRAYED = 0x1
_MF_CHECKED = 0x8
_MF_POPUP = 0x10
_TPM_RIGHTBUTTON = 0x2
_TPM_RETURNCMD = 0x100

_ID_STATE = 1
_ID_SETTINGS = 2
_ID_QUIT = 3
_ID_HISTORY = 4
# Microphone choices take one id each from here up.
_ID_FIRST_MICROPHONE = 100
_SYSTEM_DEFAULT_LABEL = "Windows default"

# Returns the microphones to offer as (name, label) and the chosen name
# ("" = the Windows default).
MicrophoneLister = Callable[[], tuple[Sequence[tuple[str, str]], str]]

_LRESULT = ctypes.c_ssize_t
_WNDPROC = ctypes.WINFUNCTYPE(_LRESULT, _wt.HWND, _wt.UINT, _wt.WPARAM, _wt.LPARAM)

_user32.RegisterWindowMessageW.argtypes = [_wt.LPCWSTR]
_user32.RegisterWindowMessageW.restype = _wt.UINT
# Broadcast by the shell after Explorer restarts, once per surviving app.
_WM_TASKBAR_CREATED = _user32.RegisterWindowMessageW("TaskbarCreated")


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", _wt.DWORD),
        ("biWidth", _wt.LONG),
        ("biHeight", _wt.LONG),
        ("biPlanes", _wt.WORD),
        ("biBitCount", _wt.WORD),
        ("biCompression", _wt.DWORD),
        ("biSizeImage", _wt.DWORD),
        ("biXPelsPerMeter", _wt.LONG),
        ("biYPelsPerMeter", _wt.LONG),
        ("biClrUsed", _wt.DWORD),
        ("biClrImportant", _wt.DWORD),
    ]


class _ICONINFO(ctypes.Structure):
    _fields_ = [
        ("fIcon", _wt.BOOL),
        ("xHotspot", _wt.DWORD),
        ("yHotspot", _wt.DWORD),
        ("hbmMask", _wt.HBITMAP),
        ("hbmColor", _wt.HBITMAP),
    ]


class _NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ("cbSize", _wt.DWORD),
        ("hWnd", _wt.HWND),
        ("uID", _wt.UINT),
        ("uFlags", _wt.UINT),
        ("uCallbackMessage", _wt.UINT),
        ("hIcon", _wt.HICON),
        ("szTip", _wt.WCHAR * 128),
        ("dwState", _wt.DWORD),
        ("dwStateMask", _wt.DWORD),
        ("szInfo", _wt.WCHAR * 256),
        ("uVersion", _wt.UINT),
        ("szInfoTitle", _wt.WCHAR * 64),
        ("dwInfoFlags", _wt.DWORD),
        ("guidItem", _wt.BYTE * 16),
        ("hBalloonIcon", _wt.HICON),
    ]


class _WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", _wt.UINT),
        ("lpfnWndProc", _WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", _wt.HINSTANCE),
        ("hIcon", _wt.HICON),
        ("hCursor", _wt.HANDLE),
        ("hbrBackground", _wt.HBRUSH),
        ("lpszMenuName", _wt.LPCWSTR),
        ("lpszClassName", _wt.LPCWSTR),
    ]


_user32.CreateIconIndirect.argtypes = [ctypes.POINTER(_ICONINFO)]
_user32.CreateIconIndirect.restype = _wt.HICON
_user32.DestroyIcon.argtypes = [_wt.HICON]
_user32.DestroyIcon.restype = _wt.BOOL
_user32.GetDC.argtypes = [_wt.HWND]
_user32.GetDC.restype = _wt.HDC
_user32.ReleaseDC.argtypes = [_wt.HWND, _wt.HDC]
_user32.ReleaseDC.restype = ctypes.c_int
_gdi32.CreateDIBSection.argtypes = [
    _wt.HDC, ctypes.c_void_p, _wt.UINT,
    ctypes.POINTER(ctypes.c_void_p), _wt.HANDLE, _wt.DWORD,
]
_gdi32.CreateDIBSection.restype = _wt.HBITMAP
_gdi32.CreateBitmap.argtypes = [
    ctypes.c_int, ctypes.c_int, _wt.UINT, _wt.UINT, ctypes.c_void_p,
]
_gdi32.CreateBitmap.restype = _wt.HBITMAP
_gdi32.DeleteObject.argtypes = [_wt.HGDIOBJ]
_gdi32.DeleteObject.restype = _wt.BOOL

# Pointer-returning / pointer-taking calls MUST declare argtypes/restype or
# ctypes defaults to 32-bit ints and truncates handles on 64-bit Windows.
_kernel32.GetModuleHandleW.argtypes = [_wt.LPCWSTR]
_kernel32.GetModuleHandleW.restype = _wt.HMODULE
_user32.RegisterClassW.argtypes = [ctypes.POINTER(_WNDCLASSW)]
_user32.RegisterClassW.restype = _wt.ATOM
_user32.CreateWindowExW.argtypes = [
    _wt.DWORD, _wt.LPCWSTR, _wt.LPCWSTR, _wt.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _wt.HWND, _wt.HMENU, _wt.HINSTANCE, _wt.LPVOID,
]
_user32.CreateWindowExW.restype = _wt.HWND
_user32.DefWindowProcW.argtypes = [_wt.HWND, _wt.UINT, _wt.WPARAM, _wt.LPARAM]
_user32.DefWindowProcW.restype = _LRESULT
_user32.DestroyWindow.argtypes = [_wt.HWND]
_user32.DestroyWindow.restype = _wt.BOOL
_user32.CreatePopupMenu.argtypes = []
_user32.CreatePopupMenu.restype = _wt.HMENU
_user32.AppendMenuW.argtypes = [_wt.HMENU, _wt.UINT, ctypes.c_size_t, _wt.LPCWSTR]
_user32.AppendMenuW.restype = _wt.BOOL
_user32.DestroyMenu.argtypes = [_wt.HMENU]
_user32.DestroyMenu.restype = _wt.BOOL
_user32.TrackPopupMenu.argtypes = [
    _wt.HMENU, _wt.UINT, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, _wt.HWND, ctypes.c_void_p,
]
_user32.TrackPopupMenu.restype = _wt.BOOL
_user32.GetCursorPos.argtypes = [ctypes.POINTER(_wt.POINT)]
_user32.GetCursorPos.restype = _wt.BOOL
_user32.SetForegroundWindow.argtypes = [_wt.HWND]
_user32.SetForegroundWindow.restype = _wt.BOOL
_shell32.Shell_NotifyIconW.argtypes = [
    _wt.DWORD, ctypes.POINTER(_NOTIFYICONDATAW),
]
_shell32.Shell_NotifyIconW.restype = _wt.BOOL


def create_hicon(state: AppState, size: int = _ICON_SIZE) -> int:
    """Render the state circle and convert it to a Win32 ``HICON``.

    The caller owns the returned handle and must ``DestroyIcon`` it.
    """
    return image_to_hicon(create_icon_image(state, size))


def image_to_hicon(image: object) -> int:
    width, height = image.size
    bgra = image.tobytes("raw", "BGRA")

    header = _BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    header.biWidth = width
    header.biHeight = -height  # negative: top-down rows, matching PIL
    header.biPlanes = 1
    header.biBitCount = 32
    header.biCompression = 0  # BI_RGB

    bits = ctypes.c_void_p()
    hdc = _user32.GetDC(None)
    try:
        hbm_color = _gdi32.CreateDIBSection(
            hdc, ctypes.byref(header), 0, ctypes.byref(bits), None, 0,
        )
    finally:
        _user32.ReleaseDC(None, hdc)
    if not hbm_color:
        raise OSError("CreateDIBSection failed building the tray icon.")
    ctypes.memmove(bits, bgra, len(bgra))

    hbm_mask = _gdi32.CreateBitmap(width, height, 1, 1, None)
    info = _ICONINFO()
    info.fIcon = True
    info.hbmMask = hbm_mask
    info.hbmColor = hbm_color
    hicon = _user32.CreateIconIndirect(ctypes.byref(info))

    _gdi32.DeleteObject(hbm_color)
    _gdi32.DeleteObject(hbm_mask)
    if not hicon:
        raise OSError("CreateIconIndirect failed building the tray icon.")
    return hicon


class TrayBackend(Protocol):
    """Native operations the adapter drives; mocked in tests."""

    def set_icon(self, hicon: int) -> None: ...
    def set_tooltip(self, text: str) -> None: ...
    def notify(self, title: str, message: str) -> None: ...
    def show(self) -> None: ...
    def hide(self) -> None: ...
    def destroy(self) -> None: ...


class Win32TrayBackend:
    """Native ``Shell_NotifyIconW`` tray icon on a hidden top-level window.

    Created on the main thread so Tk's ``mainloop`` pumps its window
    messages. The window is deliberately unowned rather than message-only:
    ``HWND_MESSAGE`` windows are excluded from broadcasts, and the
    ``TaskbarCreated`` broadcast is the only notice we get that Explorer
    restarted and dropped our icon. NOT unit-tested end-to-end (it needs a
    live shell); the :class:`Win32TrayAdapter` orchestration is tested with
    a fake backend.
    """

    def __init__(
        self,
        get_state_label: Callable[[], str],
        on_settings_clicked: Callable[[], None] | None,
        on_quit_clicked: Callable[[], None],
        on_history_clicked: Callable[[], None] | None = None,
        list_microphones: MicrophoneLister | None = None,
        on_microphone_chosen: Callable[[str], None] | None = None,
        on_devices_changed: Callable[[], None] | None = None,
    ) -> None:
        self._get_state_label = get_state_label
        self._on_settings_clicked = on_settings_clicked
        self._on_quit_clicked = on_quit_clicked
        self._on_history_clicked = on_history_clicked
        self._list_microphones = list_microphones
        self._on_microphone_chosen = on_microphone_chosen
        self._on_devices_changed = on_devices_changed
        self._offered_microphones: list[str] = []
        self._added = False
        self._wndproc = _WNDPROC(self._window_proc)  # pinned: GC would crash us
        self._hwnd = self._create_tray_window()
        self._nid = self._build_nid()

    def _create_tray_window(self) -> int:
        wndclass = _WNDCLASSW()
        wndclass.lpfnWndProc = self._wndproc
        wndclass.hInstance = _kernel32.GetModuleHandleW(None)
        wndclass.lpszClassName = "VoiceToTextTray"
        self._class_atom = _user32.RegisterClassW(ctypes.byref(wndclass))
        hwnd = _user32.CreateWindowExW(
            _WS_EX_TOOLWINDOW, wndclass.lpszClassName, "VoiceToText", 0,
            0, 0, 0, 0, None, None, wndclass.hInstance, None,
        )
        if not hwnd:
            raise OSError("CreateWindowExW failed for the tray window.")
        return hwnd

    def _build_nid(self) -> _NOTIFYICONDATAW:
        nid = _NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(_NOTIFYICONDATAW)
        nid.hWnd = self._hwnd
        nid.uID = 1
        nid.uCallbackMessage = _WM_APP_TRAY
        return nid

    def set_icon(self, hicon: int) -> None:
        self._nid.hIcon = hicon
        if self._added:
            self._modify(_NIF_ICON)

    def set_tooltip(self, text: str) -> None:
        self._nid.szTip = text[:127]
        if self._added:
            self._modify(_NIF_TIP)

    def notify(self, title: str, message: str) -> None:
        if not self._added:
            return
        self._nid.szInfoTitle = title[:63]
        self._nid.szInfo = message[:255]
        self._nid.dwInfoFlags = _NIIF_INFO
        self._modify(_NIF_INFO)

    def show(self) -> None:
        if self._added:
            return
        self._nid.uFlags = _NIF_MESSAGE | _NIF_ICON | _NIF_TIP
        _shell32.Shell_NotifyIconW(_NIM_ADD, ctypes.byref(self._nid))
        self._added = True

    def hide(self) -> None:
        if not self._added:
            return
        _shell32.Shell_NotifyIconW(_NIM_DELETE, ctypes.byref(self._nid))
        self._added = False

    def destroy(self) -> None:
        self.hide()
        if self._hwnd:
            _user32.DestroyWindow(self._hwnd)
            self._hwnd = 0

    def _modify(self, flags: int) -> None:
        self._nid.uFlags = flags
        _shell32.Shell_NotifyIconW(_NIM_MODIFY, ctypes.byref(self._nid))

    def _window_proc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == _WM_APP_TRAY:
            self._handle_tray_event(lparam & 0xFFFF)
            return 0
        if msg == _WM_TASKBAR_CREATED:
            self._readd_after_explorer_restart()
            return 0
        # Windows sends this to every top-level window when any device is
        # plugged in or out; this window is top-level for broadcasts anyway.
        if (
            msg == _WM_DEVICECHANGE
            and wparam == _DBT_DEVNODES_CHANGED
            and self._on_devices_changed is not None
        ):
            self._on_devices_changed()
        return _user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _readd_after_explorer_restart(self) -> None:
        if not self._added:
            return
        self._added = False
        self.show()

    def _handle_tray_event(self, mouse_message: int) -> None:
        if mouse_message in (_WM_RBUTTONUP, _WM_CONTEXTMENU):
            self._show_context_menu()
        elif mouse_message == _WM_LBUTTONDBLCLK and self._on_settings_clicked:
            self._on_settings_clicked()

    def _show_context_menu(self) -> None:
        menu = _user32.CreatePopupMenu()
        _user32.AppendMenuW(
            menu, _MF_STRING | _MF_GRAYED, _ID_STATE, self._get_state_label(),
        )
        _user32.AppendMenuW(menu, _MF_SEPARATOR, 0, None)
        if self._on_history_clicked is not None:
            _user32.AppendMenuW(menu, _MF_STRING, _ID_HISTORY, "History...")
        self._append_microphone_menu(menu)
        if self._on_settings_clicked is not None:
            _user32.AppendMenuW(menu, _MF_STRING, _ID_SETTINGS, "Settings...")
            _user32.AppendMenuW(menu, _MF_SEPARATOR, 0, None)
        _user32.AppendMenuW(menu, _MF_STRING, _ID_QUIT, "Quit")

        point = _wt.POINT()
        _user32.GetCursorPos(ctypes.byref(point))
        _user32.SetForegroundWindow(self._hwnd)
        command = _user32.TrackPopupMenu(
            menu, _TPM_RIGHTBUTTON | _TPM_RETURNCMD,
            point.x, point.y, 0, self._hwnd, None,
        )
        _user32.DestroyMenu(menu)
        self._dispatch_menu_command(command)

    def _append_microphone_menu(self, menu: int) -> None:
        if self._list_microphones is None or self._on_microphone_chosen is None:
            return
        try:
            connected, chosen = self._list_microphones()
        except Exception:  # a broken device list must not lose the menu
            _logger.exception("Listing microphones failed.")
            return
        entries = [("", _SYSTEM_DEFAULT_LABEL), *connected]
        # A chosen microphone that is unplugged stays listed, so the menu
        # still shows what the user picked.
        if chosen and chosen not in dict(connected):
            entries.append((chosen, f"{chosen} (not connected)"))
        self._offered_microphones = [name for name, _label in entries]
        submenu = _user32.CreatePopupMenu()
        for offset, (name, label) in enumerate(entries):
            flags = _MF_STRING | (_MF_CHECKED if name == chosen else 0)
            _user32.AppendMenuW(submenu, flags, _ID_FIRST_MICROPHONE + offset, label)
        _user32.AppendMenuW(menu, _MF_POPUP, submenu, "Microphone")

    def _dispatch_menu_command(self, command: int) -> None:
        offset = command - _ID_FIRST_MICROPHONE
        if 0 <= offset < len(self._offered_microphones):
            if self._on_microphone_chosen is not None:
                self._on_microphone_chosen(self._offered_microphones[offset])
            return
        if command == _ID_SETTINGS and self._on_settings_clicked is not None:
            self._on_settings_clicked()
        elif command == _ID_HISTORY and self._on_history_clicked is not None:
            self._on_history_clicked()
        elif command == _ID_QUIT:
            self._on_quit_clicked()


class Win32TrayAdapter:
    """``Shell_NotifyIcon``-backed :class:`voicetotext.tray.app.TrayIcon`.

    Public mutators are called from worker threads (state transitions are
    published synchronously by the transcription worker), but every Win32
    call must run on the GUI thread. Each mutator marshals through the
    injected :class:`GuiThreadDispatcher` before touching the native
    backend, mirroring what Qt's queued signals did invisibly.
    """

    def __init__(
        self,
        *,
        initial_state: AppState,
        initial_label: str,
        get_state_label: Callable[[], str],
        on_settings_clicked: Callable[[], None] | None,
        on_quit_clicked: Callable[[], None],
        root: tkinter.Misc,
        dispatcher: GuiThreadDispatcher,
        backend: TrayBackend | None = None,
        on_history_clicked: Callable[[], None] | None = None,
        list_microphones: MicrophoneLister | None = None,
        on_microphone_chosen: Callable[[str], None] | None = None,
        on_devices_changed: Callable[[], None] | None = None,
    ) -> None:
        self._root = root
        self._dispatcher = dispatcher
        self._get_state_label = get_state_label
        self._visible = False
        self._title = _TITLE_PREFIX + initial_label
        self._hicon = 0
        self._elapsed_provider: Callable[[], float] | None = None
        self._elapsed_after_id: str | None = None
        self._backend: TrayBackend = backend or Win32TrayBackend(
            get_state_label, on_settings_clicked, on_quit_clicked,
            on_history_clicked, list_microphones, on_microphone_chosen,
            on_devices_changed,
        )
        self._apply_icon(initial_state)
        self._backend.set_tooltip(self._title)

    # ------------------------------------------------------------------
    # TrayIcon Protocol
    # ------------------------------------------------------------------

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value
        if value:
            self._backend.show()
        else:
            self._backend.hide()

    def update_state(self, state: AppState, label: str) -> None:
        self._dispatcher.dispatch(lambda: self._apply_state_update(state, label))

    def set_elapsed_provider(
        self, provider: Callable[[], float] | None,
    ) -> None:
        self._dispatcher.dispatch(lambda: self._apply_elapsed_provider(provider))

    def notify(self, title: str, message: str) -> None:
        self._dispatcher.dispatch(lambda: self._backend.notify(title, message))

    def run(self, setup: Callable[["Win32TrayAdapter"], None]) -> None:
        setup(self)
        self._root.mainloop()

    def stop(self) -> None:
        self._backend.destroy()
        self._destroy_icon()
        self._root.quit()

    def current_title(self) -> str:
        return self._title

    # ------------------------------------------------------------------
    # GUI-thread appliers
    # ------------------------------------------------------------------

    def _apply_state_update(self, state: AppState, label: str) -> None:
        self._apply_icon(state)
        self._title = _TITLE_PREFIX + label
        self._backend.set_tooltip(self._title)

    def _apply_icon(self, state: AppState) -> None:
        new_hicon = create_hicon(state)
        self._backend.set_icon(new_hicon)
        self._destroy_icon()
        self._hicon = new_hicon

    def _destroy_icon(self) -> None:
        if self._hicon:
            _user32.DestroyIcon(self._hicon)
            self._hicon = 0

    def _apply_elapsed_provider(
        self, provider: Callable[[], float] | None,
    ) -> None:
        self._stop_elapsed_timer()
        self._elapsed_provider = provider
        if provider is None:
            self._backend.set_tooltip(self._title)
            return
        self._refresh_elapsed_tooltip()
        self._elapsed_after_id = self._root.after(
            _ELAPSED_REFRESH_INTERVAL_MS, self._tick_elapsed,
        )

    def _tick_elapsed(self) -> None:
        self._refresh_elapsed_tooltip()
        if self._elapsed_provider is not None:
            self._elapsed_after_id = self._root.after(
                _ELAPSED_REFRESH_INTERVAL_MS, self._tick_elapsed,
            )

    def _stop_elapsed_timer(self) -> None:
        if self._elapsed_after_id is not None:
            self._root.after_cancel(self._elapsed_after_id)
            self._elapsed_after_id = None

    def _refresh_elapsed_tooltip(self) -> None:
        if self._elapsed_provider is None:
            return
        elapsed = self._elapsed_provider()
        self._backend.set_tooltip(
            f"{_TITLE_PREFIX}Recording {format_duration(elapsed)}"
        )


def build_win32_tray_adapter(
    *,
    initial_state: AppState,
    get_state_label: Callable[[], str],
    on_settings_clicked: Callable[[], None] | None,
    on_quit_clicked: Callable[[], None],
    root: tkinter.Misc,
    dispatcher: GuiThreadDispatcher,
    on_history_clicked: Callable[[], None] | None = None,
    list_microphones: MicrophoneLister | None = None,
    on_microphone_chosen: Callable[[str], None] | None = None,
    on_devices_changed: Callable[[], None] | None = None,
) -> Win32TrayAdapter:
    """Factory matching :data:`voicetotext.tray.app.TrayIconFactory`.

    ``root`` and ``dispatcher`` are bound in bootstrap via
    ``functools.partial`` so the call site matches the factory signature
    the app expects.
    """
    return Win32TrayAdapter(
        initial_state=initial_state,
        initial_label=get_state_label(),
        get_state_label=get_state_label,
        on_settings_clicked=on_settings_clicked,
        on_quit_clicked=on_quit_clicked,
        root=root,
        dispatcher=dispatcher,
        on_history_clicked=on_history_clicked,
        list_microphones=list_microphones,
        on_microphone_chosen=on_microphone_chosen,
        on_devices_changed=on_devices_changed,
    )
