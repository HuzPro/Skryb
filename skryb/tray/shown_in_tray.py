"""Keeps the tray icon on the taskbar when the program Skryb runs as changes.

Windows 11 remembers, for each program file, whether its tray icon sits on the taskbar or in
the overflow menu (``HKCU\\Control Panel\\NotifyIconSettings``, one key per icon). An installed
Skryb used to run as ``pythonw.exe`` and now runs as ``Skryb.exe`` beside it. Windows takes that
for a new program, so an icon the user had dragged onto the taskbar would drop back into the
overflow menu.

This copies "shown" from the interpreter's icon to the same icon of the program now running.
It never touches an icon the user has already placed either way, and Explorer acts on the change
at once. Windows 10 keeps this elsewhere, in a form that can't be edited; there the icon has to be
dragged out once more.
"""

import logging
import ntpath
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

_SETTINGS_KEY = r"Control Panel\NotifyIconSettings"
_SHOWN = 1
_LONGEST_PATH = 32768


@dataclass(frozen=True)
class TrayIconSetting:
    """What Windows remembers about one tray icon."""

    key: str
    executable: str
    uid: int | None
    promoted: int | None
    """1 on the taskbar, 0 in the overflow menu, None while the user has never moved it."""


def icons_to_show(settings: Iterable[TrayIconSetting], interpreter: str, program: str) -> list[str]:
    """The keys of *program*'s icons that take over "shown" from the same icon of *interpreter*."""
    settings = list(settings)
    shown_before = {
        setting.uid for setting in settings
        if _same_file(setting.executable, interpreter) and setting.promoted == _SHOWN
    }
    if _same_file(interpreter, program):
        return []
    return [
        setting.key for setting in settings
        if _same_file(setting.executable, program) and setting.promoted is None and setting.uid in shown_before
    ]


def keep_tray_icon_shown(
    interpreter: str | None = None,
    program: str | None = None,
    read: Callable[[], list[TrayIconSetting]] | None = None,
    show: Callable[[str], None] | None = None,
) -> None:
    """Carry the user's "shown" over to the program this process runs as. Never raises."""
    try:
        _carry_shown_over(interpreter, program, read, show)
    except Exception:  # The icon's place is cosmetic: it must never get in the way of dictation.
        _logger.warning("Could not keep the tray icon's place on the taskbar.", exc_info=True)


def keep_tray_icon_shown_in_background() -> None:
    threading.Thread(target=keep_tray_icon_shown, daemon=True, name="tray-icon-place").start()


def _carry_shown_over(
    interpreter: str | None,
    program: str | None,
    read: Callable[[], list[TrayIconSetting]] | None,
    show: Callable[[str], None] | None,
) -> None:
    if sys.platform != "win32" and (read is None or show is None):
        return
    interpreter = interpreter or sys.executable
    program = program or _this_program()
    for key in icons_to_show((read or _read_settings)(), interpreter, program):
        (show or _show)(key)
        _logger.info("Kept the tray icon on the taskbar, where it was shown as %s.", ntpath.basename(interpreter))


def _same_file(one: str, other: str) -> bool:
    return bool(one) and ntpath.normcase(ntpath.normpath(one)) == ntpath.normcase(ntpath.normpath(other))


def _this_program() -> str:
    import ctypes

    path = ctypes.create_unicode_buffer(_LONGEST_PATH)
    ctypes.windll.kernel32.GetModuleFileNameW(None, path, _LONGEST_PATH)
    return path.value


def _read_settings() -> list[TrayIconSetting]:
    import winreg

    settings = []
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _SETTINGS_KEY) as icons:
        for index in range(winreg.QueryInfoKey(icons)[0]):
            key = winreg.EnumKey(icons, index)
            with winreg.OpenKey(icons, key) as icon:
                executable = _value(icon, "ExecutablePath")
                settings.append(TrayIconSetting(
                    key=key,
                    executable=executable if isinstance(executable, str) else "",
                    uid=_whole_number(_value(icon, "UID")),
                    promoted=_whole_number(_value(icon, "IsPromoted")),
                ))
    return settings


def _show(key: str) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"{_SETTINGS_KEY}\{key}", 0, winreg.KEY_SET_VALUE) as icon:
        winreg.SetValueEx(icon, "IsPromoted", 0, winreg.REG_DWORD, _SHOWN)


def _value(icon: object, name: str) -> object:
    import winreg

    try:
        return winreg.QueryValueEx(icon, name)[0]  # type: ignore[arg-type]
    except FileNotFoundError:
        return None


def _whole_number(value: object) -> int | None:
    return value if isinstance(value, int) else None
