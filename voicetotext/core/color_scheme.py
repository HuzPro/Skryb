from enum import Enum


class ColorScheme(Enum):
    LIGHT = "light"
    DARK = "dark"


# Windows keeps two switches: the taskbar and tray follow the system one,
# app windows follow the apps one, and a user can set them differently.
_SYSTEM_VALUE = "SystemUsesLightTheme"
_APPS_VALUE = "AppsUseLightTheme"


def detect_color_scheme() -> ColorScheme:
    """Read the Windows taskbar light/dark setting, defaulting to light.

    Returns :attr:`ColorScheme.LIGHT` when the registry value is missing
    or ``winreg`` is unavailable (non-Windows), so callers can run
    cross-platform without guarding the import themselves.
    """
    return _detect(_SYSTEM_VALUE)


def detect_app_color_scheme() -> ColorScheme:
    """The light/dark setting Windows applies to app windows."""
    return _detect(_APPS_VALUE)


def _detect(value_name: str) -> ColorScheme:
    try:
        import winreg
    except ImportError:
        return ColorScheme.LIGHT
    return color_scheme_from_registry(winreg, value_name)


def color_scheme_from_registry(
    winreg: object, value_name: str = _SYSTEM_VALUE,
) -> ColorScheme:
    try:
        return _read_color_scheme(winreg, value_name)
    except OSError:
        return ColorScheme.LIGHT


def _read_color_scheme(winreg: object, value_name: str) -> ColorScheme:
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        uses_light, _ = winreg.QueryValueEx(key, value_name)
    return ColorScheme.LIGHT if uses_light else ColorScheme.DARK
