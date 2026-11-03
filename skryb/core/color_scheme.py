from enum import Enum


class ColorScheme(Enum):
    LIGHT = "light"
    DARK = "dark"


# The taskbar and tray follow the system switch, not the apps one.
_SYSTEM_VALUE = "SystemUsesLightTheme"
# Windows like Settings and the pill follow the apps one.
_APPS_VALUE = "AppsUseLightTheme"
# Settings > Personalisation > Colours > Transparency effects.
_TRANSPARENCY_VALUE = "EnableTransparency"
_PERSONALIZE_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


def detect_color_scheme() -> ColorScheme:
    """Read the Windows taskbar light/dark setting, defaulting to light.

    Returns :attr:`ColorScheme.LIGHT` when the registry value is missing
    or ``winreg`` is unavailable (non-Windows), so callers can run
    cross-platform without guarding the import themselves.
    """
    return _detect(_SYSTEM_VALUE)


def detect_app_color_scheme() -> ColorScheme:
    """Read the Windows apps light/dark setting, defaulting to light the same way."""
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
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _PERSONALIZE_KEY) as key:
        uses_light, _ = winreg.QueryValueEx(key, value_name)
    return ColorScheme.LIGHT if uses_light else ColorScheme.DARK


def transparency_effects_on() -> bool:
    """Whether Windows' Transparency effects switch is on; on, as Windows ships it, when it can't be read."""
    try:
        import winreg
    except ImportError:
        return True
    return transparency_from_registry(winreg)


def transparency_from_registry(winreg: object) -> bool:
    try:
        return _read_transparency(winreg)
    except OSError:
        return True


def _read_transparency(winreg: object) -> bool:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _PERSONALIZE_KEY) as key:
        enabled, _ = winreg.QueryValueEx(key, _TRANSPARENCY_VALUE)
    return bool(enabled)
