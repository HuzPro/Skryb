from enum import Enum


class ColorScheme(Enum):
    LIGHT = "light"
    DARK = "dark"


def detect_color_scheme() -> ColorScheme:
    """Read the Windows light/dark setting, defaulting to light.

    Returns :attr:`ColorScheme.LIGHT` when the registry value is missing
    or ``winreg`` is unavailable (non-Windows), so callers can run
    cross-platform without guarding the import themselves.
    """
    try:
        import winreg
    except ImportError:
        return ColorScheme.LIGHT
    return color_scheme_from_registry(winreg)


def color_scheme_from_registry(winreg: object) -> ColorScheme:
    try:
        return _read_color_scheme(winreg)
    except OSError:
        return ColorScheme.LIGHT


def _read_color_scheme(winreg: object) -> ColorScheme:
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        uses_light, _ = winreg.QueryValueEx(key, "SystemUsesLightTheme")
    return ColorScheme.LIGHT if uses_light else ColorScheme.DARK
