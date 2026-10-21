"""Whether Windows makes the Copilot key act as Right Ctrl.

The keyboard sends the Copilot key as Win+Shift+F23. With this setting on,
Windows releases those three at once and holds Right Ctrl instead, so a
shortcut hook sees Right Ctrl while the key is down. The shortcut recorder
blocks the key, which stops Windows swapping it, so the recorder has to ask.
"""

# Settings > Personalization > Text input > Customize Copilot key on keyboard.
# Microsoft documents "Search" and "App"; this choice was read on a Dell XPS 13
# running Windows 11 build 26300.
_KEY_PATH = r"Software\Microsoft\Windows\Shell\BrandedKey"
_VALUE = "BrandedKeyChoiceType"
_RIGHT_CTRL_CHOICE = "RightControl"


def copilot_key_acts_as_right_ctrl() -> bool:
    try:
        import winreg
    except ImportError:
        return False
    return copilot_key_setting_is_right_ctrl(winreg)


def copilot_key_setting_is_right_ctrl(winreg: object) -> bool:
    try:
        return _read_choice(winreg) == _RIGHT_CTRL_CHOICE
    except OSError:
        return False


def _read_choice(winreg: object) -> object:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KEY_PATH) as key:
        choice, _ = winreg.QueryValueEx(key, _VALUE)
    return choice
