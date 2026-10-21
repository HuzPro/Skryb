import contextlib
import sys

import pytest

from voicetotext.hotkeys.win32_copilot_key import (
    copilot_key_acts_as_right_ctrl,
    copilot_key_setting_is_right_ctrl,
)

_KEY_PATH = r"Software\Microsoft\Windows\Shell\BrandedKey"


class FakeWinreg:
    """The one registry key read, holding ``values``, or missing when None."""

    HKEY_CURRENT_USER = object()

    def __init__(self, values: dict[str, object] | None) -> None:
        self._values = values

    def OpenKey(self, root: object, path: str):  # noqa: N802 - winreg's name
        assert root is self.HKEY_CURRENT_USER and path == _KEY_PATH
        if self._values is None:
            raise FileNotFoundError(path)
        return contextlib.nullcontext(self._values)

    def QueryValueEx(self, key: dict[str, object], name: str):  # noqa: N802 - winreg's name
        if name not in key:
            raise FileNotFoundError(name)
        return key[name], 1


class TestCopilotKeySetting:
    def test_the_right_control_choice_means_the_key_acts_as_right_ctrl(self):
        winreg = FakeWinreg({
            "BrandedKeyChoiceType": "RightControl",
            "AppAumid": "Microsoft.Copilot_8wekyb3d8bbwe!App",
        })

        assert copilot_key_setting_is_right_ctrl(winreg) is True

    @pytest.mark.parametrize("choice", ["Search", "App", "AppEnforcedByPolicy", "", 1, "rightcontrol"])
    def test_every_other_choice_leaves_the_key_as_win_shift_f23(self, choice):
        assert copilot_key_setting_is_right_ctrl(FakeWinreg({"BrandedKeyChoiceType": choice})) is False

    def test_a_missing_choice_or_key_means_no_swap(self):
        assert copilot_key_setting_is_right_ctrl(FakeWinreg({"AppAumid": "x"})) is False
        assert copilot_key_setting_is_right_ctrl(FakeWinreg(None)) is False

    def test_without_a_windows_registry_there_is_no_swap(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "winreg", None)

        assert copilot_key_acts_as_right_ctrl() is False
