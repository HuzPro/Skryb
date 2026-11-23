"""Keeping the tray icon on the taskbar when Skryb goes from running as pythonw.exe to Skryb.exe.

Windows files the choice under the program file, so the tests work on what it remembers: a list
of icons, each with its program, its number within that program and where the user put it. The
real registry is never read or written here.
"""

import random

import pytest

from skryb.tray.shown_in_tray import TrayIconSetting, icons_to_show, keep_tray_icon_shown

_APP = r"C:\Users\Zoë O'Hara\AppData\Local\Programs\Skryb\python"
_PYTHONW = _APP + r"\pythonw.exe"
_SKRYB = _APP + r"\Skryb.exe"
_SHOWN, _HIDDEN, _NEVER_MOVED = 1, 0, None


def _icon(key: str, executable: str, promoted: int | None, uid: int | None = 1) -> TrayIconSetting:
    return TrayIconSetting(key=key, executable=executable, uid=uid, promoted=promoted)


def _other_programs(count: int = 2500) -> list[TrayIconSetting]:
    """What a long-used PC has: thousands of icons, repeated programs, blank and missing values."""
    chooser = random.Random(7)
    programs = [
        r"C:\Windows\explorer.exe", r"C:\Program Files\Mozilla Firefox\firefox.exe", "",
        r"C:\Users\Zoë O'Hara\AppData\Local\Programs\Другое\python\pythonw.exe",
        r"D:\Tools\python\pythonw.exe", r"C:\Program Files\Skryb Tools\Skryb.exe",
    ]
    return [
        _icon(str(10_000_000_000 + index), chooser.choice(programs), chooser.choice([_SHOWN, _HIDDEN, _NEVER_MOVED]),
              chooser.choice([None, 0, 1, 2, 4_294_967_295]))
        for index in range(count)
    ]


class TestWhichIconsTakeOverShown:
    def test_an_icon_shown_as_pythonw_is_shown_as_skryb_exe(self):
        settings = [_icon("651", _PYTHONW, _SHOWN), _icon("172", _SKRYB, _NEVER_MOVED)]

        assert icons_to_show(settings, _PYTHONW, _SKRYB) == ["172"]

    @pytest.mark.parametrize("before", [_HIDDEN, _NEVER_MOVED])
    def test_an_icon_left_in_the_overflow_menu_stays_there(self, before):
        settings = [_icon("651", _PYTHONW, before), _icon("172", _SKRYB, _NEVER_MOVED)]

        assert icons_to_show(settings, _PYTHONW, _SKRYB) == []

    @pytest.mark.parametrize("already", [_HIDDEN, _SHOWN])
    def test_a_place_the_user_already_chose_for_skryb_exe_is_never_changed(self, already):
        settings = [_icon("651", _PYTHONW, _SHOWN), _icon("172", _SKRYB, already)]

        assert icons_to_show(settings, _PYTHONW, _SKRYB) == []

    def test_before_windows_has_seen_skryb_exes_icon_there_is_nothing_to_change(self):
        assert icons_to_show([_icon("651", _PYTHONW, _SHOWN)], _PYTHONW, _SKRYB) == []
        assert icons_to_show([], _PYTHONW, _SKRYB) == []

    def test_only_the_same_icon_of_the_program_takes_over(self):
        settings = [
            _icon("651", _PYTHONW, _SHOWN, uid=1),
            _icon("652", _PYTHONW, _HIDDEN, uid=2),
            _icon("172", _SKRYB, _NEVER_MOVED, uid=1),
            _icon("173", _SKRYB, _NEVER_MOVED, uid=2),
            _icon("174", _SKRYB, _NEVER_MOVED, uid=None),
        ]

        assert icons_to_show(settings, _PYTHONW, _SKRYB) == ["172"]

    def test_windows_spelling_the_paths_differently_is_the_same_program(self):
        settings = [
            _icon("651", _PYTHONW.upper(), _SHOWN),
            _icon("172", _SKRYB.replace("\\", "/").replace("Skryb.exe", "SKRYB.EXE"), _NEVER_MOVED),
        ]

        assert icons_to_show(settings, _PYTHONW, _SKRYB) == ["172"]

    def test_other_programs_and_other_pythons_change_nothing(self):
        settings = _other_programs()

        assert icons_to_show(settings, _PYTHONW, _SKRYB) == []

    def test_skrybs_icons_are_found_among_thousands(self):
        settings = _other_programs()
        settings.insert(900, _icon("651", _PYTHONW, _SHOWN))
        settings.insert(1700, _icon("172", _SKRYB, _NEVER_MOVED))
        settings.append(_icon("172-again", _SKRYB, _NEVER_MOVED))

        assert icons_to_show(settings, _PYTHONW, _SKRYB) == ["172", "172-again"]

    def test_a_skryb_started_through_python_itself_has_nothing_to_carry_over(self):
        # A source checkout, or `pythonw.exe -m skryb` by hand: the program is the interpreter.
        settings = [_icon("651", _PYTHONW, _SHOWN), _icon("652", _PYTHONW, _NEVER_MOVED)]

        assert icons_to_show(settings, _PYTHONW, _PYTHONW) == []


class TestKeepingTheIconShown:
    def test_it_shows_each_icon_that_takes_over(self):
        shown: list[str] = []
        settings = [_icon("651", _PYTHONW, _SHOWN), _icon("172", _SKRYB, _NEVER_MOVED), *_other_programs(40)]

        keep_tray_icon_shown(_PYTHONW, _SKRYB, read=lambda: settings, show=shown.append)

        assert shown == ["172"]

    def test_a_registry_that_cannot_be_read_or_written_never_stops_skryb(self):
        def unreadable() -> list[TrayIconSetting]:
            raise PermissionError(5, "Access is denied")

        def unwritable(_key: str) -> None:
            raise OSError(1018, "Illegal operation attempted on a registry key that has been marked for deletion")

        keep_tray_icon_shown(_PYTHONW, _SKRYB, read=unreadable, show=lambda _key: None)
        keep_tray_icon_shown(
            _PYTHONW, _SKRYB,
            read=lambda: [_icon("651", _PYTHONW, _SHOWN), _icon("172", _SKRYB, _NEVER_MOVED)], show=unwritable,
        )
