"""Checks for the IME composition hook; the emoji panel itself stays manual."""

import ctypes
import sys
import tkinter

import pytest

from voicetotext.settings.win32_ime_composition_hook import (
    Win32ImeCompositionHook,
    contains_non_bmp_characters,
    unichar_reposts,
)

GRINNING_FACE = "\U0001F600"
WM_NULL = 0x0000
WM_KEYDOWN = 0x0100
WM_IME_COMPOSITION = 0x010F
GCS_RESULTSTR = 0x0800
VK_PACKET = 0xE7

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="the hook subclasses a Win32 window proc",
)


def test_should_detect_characters_outside_the_bmp() -> None:
    assert contains_non_bmp_characters(f"ok {GRINNING_FACE}")
    assert not contains_non_bmp_characters("plain é text")


def test_should_repost_every_character_of_a_result_containing_an_emoji() -> None:
    assert unichar_reposts(f"a{GRINNING_FACE}") == (ord("a"), ord(GRINNING_FACE))


def test_should_repost_nothing_when_tk_can_deliver_the_result_itself() -> None:
    assert unichar_reposts("plain é text") == ()


@windows_only
def test_should_chain_unrelated_messages_and_restore_the_proc_on_destroy(
    tk_root: tkinter.Tk,
) -> None:
    top = tkinter.Toplevel(tk_root)
    top.withdraw()
    top.update()
    Win32ImeCompositionHook(top)

    chained = ctypes.windll.user32.SendMessageW(
        ctypes.c_void_p(top.winfo_id()), WM_NULL, 0, 0,
    )

    assert chained == 0
    top.destroy()
    tk_root.update()


@pytest.fixture()
def hooked_window(tk_root: tkinter.Tk):
    top = tkinter.Toplevel(tk_root)
    top.withdraw()
    top.update()
    hook = Win32ImeCompositionHook(top)
    yield top, hook
    top.destroy()
    tk_root.update()


@windows_only
@pytest.mark.parametrize(("message", "w_param", "l_param"), [
    (WM_IME_COMPOSITION, 0, 0),
    (WM_IME_COMPOSITION, 0, GCS_RESULTSTR),
    (WM_KEYDOWN, VK_PACKET, 1),
    (WM_NULL, 0, 0),
])
def test_odd_messages_go_through_the_real_proc_to_tk_without_raising(
    hooked_window, monkeypatch, message, w_param, l_param,
) -> None:
    top, hook = hooked_window
    reported: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", reported.append)

    result = hook._proc(top.winfo_id(), message, w_param, l_param)

    assert isinstance(result, int)
    assert reported == []


@windows_only
def test_a_failure_while_chaining_falls_back_to_windows_and_is_logged_once(
    hooked_window, monkeypatch, app_caplog,
) -> None:
    top, hook = hooked_window
    reported: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", reported.append)

    def broken_chain(*_args: object) -> int:
        raise ctypes.ArgumentError("argument 2: wrong type")

    monkeypatch.setattr(hook, "_call_original", broken_chain)

    results = [hook._proc(top.winfo_id(), WM_NULL, 0, 0) for _ in range(50)]

    assert results == [0] * 50
    failures = [r for r in app_caplog.records if "IME window hook failed" in r.getMessage()]
    assert len(failures) == 1
    assert reported == []
