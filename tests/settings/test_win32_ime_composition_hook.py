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
