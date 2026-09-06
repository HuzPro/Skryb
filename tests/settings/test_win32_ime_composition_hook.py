"""Checks for the IME composition hook; the emoji panel itself stays manual."""

import ctypes
import sys
import tkinter
from collections.abc import Iterator
from tkinter import ttk

import pytest

from voicetotext.settings.win32_ime_composition_hook import (
    Win32ImeCompositionHook,
    contains_non_bmp_characters,
    unichar_reposts,
)

GRINNING_FACE = "\U0001F600"
BEAMING_FACE_CODE_POINT = 0x1F601
WM_UNICHAR = 0x0109
WM_NULL = 0x0000

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="the hook subclasses a Win32 window proc",
)


@pytest.fixture()
def window(tk_root: tkinter.Tk) -> Iterator[tkinter.Toplevel]:
    top = tkinter.Toplevel(tk_root)
    top.withdraw()
    yield top
    top.destroy()


def test_should_detect_characters_outside_the_bmp() -> None:
    assert contains_non_bmp_characters(f"ok {GRINNING_FACE}")
    assert not contains_non_bmp_characters("plain é text")


def test_should_repost_every_character_of_a_result_containing_an_emoji() -> None:
    assert unichar_reposts(f"a{GRINNING_FACE}") == (ord("a"), ord(GRINNING_FACE))


def test_should_repost_nothing_when_tk_can_deliver_the_result_itself() -> None:
    assert unichar_reposts("plain é text") == ()


@windows_only
def test_should_deliver_a_reposted_unichar_as_a_whole_code_point_keypress(
    window: tkinter.Toplevel,
) -> None:
    ttk.Entry(window).pack()
    window.update()
    delivered: list[int] = []
    window.bind("<Key>", lambda event: delivered.append(ord(event.char)))

    ctypes.windll.user32.PostMessageW(
        ctypes.c_void_p(window.winfo_id()),
        WM_UNICHAR,
        ctypes.c_size_t(BEAMING_FACE_CODE_POINT),
        ctypes.c_ssize_t(0),
    )
    for _ in range(5):
        window.update()

    assert delivered == [BEAMING_FACE_CODE_POINT]


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
