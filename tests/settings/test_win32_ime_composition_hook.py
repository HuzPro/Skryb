"""Tk-side checks for the IME composition hook; the Win32 half is manual."""

import tkinter
from collections.abc import Iterator
from tkinter import ttk

import pytest

from voicetotext.settings.win32_ime_composition_hook import (
    contains_non_bmp_characters,
    insert_as_typed,
)

GRINNING_FACE = "\U0001F600"


@pytest.fixture()
def window(tk_root: tkinter.Tk) -> Iterator[tkinter.Toplevel]:
    top = tkinter.Toplevel(tk_root)
    yield top
    top.destroy()


def test_should_insert_at_caret_replacing_selection(
    window: tkinter.Toplevel,
) -> None:
    entry = ttk.Entry(window)
    entry.insert(0, "hello")
    entry.selection_range(1, 4)
    entry.icursor(4)

    insert_as_typed(entry, GRINNING_FACE)

    assert entry.get() == f"h{GRINNING_FACE}o"
    assert entry.index("insert") == entry.index("end") - 1


def test_should_insert_at_caret_without_a_selection(
    window: tkinter.Toplevel,
) -> None:
    entry = ttk.Entry(window)
    entry.insert(0, "helo")
    entry.icursor(3)

    insert_as_typed(entry, GRINNING_FACE)

    assert entry.get() == f"hel{GRINNING_FACE}o"
    assert entry.index("insert") == entry.index("end") - 1


def test_should_leave_readonly_entries_untouched(
    window: tkinter.Toplevel,
) -> None:
    combo = ttk.Combobox(window, values=["a", "b"], state="readonly")
    combo.set("a")

    insert_as_typed(combo, GRINNING_FACE)

    assert combo.get() == "a"


def test_should_detect_characters_outside_the_bmp() -> None:
    assert contains_non_bmp_characters(f"ok {GRINNING_FACE}")
    assert not contains_non_bmp_characters("plain \u00e9 text")
