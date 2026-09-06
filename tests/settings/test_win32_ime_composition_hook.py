"""``insert_as_typed`` is the Tk half of the IME composition hook; the Win32
window-procedure half is checked by the manual Windows emoji-panel smoke."""

import tkinter
from collections.abc import Iterator
from tkinter import ttk

import pytest

from voicetotext.settings.win32_ime_composition_hook import (
    insert_as_typed,
    needs_surrogate_pairs,
)

GRINNING_FACE = "\U0001F600"


@pytest.fixture()
def window(tk_root: tkinter.Tk) -> Iterator[tkinter.Toplevel]:
    top = tkinter.Toplevel(tk_root)
    yield top
    top.destroy()


def test_should_insert_at_caret_replacing_selection(window: tkinter.Toplevel):
    entry = ttk.Entry(window)
    entry.insert(0, "hello")
    entry.selection_range(1, 4)
    entry.icursor(4)

    insert_as_typed(entry, GRINNING_FACE)

    assert entry.get() == f"h{GRINNING_FACE}o"
    assert entry.index("insert") == entry.index("end") - 1


def test_should_leave_readonly_entries_untouched(window: tkinter.Toplevel):
    combo = ttk.Combobox(window, values=["a", "b"], state="readonly")
    combo.set("a")

    insert_as_typed(combo, GRINNING_FACE)

    assert combo.get() == "a"


def test_should_detect_characters_outside_the_bmp():
    assert needs_surrogate_pairs(f"ok {GRINNING_FACE}")
    assert not needs_surrogate_pairs("plain \u00e9 text")
