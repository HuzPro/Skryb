"""Behavioral tests for ``AutoHideScrollableFrame``.

Verifies the adapter wrapping ``ctk.CTkScrollableFrame`` hides its
scrollbar when content fits and shows it again when content overflows.
Follows the real-Tk testing pattern established in ``test_tk_cleanup.py``.
"""

import tkinter

import customtkinter as ctk
import pytest

from voicetotext.settings.auto_hide_scrollable_frame import (
    AutoHideScrollableFrame,
)


@pytest.fixture(scope="module")
def tk_root():
    """Provide a single withdrawn Tk root shared across this module.

    Creating and destroying multiple ``CTk()`` instances in the same
    process is flaky on Windows — Tcl's init files sometimes fail to
    reload between roots. Sharing one root and wiping its children
    between tests avoids the issue.
    """
    root = ctk.CTk()
    root.withdraw()
    try:
        yield root
    finally:
        try:
            root.destroy()
        except tkinter.TclError:
            pass


@pytest.fixture
def clean_root(tk_root):
    """Wipe child widgets before each test so the module-scoped root
    starts every test case empty."""
    for child in list(tk_root.winfo_children()):
        child.destroy()
    tk_root.update_idletasks()
    return tk_root


def _settle(root: tkinter.Tk) -> None:
    """Flush pending geometry work so scrollregion/yscrollcommand fire.

    ``update()`` (not just ``update_idletasks``) is required so the canvas
    actually realizes its size and fires the ``<Configure>`` that sets
    ``scrollregion`` — which in turn invokes our ``_autohide_set`` callback.
    """
    for _ in range(3):
        root.update()


def _scrollbar_is_gridded(scrollbar) -> bool:
    """True when the scrollbar is currently placed by the grid manager.

    ``winfo_manager()`` returns ``"grid"`` when a widget is visible via
    ``.grid()`` and ``""`` after ``.grid_remove()``. This works with a
    withdrawn root, unlike ``winfo_ismapped()`` which requires the
    ancestor toplevel to actually be on screen.
    """
    return scrollbar.winfo_manager() == "grid"


class TestAutoHideScrollableFrame:
    def test_should_hide_scrollbar_when_content_fits(self, clean_root):
        clean_root.geometry("400x500")
        frame = AutoHideScrollableFrame(clean_root, width=300, height=400)
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(frame, text="only child").pack()

        _settle(clean_root)

        assert _scrollbar_is_gridded(frame._scrollbar) is False

    def test_should_show_scrollbar_when_content_overflows(self, clean_root):
        clean_root.geometry("300x200")
        frame = AutoHideScrollableFrame(clean_root, width=300, height=200)
        frame.pack(fill="both", expand=True)
        for i in range(30):
            ctk.CTkLabel(frame, text=f"row {i}", height=40).pack(fill="x")

        _settle(clean_root)

        assert _scrollbar_is_gridded(frame._scrollbar) is True

    def test_should_toggle_scrollbar_when_content_shrinks_back_to_fit(
        self, clean_root
    ):
        clean_root.geometry("300x200")
        frame = AutoHideScrollableFrame(clean_root, width=300, height=200)
        frame.pack(fill="both", expand=True)
        rows = [
            ctk.CTkLabel(frame, text=f"row {i}", height=40)
            for i in range(30)
        ]
        for row in rows:
            row.pack(fill="x")
        _settle(clean_root)
        assert _scrollbar_is_gridded(frame._scrollbar) is True  # sanity

        for row in rows[1:]:
            row.destroy()
        _settle(clean_root)

        assert _scrollbar_is_gridded(frame._scrollbar) is False

    def test_should_preserve_original_grid_options_after_hide_show_cycle(
        self, clean_root
    ):
        clean_root.geometry("400x500")
        frame = AutoHideScrollableFrame(clean_root, width=300, height=400)
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(frame, text="only").pack()
        _settle(clean_root)
        assert _scrollbar_is_gridded(frame._scrollbar) is False  # sanity

        for i in range(30):
            ctk.CTkLabel(frame, text=f"row {i}", height=40).pack(fill="x")
        _settle(clean_root)

        # After re-showing, the scrollbar must land back in its original
        # slot (row=1, column=1 per CTkScrollableFrame._create_grid).
        assert _scrollbar_is_gridded(frame._scrollbar) is True
        info = frame._scrollbar.grid_info()
        assert int(info["row"]) == 1
        assert int(info["column"]) == 1
