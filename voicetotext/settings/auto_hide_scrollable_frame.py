"""Auto-hiding scrollable frame adapter.

Wraps :class:`customtkinter.CTkScrollableFrame` so its internal scrollbar
disappears when the content fits inside the viewport and reappears when
the content overflows — the standard Tk auto-hide scrollbar pattern.
"""

import customtkinter as ctk


class AutoHideScrollableFrame(ctk.CTkScrollableFrame):
    """``CTkScrollableFrame`` whose scrollbar hides when content fits.

    Adapter around the CustomTkinter widget: rewires the inner canvas's
    ``yscrollcommand`` so the scrollbar is ``grid_remove``'d when the
    visible range covers the full scrollregion (``first <= 0`` and
    ``last >= 1``) and ``grid``'d again when content overflows.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        if hasattr(self, "_parent_canvas") and hasattr(self, "_scrollbar"):
            self._parent_canvas.configure(yscrollcommand=self._autohide_set)

    def _autohide_set(self, first, last) -> None:
        if float(first) <= 0.0 and float(last) >= 1.0:
            self._scrollbar.grid_remove()
        else:
            self._scrollbar.grid()
        self._scrollbar.set(first, last)
