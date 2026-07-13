import tkinter
from tkinter import ttk

import pytest

from voicetotext.core.color_scheme import ColorScheme
from voicetotext.settings.theme import (
    DARK_THEME,
    LIGHT_THEME,
    apply_theme,
    theme_for,
)


@pytest.fixture
def container(tk_root):
    frame = tkinter.Frame(tk_root)
    yield frame
    frame.destroy()


class TestThemeFor:
    def test_should_return_dark_theme_for_dark_scheme(self):
        assert theme_for(ColorScheme.DARK) is DARK_THEME

    def test_should_return_light_theme_for_light_scheme(self):
        assert theme_for(ColorScheme.LIGHT) is LIGHT_THEME

    def test_dark_and_light_backgrounds_should_differ(self):
        assert DARK_THEME.background != LIGHT_THEME.background


class TestApplyTheme:
    def test_should_switch_ttk_to_the_recolourable_clam_theme(self, container):
        apply_theme(container, DARK_THEME)

        assert ttk.Style(container).theme_use() == "clam"

    def test_should_recolour_ttk_frame_background(self, container):
        apply_theme(container, DARK_THEME)

        assert ttk.Style(container).lookup("TFrame", "background") == (
            DARK_THEME.background
        )

    def test_should_recolour_ttk_label_foreground(self, container):
        apply_theme(container, DARK_THEME)

        assert ttk.Style(container).lookup("TLabel", "foreground") == (
            DARK_THEME.foreground
        )

    def test_should_recolour_combobox_field(self, container):
        apply_theme(container, DARK_THEME)

        assert ttk.Style(container).lookup("TCombobox", "fieldbackground") == (
            DARK_THEME.surface
        )

    def test_should_set_container_background(self, container):
        apply_theme(container, DARK_THEME)

        assert container.cget("bg") == DARK_THEME.background

    def test_should_recolour_descendant_listbox(self, container):
        listbox = tkinter.Listbox(container)

        apply_theme(container, DARK_THEME)

        assert listbox.cget("bg") == DARK_THEME.background
        assert listbox.cget("fg") == DARK_THEME.foreground

    def test_should_define_filled_selection_style(self, container):
        apply_theme(container, DARK_THEME)

        assert ttk.Style(container).lookup("Selected.TFrame", "background") == (
            DARK_THEME.selection_background
        )

    def test_should_hide_the_dotted_focus_ring(self, container):
        apply_theme(container, DARK_THEME)

        assert ttk.Style(container).lookup(".", "focuscolor") == (
            DARK_THEME.background
        )

    def test_should_colour_the_scrollbar_thumb(self, container):
        apply_theme(container, DARK_THEME)

        assert ttk.Style(container).lookup("Vertical.TScrollbar", "background") == (
            DARK_THEME.scrollbar_thumb
        )

    def test_should_recolour_descendant_canvas(self, container):
        canvas = tkinter.Canvas(container)

        apply_theme(container, DARK_THEME)

        assert canvas.cget("bg") == DARK_THEME.background

    def test_should_apply_light_palette_for_light_theme(self, container):
        apply_theme(container, LIGHT_THEME)

        assert container.cget("bg") == LIGHT_THEME.background
        assert ttk.Style(container).lookup("TLabel", "foreground") == (
            LIGHT_THEME.foreground
        )
