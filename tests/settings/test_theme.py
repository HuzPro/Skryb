import tkinter
from tkinter import ttk

import pytest

from voicetotext.core.color_scheme import ColorScheme
from voicetotext.settings.theme import (
    DARK_THEME,
    LIGHT_THEME,
    _scroll_region_height,
    _should_show_scrollbar,
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

    def test_hovered_checkbutton_should_keep_the_theme_background(self, container):
        apply_theme(container, DARK_THEME)

        hovered = ttk.Style(container).lookup(
            "TCheckbutton", "background", state=["active"],
        )
        assert hovered == DARK_THEME.background

    def test_hovered_radiobutton_should_keep_the_theme_background(self, container):
        apply_theme(container, DARK_THEME)

        hovered = ttk.Style(container).lookup(
            "TRadiobutton", "background", state=["active"],
        )
        assert hovered == DARK_THEME.background

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

    def test_a_theme_change_repaints_the_switches_already_on_screen(self, container):
        apply_theme(container, LIGHT_THEME)
        light_track = _pixel(container, "vtt_switch_on", 6)
        apply_theme(container, DARK_THEME)
        dark_track = _pixel(container, "vtt_switch_on", 6)

        assert light_track == LIGHT_THEME.accent
        assert dark_track == DARK_THEME.accent

    def test_a_switch_that_is_off_is_not_drawn_in_the_on_colour(self, container):
        apply_theme(container, DARK_THEME)

        assert _pixel(container, "vtt_switch_off", 28) == DARK_THEME.strong_border
        assert _pixel(container, "vtt_check_off", 8) == DARK_THEME.surface

    @pytest.mark.parametrize("theme", [LIGHT_THEME, DARK_THEME], ids=["light", "dark"])
    def test_a_chosen_radio_button_shows_a_dot_inside_its_ring(self, container, theme):
        apply_theme(container, theme)

        assert "Radio.indicator" in str(ttk.Style(container).layout("TRadiobutton"))
        assert _pixel(container, "vtt_radio_on", 8) == theme.accent
        assert _pixel(container, "vtt_radio_on", 3) != theme.accent
        assert _pixel(container, "vtt_radio_off", 8) == theme.surface


class TestScrollArea:
    def test_shows_the_scrollbar_only_when_the_page_overflows(self):
        assert _should_show_scrollbar(200, 100) is True
        assert _should_show_scrollbar(100, 100) is False
        assert _should_show_scrollbar(80, 100) is False

    def test_a_short_page_cannot_be_scrolled_into_empty_space(self):
        assert _scroll_region_height(80, 100) == 100
        assert _scroll_region_height(200, 100) == 200


def _pixel(widget, image: str, x: int) -> str:
    """The colour of *image* at column *x*, half way down, as ``#rrggbb``."""
    height = int(widget.tk.call("image", "height", image))
    value = widget.tk.call(image, "get", x, height // 2)
    channels = value.split() if isinstance(value, str) else value
    red, green, blue = (int(channel) for channel in channels[:3])
    return f"#{red:02x}{green:02x}{blue:02x}"
