import pytest

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.tk_pages.output_page import TkOutputPage


@pytest.fixture
def page(tk_root):
    widget = TkOutputPage(tk_root)
    yield widget
    widget.destroy()


class TestTkOutputPageInitial:
    def test_should_default_to_direct_paste(self, page):
        assert page.current_mode() is OutputMode.DIRECT_PASTE


class TestTkOutputPageSetMode:
    def test_should_reflect_paste_and_clipboard_when_set(self, page):
        page.set_mode(OutputMode.PASTE_AND_CLIPBOARD)

        assert page.current_mode() is OutputMode.PASTE_AND_CLIPBOARD

    def test_should_reflect_clipboard_only_when_set(self, page):
        page.set_mode(OutputMode.CLIPBOARD_ONLY)

        assert page.current_mode() is OutputMode.CLIPBOARD_ONLY

    def test_should_round_trip_direct_paste(self, page):
        page.set_mode(OutputMode.CLIPBOARD_ONLY)
        page.set_mode(OutputMode.DIRECT_PASTE)

        assert page.current_mode() is OutputMode.DIRECT_PASTE


class TestTkOutputPageButtons:
    def test_should_render_one_radio_per_output_mode(self, page):
        assert len(page.radio_buttons()) == len(list(OutputMode))

    def test_each_mode_has_its_own_card_title(self, page):
        labels = [btn.cget("text") for btn in page.radio_buttons()]

        assert labels == ["Paste only", "Paste and copy", "Clipboard only"]

    def test_clicking_a_card_chooses_its_mode(self, page):
        page.radio_buttons()[2].invoke()

        assert page.current_mode() is OutputMode.CLIPBOARD_ONLY


class TestResetPillPosition:
    def test_the_button_resets_the_pill_and_says_where_it_will_appear(self, tk_root):
        resets: list[bool] = []
        widget = TkOutputPage(tk_root, on_reset_pill_position=lambda: resets.append(True))
        try:
            widget._reset_pill_button.invoke()

            assert resets == [True]
            assert "bottom centre" in widget._pill_status.cget("text")
        finally:
            widget.destroy()


class TestTkOutputPagePasteMethods:
    def test_adding_an_app_stores_its_exe_name_and_method(self, page):
        page._new_process.set("  Mintty ")
        page._new_method.set("Ctrl+Shift+V")
        page._add_override()
        page._new_process.set("vmconnect.exe")
        page._new_method.set("Type the text")
        page._add_override()
        page._new_process.set("   ")
        page._add_override()

        assert page.paste_method_overrides() == (
            ("mintty.exe", "ctrl_shift_v"), ("vmconnect.exe", "type"),
        )

    def test_adding_the_same_app_again_replaces_its_method(self, page):
        page.set_paste_method_overrides((("mintty.exe", "ctrl_v"),))
        page._new_process.set("MINTTY.EXE")
        page._new_method.set("Shift+Insert")
        page._add_override()

        assert page.paste_method_overrides() == (("mintty.exe", "shift_insert"),)

    def test_removing_the_selected_app_drops_only_that_one(self, page):
        page.set_paste_method_overrides(
            (("code.exe", "ctrl_shift_v"), ("mintty.exe", "shift_insert"), ("x.exe", "junk")),
        )
        page._paste_table.selection_set("code.exe")
        page._remove_override()

        assert page.paste_method_overrides() == (("mintty.exe", "shift_insert"),)
