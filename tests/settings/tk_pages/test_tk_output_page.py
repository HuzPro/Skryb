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

    def test_should_display_label_text_for_each_mode(self, page):
        labels = [btn.cget("text") for btn in page.radio_buttons()]

        for mode in OutputMode:
            assert mode.display_label in labels


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
