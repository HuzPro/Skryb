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


class TestTkOutputPageHistory:
    def test_the_history_checkbox_is_the_saved_history_switch(self, page):
        page.set_capture_content(True)

        assert page.captures_content() is True

    def test_retention_offers_forever_and_the_usual_periods(self, page):
        page.set_history_retention_days(0)

        assert list(page._retention_box.cget("values")) == [
            "Forever", "7 days", "30 days", "90 days",
        ]
        assert page.history_retention_days() == 0
        page._retention.set("90 days")
        assert page.history_retention_days() == 90

    def test_a_hand_edited_period_is_kept_rather_than_reset(self, page):
        page.set_history_retention_days(14)

        assert page.history_retention_days() == 14
        assert "14 days" in page._retention_box.cget("values")

    def test_delete_all_history_runs_only_after_confirming(self, tk_root, monkeypatch):
        from voicetotext.settings.tk_pages import output_page

        deleted: list[bool] = []
        answers = iter([False, True])
        monkeypatch.setattr(
            output_page.messagebox, "askyesno", lambda *_a, **_k: next(answers),
        )
        page = TkOutputPage(tk_root, delete_all_history=lambda: deleted.append(True))
        try:
            page._confirm_delete_all()
            page._confirm_delete_all()
        finally:
            page.destroy()

        assert deleted == [True]


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
