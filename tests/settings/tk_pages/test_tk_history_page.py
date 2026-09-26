import tkinter

import pytest

from voicetotext.settings.tk_pages import history_page
from voicetotext.settings.tk_pages.history_page import TkHistoryPage


@pytest.fixture
def page(tk_root):
    widget = TkHistoryPage(tk_root, logs_folder="D:/VoiceToText_logs/Zoë's café ✓")
    yield widget
    widget.destroy()


def test_the_save_history_switch_round_trips(page):
    page.set_capture_content(True)
    assert page.captures_content() is True

    page.set_capture_content(False)
    assert page.captures_content() is False


def test_retention_offers_forever_and_the_usual_periods(page):
    page.set_history_retention_days(0)

    assert list(page._retention_box.cget("values")) == [
        "Forever", "7 days", "30 days", "90 days",
    ]
    assert page.history_retention_days() == 0
    page._retention.set("90 days")
    assert page.history_retention_days() == 90


@pytest.mark.parametrize("days", [14, 1, 3650])
def test_a_hand_edited_period_is_kept_rather_than_reset(page, days):
    page.set_history_retention_days(days)

    assert page.history_retention_days() == days
    assert f"{days} days" in page._retention_box.cget("values")


def test_a_negative_period_means_forever(page):
    page.set_history_retention_days(-5)

    assert page.history_retention_days() == 0


def test_delete_all_history_runs_only_after_confirming_over_the_settings_window(
    tk_root, monkeypatch,
):
    deleted: list[bool] = []
    asked: list[tuple] = []
    answers = iter([False, True])

    def ask(parent, _theme, heading: str, _message: str) -> bool:
        asked.append((parent, heading))
        return next(answers)

    monkeypatch.setattr(history_page, "confirm_deletion", ask)
    window = tkinter.Toplevel(tk_root)
    window.withdraw()
    page = TkHistoryPage(window, delete_all_history=lambda: deleted.append(True))
    try:
        page._confirm_delete_all()
        page._confirm_delete_all()
    finally:
        window.destroy()

    assert deleted == [True]
    assert asked == [(window, "Delete all history")] * 2


def test_the_page_says_which_folder_the_history_lives_in(page):
    notes = [
        str(child.cget("text")) for child in page.winfo_children()
        if child.winfo_class() == "TLabel"
    ]

    assert any("D:/VoiceToText_logs/Zoë's café ✓" in note for note in notes)
