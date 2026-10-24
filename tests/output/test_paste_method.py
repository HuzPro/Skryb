import pytest

from skryb.output.paste_method import (
    PasteMethod,
    normalized_process_name,
    paste_method_for,
    utf16_chunks,
)


def test_mintty_gets_shift_insert_and_everything_else_ctrl_v_by_default() -> None:
    assert paste_method_for("mintty.exe", {}) is PasteMethod.SHIFT_INSERT
    assert paste_method_for("MINTTY.EXE", None) is PasteMethod.SHIFT_INSERT
    assert paste_method_for("chrome.exe", {}) is PasteMethod.CTRL_V
    assert paste_method_for("", {}) is PasteMethod.CTRL_V


def test_a_per_app_override_beats_the_automatic_rule() -> None:
    overrides = {"mintty.exe": "ctrl_shift_v", "WindowsTerminal.exe": "ctrl_shift_v"}

    assert paste_method_for("mintty.exe", overrides) is PasteMethod.CTRL_SHIFT_V
    assert paste_method_for("windowsterminal.exe", overrides) is PasteMethod.CTRL_SHIFT_V


@pytest.mark.parametrize("process", ["notepad.exe", "unknown-app.exe", "文字.exe", ""])
def test_unknown_apps_and_broken_overrides_fall_back_to_ctrl_v(process: str) -> None:
    overrides = {
        "vmconnect.exe": "type",
        "notepad.exe": "middle_click",
        "": "type",
        42: "type",
        "文字.exe": ["type"],
    }

    assert paste_method_for(process, overrides) is PasteMethod.CTRL_V


@pytest.mark.parametrize("configured", [None, "ctrl_v", ["mintty.exe"], 7])
def test_overrides_that_are_not_a_mapping_are_ignored(configured: object) -> None:
    assert paste_method_for("mintty.exe", configured) is PasteMethod.SHIFT_INSERT


@pytest.mark.parametrize(
    ("typed", "stored"),
    [("mintty", "mintty.exe"), (" Code.EXE ", "code.exe"), ("", ""), ("   ", "")],
)
def test_process_names_are_stored_the_way_windows_reports_them(
    typed: str, stored: str,
) -> None:
    assert normalized_process_name(typed) == stored


def test_typing_splits_five_thousand_characters_into_batches_without_breaking_emoji(
) -> None:
    text = ("Grüße 🎙️ naïve café, 東京 → 👩‍💻 done.\n" * 150)[:5000]

    chunks = utf16_chunks(text, max_units=64)

    assert all(len(chunk) <= 64 for chunk in chunks)
    joined = b"".join(unit.to_bytes(2, "little") for chunk in chunks for unit in chunk)
    assert joined.decode("utf-16-le") == text
    for chunk in chunks:
        assert not 0xD800 <= chunk[-1] <= 0xDBFF, "a surrogate pair was split"
        assert not 0xDC00 <= chunk[0] <= 0xDFFF, "a surrogate pair was split"


def test_typing_nothing_sends_nothing() -> None:
    assert utf16_chunks("") == []
