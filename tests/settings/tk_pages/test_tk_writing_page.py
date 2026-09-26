from collections.abc import Callable
from pathlib import Path

import pytest

from voicetotext.settings.tk_pages.writing_page import TkWritingPage, status_text
from voicetotext.writing.install import DOWNLOAD_FAILED
from voicetotext.writing.service import WritingStatus
from voicetotext.writing.settings import WritingSettings
from voicetotext.writing.transforms import BUILTIN_TRANSFORMS


class FakeWriting:
    """The writing facade as the page sees it, with every call recorded."""

    def __init__(self) -> None:
        self.current = WritingStatus("not_installed")
        self.calls: list[str] = []
        self.listeners: list[Callable[[], None]] = []
        self.summary = "nothing yet"
        self.saved: dict[str, str] = dict(BUILTIN_TRANSFORMS)

    def status(self) -> WritingStatus:
        return self.current

    def subscribe(self, listener: Callable[[], None]) -> None:
        self.listeners.append(listener)

    def unsubscribe(self, listener: Callable[[], None]) -> None:
        self.listeners.remove(listener)

    def install(self) -> None:
        self.calls.append("install")

    def cancel_install(self) -> None:
        self.calls.append("cancel_install")

    def retry(self) -> None:
        self.calls.append("retry")

    def open_log(self) -> None:
        self.calls.append("open_log")

    def learned_style_summary(self) -> str:
        return self.summary

    def forget_learned_style(self) -> None:
        self.calls.append("forget_learned_style")
        self.summary = "nothing yet"

    def reset_learning(self) -> None:
        self.calls.append("reset_learning")
        self.summary = "nothing yet"

    def transforms(self) -> dict[str, str]:
        return dict(self.saved)

    def save_transforms(self, transforms: dict[str, str]) -> None:
        self.calls.append("save_transforms")
        self.saved = dict(transforms)

    def forget_learned_words(self, pairs: tuple[tuple[str, str], ...]) -> None:
        self.calls.append(f"forget_learned_words {list(pairs)}")


@pytest.fixture
def writing() -> FakeWriting:
    return FakeWriting()


@pytest.fixture
def scheduled() -> list[Callable[[], None]]:
    return []


@pytest.fixture
def page(tk_root, writing, scheduled):
    widget = TkWritingPage(tk_root, writing, scheduled.append)  # type: ignore[arg-type]
    yield widget
    widget.destroy()


_CUSTOM = WritingSettings(
    cleanup="polished",
    lists=False,
    deadline_seconds=12.5,
    context=False,
    learn_words=False,
    learn_style=True,
    style_notes="British spelling, no exclamation marks.\nSign off with “Cheers” ✓",
    after_dictation_transform="Casual",
    runtime_path=str(Path("C:/tools/llama café/llama-server.exe")),
    model_path=str(Path("D:/models/Qwen3.5-4B-Q4_K_M.gguf")),
)


class TestStagedSettings:
    def test_the_page_round_trips_every_writing_setting(self, page):
        page.set_settings(_CUSTOM)

        assert page.settings() == _CUSTOM

    def test_defaults_round_trip_and_blank_style_notes_read_back_empty(self, page):
        page.set_settings(WritingSettings())

        assert page.settings() == WritingSettings()
        assert page.style_notes() == ""

    @pytest.mark.parametrize(
        ("typed", "expected"),
        [("0.1", 0.5), ("45", 30.0), ("7.5", 7.5), ("four", 4.0), ("", 4.0)],
    )
    def test_a_typed_deadline_is_kept_within_half_a_second_and_thirty(self, page, typed, expected):
        page._deadline.set(typed)

        assert page.settings().deadline_seconds == expected

    def test_a_transform_that_no_longer_exists_reads_as_none(self, page):
        page.set_settings(WritingSettings(after_dictation_transform="Pirate voice"))

        assert page.settings().after_dictation_transform == ""


class TestStatus:
    @pytest.mark.parametrize(
        ("status", "line", "action", "log", "progress"),
        [
            (WritingStatus("not_installed"), "Not installed", "Install (2.8 GB)", False, False),
            (
                WritingStatus("downloading", bytes_done=1_200_000_000, bytes_total=2_774_000_000),
                "Downloading 43% (1.2 of 2.8 GB)", "Cancel", False, True,
            ),
            (WritingStatus("starting"), "Starting...", "", False, False),
            (WritingStatus("ready"), "Ready", "", False, False),
            (
                WritingStatus("ready", last_cleanup_seconds=0.312),
                "Ready. Last cleanup 0.31 s", "", False, False,
            ),
            (
                WritingStatus("failed", "loading model\nggml_vulkan: out of device memory"),
                "Failed: ggml_vulkan: out of device memory", "Retry", True, False,
            ),
            (
                WritingStatus("install_failed", DOWNLOAD_FAILED),
                DOWNLOAD_FAILED, "Retry", True, False,
            ),
            (WritingStatus("stopped"), "Starts once dictation is ready.", "", False, False),
        ],
        ids=["not-installed", "downloading", "starting", "ready", "ready-timed", "failed",
             "install-failed", "stopped"],
    )
    def test_every_status_state_renders_its_line_and_buttons(
        self, page, writing, status, line, action, log, progress,
    ):
        writing.current = status

        page._refresh_status()

        assert page.status_line() == line
        assert page.action_label() == action
        assert page.log_button_shown() is log
        assert page.progress_shown() is progress

    @pytest.mark.parametrize(
        ("state", "call"),
        [("not_installed", "install"), ("downloading", "cancel_install"),
         ("failed", "retry"), ("install_failed", "retry")],
    )
    def test_the_action_button_does_what_it_says(self, page, writing, state, call):
        writing.current = WritingStatus(state)
        page._refresh_status()

        page.press_action()

        assert writing.calls == [call]

    def test_status_changes_from_other_threads_are_handed_to_the_gui_thread(
        self, page, writing, scheduled,
    ):
        writing.current = WritingStatus("ready")

        writing.listeners[0]()
        assert page.status_line() == "Not installed"
        scheduled.pop()()

        assert page.status_line() == "Ready"

    def test_closing_the_page_stops_listening(self, tk_root, writing):
        page = TkWritingPage(tk_root, writing, lambda action: action())  # type: ignore[arg-type]

        page.destroy()

        assert writing.listeners == []

    def test_a_zero_byte_download_does_not_divide_by_zero(self):
        assert status_text(WritingStatus("downloading")) == "Downloading 0% (0.0 of 0.0 GB)"


class TestTransformations:
    def test_built_ins_are_listed_and_editable(self, page):
        assert page.transform_names() == ["Concise", "Bullet points", "Professional", "Casual"]
        page.select_transform("Casual")
        page._transform_name.set("Friendly")
        page._transform_text.set("Warm and friendly, keep every fact.")

        page.edit_transform()

        assert page.transform_names() == ["Concise", "Bullet points", "Professional", "Friendly"]
        assert page.transforms()["Friendly"] == "Warm and friendly, keep every fact."
        assert page.transforms_dirty()

    def test_add_remove_and_restore_built_ins(self, page):
        page._transform_name.set("Emoji-free ✓")
        page._transform_text.set("Remove every emoji.")
        page.add_transform()
        page.select_transform("Concise")
        page.remove_transform()
        assert "Concise" not in page.transform_names()

        page.restore_builtins()

        assert set(BUILTIN_TRANSFORMS) | {"Emoji-free ✓"} == set(page.transform_names())

    def test_removing_the_chosen_transform_sets_the_choice_back_to_none(self, page):
        page.set_settings(WritingSettings(after_dictation_transform="Concise"))
        page.select_transform("Concise")

        page.remove_transform()

        assert page.settings().after_dictation_transform == ""

    def test_renaming_the_chosen_transform_keeps_it_chosen(self, page):
        page.set_settings(WritingSettings(after_dictation_transform="Concise"))
        page.select_transform("Concise")
        page._transform_name.set("Short")
        page._transform_text.set("Make it short.")

        page.edit_transform()

        assert page.settings().after_dictation_transform == "Short"

    @pytest.mark.parametrize(("name", "text"), [("", "Do it."), ("Named", ""), ("  ", "  ")])
    def test_a_transformation_needs_a_name_and_an_instruction(self, page, name, text):
        page._transform_name.set(name)
        page._transform_text.set(text)

        page.add_transform()

        assert not page.transforms_dirty()
        assert page._message.get()

    def test_edit_and_remove_need_a_selection(self, page):
        page.remove_transform()
        page.edit_transform()

        assert not page.transforms_dirty()


class TestLearning:
    def test_forget_and_reset_act_at_once_and_refresh_the_learned_line(self, page, writing):
        writing.summary = "prefers contractions; drops the final full stop"
        page._refresh_learned()
        assert page.learned_line() == "Learned so far: prefers contractions; drops the final full stop"

        page._forget_style()
        page._reset_learning()

        assert writing.calls == ["forget_learned_style", "reset_learning"]
        assert page.learned_line() == "Learned so far: nothing yet"
