import pytest

from skryb.output.caret_context import (
    CONTEXT_CHARS,
    NO_CONTEXT,
    SELECTION_OUTSIDE_FIELD,
    CaretContext,
    fit_to_context,
)
from skryb.output.paste_memory import PasteMemory, read_or_remember

_OWN_TAG = 0x56325458
_INJECTED = 0x10
_CHAT = 101
_BROWSER = 202

_VK_LCONTROL = 0xA2
_VK_LSHIFT = 0xA0
_VK_LMENU = 0xA4
_VK_CAPITAL = 0x14
_VK_M = 0x4D
_VK_V = 0x56
_VK_A = 0x41
_VK_BACK = 0x08
_VK_RETURN = 0x0D
_VK_LEFT = 0x25
_VK_SPACE = 0x20
_VK_F13 = 0x7C
_HOTKEY = (_VK_LCONTROL, _VK_LSHIFT, _VK_LMENU, _VK_M)
_SECOND_HOTKEY = (_VK_F13,)


class Desktop:
    def __init__(self) -> None:
        self.window = _CHAT
        self.clicks = 0


@pytest.fixture
def desktop() -> Desktop:
    return Desktop()


@pytest.fixture
def memory(desktop: Desktop) -> PasteMemory:
    memory = PasteMemory(lambda: desktop.window, lambda: desktop.clicks, _OWN_TAG)
    memory.set_hotkeys([_HOTKEY, _SECOND_HOTKEY])
    return memory


def press(memory: PasteMemory, vk: int, flags: int = 0, extra: int = 0) -> None:
    memory.observe_key(True, vk, flags, extra)


def release(memory: PasteMemory, vk: int) -> None:
    memory.observe_key(False, vk, 0, 0)


def press_hotkey(memory: PasteMemory) -> None:
    for vk in _HOTKEY:
        press(memory, vk)
    for vk in reversed(_HOTKEY):
        release(memory, vk)


class TestRemembering:
    def test_has_nothing_to_offer_before_anything_is_pasted(self, memory):
        assert memory.context() == NO_CONTEXT

    def test_offers_the_last_paste_while_the_caret_is_untouched(self, memory):
        memory.remember("okay")

        assert memory.context() == CaretContext("okay", "", "")

    def test_joins_back_to_back_pastes_into_the_same_window(self, memory):
        memory.remember("Okay. ")
        memory.remember("Okay. ")

        assert memory.context().before == "Okay. Okay. "

    def test_keeps_only_the_context_width_of_a_long_run_of_pastes(self, memory):
        for index in range(2000):
            memory.remember(f"café {index} ✓ ")

        before = memory.context().before
        assert len(before) == CONTEXT_CHARS
        assert before.endswith("café 1999 ✓ ")

    def test_starts_over_when_a_paste_lands_in_another_window(self, memory, desktop):
        memory.remember("in the chat")
        desktop.window = _BROWSER
        memory.remember("in the browser")

        assert memory.context().before == "in the browser"


class TestForgetting:
    @pytest.mark.parametrize(
        "vk", [_VK_A, _VK_RETURN, _VK_LEFT, _VK_SPACE],
        ids=["a letter the layout types nothing for", "enter", "an arrow", "space the layout types nothing for"],
    )
    def test_forgets_once_the_user_presses_a_key_it_cannot_follow(self, memory, vk):
        memory.remember("okay")

        press(memory, vk)

        assert memory.context() == NO_CONTEXT

    def test_forgets_a_paste_or_undo_the_user_types_themselves(self, memory):
        memory.remember("okay")

        press(memory, _VK_LCONTROL)
        press(memory, _VK_V)

        assert memory.context() == NO_CONTEXT

    def test_forgets_keys_another_program_types_into_the_box(self, memory):
        memory.remember("okay")

        press(memory, _VK_A, flags=_INJECTED)

        assert memory.context() == NO_CONTEXT

    def test_forgets_after_a_mouse_click(self, memory, desktop):
        memory.remember("okay")

        desktop.clicks += 1

        assert memory.context() == NO_CONTEXT

    def test_forgets_after_switching_windows_even_when_switching_back(
        self, memory, desktop,
    ):
        memory.remember("okay")

        desktop.window = _BROWSER
        assert memory.context() == NO_CONTEXT
        desktop.window = _CHAT

        assert memory.context() == NO_CONTEXT

    def test_offers_nothing_when_no_window_has_the_focus(self, memory, desktop):
        memory.remember("okay")

        desktop.window = 0

        assert memory.context() == NO_CONTEXT

    def test_forgets_the_hotkey_letter_typed_on_its_own(self, memory):
        memory.remember("okay")

        press(memory, _VK_M)

        assert memory.context() == NO_CONTEXT


_VK_PERIOD = 0xBE
_VK_SLASH = 0xBF
_VK_2 = 0x32
_VK_DELETE = 0x2E
_VK_Z = 0x5A


def us_layout(vk: int, shift: bool) -> str:
    """A US keyboard for the keys these tests press."""
    if 0x41 <= vk <= 0x5A:
        letter = chr(vk)
        return letter if shift else letter.lower()
    return {
        _VK_SPACE: " ", _VK_PERIOD: ">" if shift else ".", _VK_SLASH: "?" if shift else "/",
        _VK_2: "@" if shift else "2",
    }.get(vk, "")


@pytest.fixture
def typing_memory(desktop: Desktop) -> PasteMemory:
    memory = PasteMemory(lambda: desktop.window, lambda: desktop.clicks, _OWN_TAG, us_layout)
    memory.set_hotkeys([_HOTKEY, _SECOND_HOTKEY])
    return memory


def type_text(memory: PasteMemory, text: str) -> None:
    for character in text:
        shifted = character.isupper() or character in "?@>"
        if shifted:
            press(memory, _VK_LSHIFT)
        vk = {" ": _VK_SPACE, ".": _VK_PERIOD, "/": _VK_SLASH, "?": _VK_SLASH, "@": _VK_2}.get(
            character, ord(character.upper()),
        )
        press(memory, vk)
        release(memory, vk)
        if shifted:
            release(memory, _VK_LSHIFT)


class TestFollowingTyping:
    def test_keeps_up_with_letters_typed_after_a_paste(self, typing_memory):
        typing_memory.remember("Send it to ")

        type_text(typing_memory, "Omar")

        assert typing_memory.context().before == "Send it to Omar"

    def test_backspace_takes_the_last_character_off(self, typing_memory):
        typing_memory.remember("Okay. ")

        press(typing_memory, _VK_BACK)
        press(typing_memory, _VK_BACK)
        type_text(typing_memory, "?")

        assert typing_memory.context().before == "Okay?"

    def test_remembers_what_was_typed_when_nothing_was_pasted_yet(self, typing_memory):
        type_text(typing_memory, "the admin password")

        assert typing_memory.context().before == "the admin password"

    def test_typing_after_a_click_starts_over_from_the_new_caret(self, typing_memory, desktop):
        typing_memory.remember("first message ")
        desktop.clicks += 1

        type_text(typing_memory, "talha")

        assert typing_memory.context().before == "talha"

    def test_keeps_only_the_context_width_of_a_long_run_of_typing(self, typing_memory):
        type_text(typing_memory, "word " * 3000)

        before = typing_memory.context().before
        assert len(before) == CONTEXT_CHARS
        assert before.endswith("word word ")

    @pytest.mark.parametrize("vk", [_VK_RETURN, _VK_LEFT, _VK_DELETE], ids=["enter", "an arrow", "delete"])
    def test_forgets_on_a_key_that_moves_or_sends_the_text(self, typing_memory, vk):
        type_text(typing_memory, "hello")

        press(typing_memory, vk)

        assert typing_memory.context() == NO_CONTEXT

    def test_forgets_on_a_ctrl_shortcut_even_one_that_types_a_letter(self, typing_memory):
        type_text(typing_memory, "hello")

        press(typing_memory, _VK_LCONTROL)
        press(typing_memory, _VK_Z)

        assert typing_memory.context() == NO_CONTEXT

    def test_forgets_letters_another_program_injects(self, typing_memory):
        type_text(typing_memory, "hello")

        press(typing_memory, _VK_A, flags=_INJECTED)

        assert typing_memory.context() == NO_CONTEXT

    def test_backspacing_everything_away_knows_nothing(self, typing_memory):
        type_text(typing_memory, "ab")

        for _ in range(5):
            press(typing_memory, _VK_BACK)

        assert typing_memory.context() == NO_CONTEXT


class TestSpacingAfterTypingIntoAnUnreadableBox:
    def dictate(self, memory: PasteMemory, text: str) -> str:
        read = read_or_remember(lambda: NO_CONTEXT, memory, lambda: None)
        fitted = fit_to_context(text, read())
        memory.remember(fitted)
        return fitted

    def test_puts_a_space_before_a_dictation_that_follows_a_typed_word(self, typing_memory):
        type_text(typing_memory, "talha")

        assert self.dictate(typing_memory, "Can we somehow just tell Rafael. ") == " can we somehow just tell Rafael. "

    def test_puts_a_space_after_retyping_a_character_that_was_backspaced(self, typing_memory):
        self.dictate(typing_memory, "Check the admin passwords. ")
        for _ in range(3):
            press(typing_memory, _VK_BACK)
        type_text(typing_memory, "d")

        assert self.dictate(typing_memory, "How about this? ") == " how about this? "

    def test_glues_nothing_after_a_typed_slash(self, typing_memory):
        type_text(typing_memory, "github.com/")

        assert self.dictate(typing_memory, "HuzPro") == "HuzPro"


class TestKeysThatLeaveTheCaretAlone:
    def test_keeps_the_paste_across_the_hotkey_that_starts_the_next_dictation(
        self, memory,
    ):
        memory.remember("okay")

        press_hotkey(memory)
        press_hotkey(memory)

        assert memory.context().before == "okay"

    def test_keeps_the_paste_across_every_other_dictation_shortcut(self, memory):
        memory.remember("okay")

        press(memory, _VK_F13)
        release(memory, _VK_F13)

        assert memory.context().before == "okay"

    def test_keeps_the_paste_across_its_own_ctrl_v(self, memory):
        memory.remember("okay")

        press(memory, _VK_LCONTROL, flags=_INJECTED, extra=_OWN_TAG)
        press(memory, _VK_V, flags=_INJECTED, extra=_OWN_TAG)

        assert memory.context().before == "okay"

    @pytest.mark.parametrize("vk", [_VK_LSHIFT, _VK_LCONTROL, _VK_CAPITAL])
    def test_keeps_the_paste_when_only_a_modifier_is_pressed(self, memory, vk):
        memory.remember("okay")

        press(memory, vk)
        release(memory, vk)

        assert memory.context().before == "okay"


class TestReadOrRemember:
    def test_uses_a_successful_read_without_waiting(self, memory):
        waits: list[None] = []
        read = CaretContext("Hello ", "", "")

        context = read_or_remember(lambda: read, memory, lambda: waits.append(None))

        assert context() == read
        assert waits == []

    def test_reads_once_more_after_a_short_wait(self, memory):
        reads = iter([NO_CONTEXT, CaretContext("Hello ", "", "")])
        waits: list[None] = []

        context = read_or_remember(
            lambda: next(reads), memory, lambda: waits.append(None),
        )

        assert context() == CaretContext("Hello ", "", "")
        assert len(waits) == 1

    def test_a_selection_outside_the_field_goes_straight_to_the_last_paste(self, memory):
        memory.remember("Ok, ")
        waits: list[None] = []

        context = read_or_remember(
            lambda: SELECTION_OUTSIDE_FIELD, memory, lambda: waits.append(None),
        )

        assert context() == CaretContext("Ok, ", "", "")
        assert waits == []

    def test_falls_back_to_the_last_paste_when_both_reads_fail(self, memory):
        memory.remember("okay")

        context = read_or_remember(lambda: NO_CONTEXT, memory, lambda: None)

        assert context() == CaretContext("okay", "", "")

    def test_reports_no_context_when_the_box_is_unreadable_and_memory_is_stale(
        self, memory,
    ):
        memory.remember("okay")
        press(memory, _VK_A)

        context = read_or_remember(lambda: NO_CONTEXT, memory, lambda: None)

        assert context() == NO_CONTEXT


class TestDictatingIntoAnUnreadableBox:
    def dictate(self, memory: PasteMemory, text: str) -> str:
        read = read_or_remember(lambda: NO_CONTEXT, memory, lambda: None)
        fitted = fit_to_context(text, read())
        memory.remember(fitted)
        return fitted

    def test_spaces_and_lowercases_a_sentence_that_carries_on(self, memory):
        typed = self.dictate(memory, "okay") + self.dictate(memory, "Good luck. ")

        assert typed == "okay good luck. "

    def test_capitalises_after_a_finished_sentence(self, memory):
        typed = self.dictate(memory, "okay. ") + self.dictate(memory, "good luck. ")

        assert typed == "okay. Good luck. "

    def test_never_puts_a_space_at_the_start_of_the_first_dictation(self, memory):
        assert self.dictate(memory, "Good luck. ") == "Good luck. "

    def test_leaves_the_next_dictation_alone_after_the_user_clicks(
        self, memory, desktop,
    ):
        self.dictate(memory, "okay")
        desktop.clicks += 1

        assert self.dictate(memory, "Good luck. ") == "Good luck. "


class TestClicksOnTheRecordingPill:
    def test_a_click_on_the_pill_leaves_the_paste_in_place(self, memory, desktop):
        memory.remember("okay")
        memory.sync_clicks()

        desktop.clicks += 1
        memory.excuse_click()

        assert memory.context().before == "okay"

    def test_a_click_elsewhere_caught_by_a_frame_poll_is_never_excused(
        self, memory, desktop,
    ):
        memory.remember("okay")
        desktop.clicks += 1
        memory.sync_clicks()

        desktop.clicks += 1
        memory.excuse_click()

        assert memory.context() == NO_CONTEXT

    def test_polling_without_any_click_keeps_the_paste(self, memory):
        memory.remember("okay")

        for _ in range(1000):
            memory.sync_clicks()

        assert memory.context().before == "okay"

    def test_excusing_a_click_with_nothing_remembered_remembers_nothing(
        self, memory, desktop,
    ):
        desktop.clicks += 1

        memory.excuse_click()

        assert memory.context() == NO_CONTEXT
