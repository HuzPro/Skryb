from dataclasses import dataclass, replace
from enum import StrEnum


class EditFlag(StrEnum):
    """A reason the replayed buffer may not match what the box really holds."""

    MOUSE_CLICK = "mouse_click"
    UNDO = "undo"
    FOREIGN_PASTE = "foreign_paste"
    SPLIT_PASTE = "split_paste"
    SELECT_ALL = "select_all"
    CARET_BEFORE_SPAN = "caret_before_span"
    CARET_AFTER_SPAN = "caret_after_span"
    VERTICAL_ARROW = "vertical_arrow"
    TAB = "tab"
    ESCAPE = "escape"
    IME = "ime"


# After any of these the replay can no longer say where the caret sits or what
# the box holds, so every later edit lands at an offset we made up. SELECT_ALL
# and the two caret-edge flags stay out because they only say the edit may have
# reached outside our own span.
UNRELIABLE_FLAGS = frozenset({
    EditFlag.MOUSE_CLICK,
    EditFlag.UNDO,
    EditFlag.FOREIGN_PASTE,
    EditFlag.SPLIT_PASTE,
    EditFlag.VERTICAL_ARROW,
    EditFlag.TAB,
    EditFlag.ESCAPE,
    EditFlag.IME,
})


@dataclass(frozen=True)
class KeyStroke:
    """One key-down event, already translated to the character it types."""

    vk: int
    char: str
    shift: bool = False
    ctrl: bool = False
    alt: bool = False
    click_since_last: bool = False


@dataclass(frozen=True)
class PasteSpan:
    """What one pasted transcription became by the time the text was sent."""

    ordinal: int
    pasted_text: str
    final_text: str


@dataclass(frozen=True)
class EditState:
    """The replayed buffer after one input, and whether that input sent it."""

    committed: bool
    text: str
    spans: tuple[PasteSpan, ...]
    flags: frozenset[EditFlag]

    @property
    def reliable(self) -> bool:
        return not (self.flags & UNRELIABLE_FLAGS)

    @property
    def pasted_text(self) -> str:
        return "".join(span.pasted_text for span in self.spans)


VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_ESCAPE = 0x1B
VK_PRIOR = 0x21
VK_NEXT = 0x22
VK_END = 0x23
VK_HOME = 0x24
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_DELETE = 0x2E
VK_A = 0x41
VK_V = 0x56
VK_X = 0x58
VK_Y = 0x59
VK_Z = 0x5A
VK_PROCESSKEY = 0xE5

NEWLINE = "\n"
TYPED_ORIGIN = -1


class EditReplay:
    """The text box as we believe it stands, rebuilt from our own paste.

    We know the exact string we pasted, so replaying the keystrokes that
    followed tells us what the user sent without ever reading the box.
    """

    def __init__(self) -> None:
        self._chars: list[str] = []
        self._origins: list[int] = []
        self._pasted_texts: list[str] = []
        self._cursor = 0
        self._anchor = 0
        self._flags: set[EditFlag] = set()

    def apply_paste(self, text: str) -> EditState:
        if self._lands_inside_earlier_text():
            self._flags.add(EditFlag.SPLIT_PASTE)
        self._insert(text, len(self._pasted_texts))
        self._pasted_texts.append(text)
        return self.snapshot()

    def _lands_inside_earlier_text(self) -> bool:
        """Whether the paste splits a span instead of extending the buffer.

        A selection about to be replaced counts as already consumed, so the
        far edge of the selection is where the insertion effectively sits.
        """
        return max(self._anchor, self._cursor) != len(self._chars)

    def apply_key(self, stroke: KeyStroke) -> EditState:
        if stroke.click_since_last:
            self._flags.add(EditFlag.MOUSE_CLICK)
        if stroke.vk == VK_PROCESSKEY:
            self._flags.add(EditFlag.IME)
        elif stroke.vk == VK_RETURN:
            return self._apply_return(stroke)
        elif stroke.ctrl and not stroke.alt:
            self._apply_control_key(stroke)
        elif stroke.alt:
            self._apply_alt_key(stroke)
        else:
            self._apply_plain_key(stroke)
        return self.snapshot()

    def _apply_return(self, stroke: KeyStroke) -> EditState:
        """Shift+Enter is a newline in every chat app; plain Enter sends.

        Ctrl+Enter sends in some, so it commits too. Getting this wrong
        would end a dictation on a line break and report half a message.
        """
        if stroke.shift:
            self._insert(NEWLINE, TYPED_ORIGIN)
            return self.snapshot()
        return self._commit()

    def snapshot(self) -> EditState:
        return EditState(
            committed=False,
            text="".join(self._chars),
            spans=self._spans(),
            flags=frozenset(self._flags),
        )

    def _commit(self) -> EditState:
        """Reports the buffer as it stood before the newline lands.

        A chat box sends on Enter while an editor keeps the newline, so the
        pre-newline buffer is the sent message in one and a checkpoint in the
        other. Nothing here has to know which kind of box it pasted into.
        """
        sent = replace(self.snapshot(), committed=True)
        self._insert(NEWLINE, TYPED_ORIGIN)
        return sent

    def _apply_control_key(self, stroke: KeyStroke) -> None:
        if stroke.vk == VK_A:
            self._select_all()
        elif stroke.vk in (VK_Z, VK_Y):
            self._flags.add(EditFlag.UNDO)
        elif stroke.vk == VK_V:
            self._flags.add(EditFlag.FOREIGN_PASTE)
        elif stroke.vk == VK_X:
            self._delete_selection()
        elif stroke.vk == VK_BACK:
            self._delete_word_before_cursor()
        elif stroke.vk == VK_DELETE:
            self._delete_word_after_cursor()
        elif stroke.vk in (VK_LEFT, VK_RIGHT):
            self._move_by_word(stroke)
        elif stroke.vk == VK_HOME:
            self._flags.add(EditFlag.CARET_BEFORE_SPAN)
            self._move_caret(0, extend_selection=stroke.shift)
        elif stroke.vk == VK_END:
            self._flags.add(EditFlag.CARET_AFTER_SPAN)
            self._move_caret(len(self._chars), extend_selection=stroke.shift)

    def _apply_alt_key(self, stroke: KeyStroke) -> None:
        """Windows sends AltGr as Ctrl+Alt, so that pair still types."""
        if stroke.vk == VK_BACK:
            self._flags.add(EditFlag.UNDO)
        elif stroke.ctrl and stroke.char:
            self._insert(stroke.char, TYPED_ORIGIN)

    def _apply_plain_key(self, stroke: KeyStroke) -> None:
        if stroke.vk == VK_BACK:
            self._delete_before_cursor()
        elif stroke.vk == VK_DELETE:
            self._delete_at_cursor()
        elif stroke.vk in (VK_LEFT, VK_RIGHT):
            self._move_by_character(stroke)
        elif stroke.vk in (VK_HOME, VK_END):
            self._move_to_line_edge(stroke)
        elif stroke.vk in (VK_UP, VK_DOWN, VK_PRIOR, VK_NEXT):
            self._flags.add(EditFlag.VERTICAL_ARROW)
        elif stroke.vk == VK_TAB:
            self._flags.add(EditFlag.TAB)
        elif stroke.vk == VK_ESCAPE:
            self._flags.add(EditFlag.ESCAPE)
        elif stroke.char:
            self._insert(stroke.char, TYPED_ORIGIN)

    def _select_all(self) -> None:
        self._flags.add(EditFlag.SELECT_ALL)
        self._anchor = 0
        self._cursor = len(self._chars)

    def _move_by_character(self, stroke: KeyStroke) -> None:
        if self._collapses_a_selection(stroke):
            self._collapse_selection_toward(stroke)
            return
        step = -1 if stroke.vk == VK_LEFT else 1
        target = self._clamp_and_flag_overrun(self._cursor + step)
        self._move_caret(target, extend_selection=stroke.shift)

    def _move_by_word(self, stroke: KeyStroke) -> None:
        if self._collapses_a_selection(stroke):
            self._collapse_selection_toward(stroke)
            return
        target = (
            self._word_start_before_cursor()
            if stroke.vk == VK_LEFT
            else self._next_word_start_after_cursor()
        )
        self._move_caret(target, extend_selection=stroke.shift)

    def _collapses_a_selection(self, stroke: KeyStroke) -> bool:
        """An unshifted arrow over a selection only drops the highlight.

        Windows puts the caret on the edge the arrow points at and stops
        there, one character short of where a plain step would land it.
        """
        return self._has_selection() and not stroke.shift

    def _collapse_selection_toward(self, stroke: KeyStroke) -> None:
        edges = (self._anchor, self._cursor)
        target = min(edges) if stroke.vk == VK_LEFT else max(edges)
        self._move_caret(target, extend_selection=False)

    def _move_to_line_edge(self, stroke: KeyStroke) -> None:
        target = self._line_start() if stroke.vk == VK_HOME else self._line_end()
        self._flag_span_edge(target)
        self._move_caret(target, extend_selection=stroke.shift)

    def _move_caret(self, position: int, extend_selection: bool) -> None:
        self._cursor = position
        if not extend_selection:
            self._anchor = position

    def _clamp_and_flag_overrun(self, position: int) -> int:
        if position < 0:
            self._flags.add(EditFlag.CARET_BEFORE_SPAN)
            return 0
        if position > len(self._chars):
            self._flags.add(EditFlag.CARET_AFTER_SPAN)
            return len(self._chars)
        return position

    def _flag_span_edge(self, position: int) -> None:
        if position == 0:
            self._flags.add(EditFlag.CARET_BEFORE_SPAN)
        if position == len(self._chars):
            self._flags.add(EditFlag.CARET_AFTER_SPAN)

    def _line_start(self) -> int:
        position = self._cursor
        while position > 0 and self._chars[position - 1] != NEWLINE:
            position -= 1
        return position

    def _line_end(self) -> int:
        position = self._cursor
        while position < len(self._chars) and self._chars[position] != NEWLINE:
            position += 1
        return position

    def _word_start_before_cursor(self) -> int:
        position = self._cursor
        while position > 0 and self._chars[position - 1].isspace():
            position -= 1
        while position > 0 and not self._chars[position - 1].isspace():
            position -= 1
        return position

    def _next_word_start_after_cursor(self) -> int:
        """Where Ctrl+Right lands and Ctrl+Delete deletes up to.

        Windows leaves the caret at the start of the following word, so the
        run being crossed is the rest of this word and then the space after
        it. Skipping the space first would land a word further on and put
        every later edit at the wrong offset.
        """
        position = self._cursor
        end = len(self._chars)
        while position < end and not self._chars[position].isspace():
            position += 1
        while position < end and self._chars[position].isspace():
            position += 1
        return position

    def _delete_before_cursor(self) -> None:
        if self._has_selection():
            self._delete_selection()
        elif self._cursor == 0:
            self._flags.add(EditFlag.CARET_BEFORE_SPAN)
        else:
            self._delete_range(self._cursor - 1, self._cursor)

    def _delete_at_cursor(self) -> None:
        if self._has_selection():
            self._delete_selection()
        elif self._cursor == len(self._chars):
            self._flags.add(EditFlag.CARET_AFTER_SPAN)
        else:
            self._delete_range(self._cursor, self._cursor + 1)

    def _delete_word_before_cursor(self) -> None:
        if self._has_selection():
            self._delete_selection()
        else:
            self._delete_range(self._word_start_before_cursor(), self._cursor)

    def _delete_word_after_cursor(self) -> None:
        if self._has_selection():
            self._delete_selection()
        else:
            self._delete_range(self._cursor, self._next_word_start_after_cursor())

    def _has_selection(self) -> bool:
        return self._anchor != self._cursor

    def _delete_selection(self) -> None:
        if self._has_selection():
            self._delete_range(
                min(self._anchor, self._cursor), max(self._anchor, self._cursor)
            )

    def _delete_range(self, start: int, stop: int) -> None:
        del self._chars[start:stop]
        del self._origins[start:stop]
        self._move_caret(start, extend_selection=False)

    def _insert(self, text: str, origin: int) -> None:
        self._delete_selection()
        at = self._cursor
        self._chars[at:at] = list(text)
        self._origins[at:at] = [origin] * len(text)
        self._move_caret(at + len(text), extend_selection=False)

    def _spans(self) -> tuple[PasteSpan, ...]:
        owners = self._paste_owners()
        return tuple(
            PasteSpan(ordinal, pasted, self._text_owned_by(owners, ordinal))
            for ordinal, pasted in enumerate(self._pasted_texts)
        )

    def _paste_owners(self) -> list[int]:
        owners = list(self._origins)
        preceding = TYPED_ORIGIN
        for index, origin in enumerate(owners):
            if origin == TYPED_ORIGIN:
                owners[index] = preceding
            else:
                preceding = origin
        following = TYPED_ORIGIN
        for index in reversed(range(len(owners))):
            if owners[index] == TYPED_ORIGIN:
                owners[index] = following
            else:
                following = owners[index]
        return owners

    def _text_owned_by(self, owners: list[int], ordinal: int) -> str:
        """The one run around this paste, not every stretch it ever touched.

        A second dictation typed into the middle of the first splits the
        first's characters in two. Taking the run holding its own first
        character keeps the inner paste out of the outer paste's label.
        """
        if ordinal not in owners:
            return ""
        start = self._origins.index(ordinal)
        while start > 0 and owners[start - 1] == ordinal:
            start -= 1
        stop = start
        while stop < len(owners) and owners[stop] == ordinal:
            stop += 1
        return "".join(self._chars[start:stop])
