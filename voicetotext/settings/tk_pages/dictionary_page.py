import re
import tkinter
from collections.abc import Callable
from dataclasses import replace
from tkinter import ttk

from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryOptions,
    DictionaryRule,
    LiteralRule,
    RegexRule,
)
from voicetotext.output.punctuation_stage import sherpa_onnx_available
from voicetotext.settings.theme import (
    BODY_FONT,
    CARD_BODY_STYLE,
    CARD_MUTED_STYLE,
    CARD_PADDING,
    CARD_TEXT_STYLE,
    CARD_TEXT_WRAP,
    CHECK_STYLE,
    CONTROL_GAP,
    DANGER_BUTTON_STYLE,
    LINK_BUTTON_STYLE,
    RADIO_STYLE,
    RowGroup,
    card,
    hide_while_empty,
    page_note,
    section_heading,
    switch,
)

_HINT = (
    "Replace spoken words or phrases with typed text. Separate literal aliases "
    "with commas; regex rules use one pattern. A space at the start or end of "
    "the aliases box glues every alias to the neighbouring text."
)
_EMPTY_LIST = "No words yet. Add one below, or correct a pasted word and it is learned."
_EMPTY_SPOKEN_ERROR = "At least one spoken alias or regex pattern is required."
_INVALID_REGEX_ERROR = "The regex pattern or replacement is invalid: {}"
_NO_SELECTION_ERROR = "Select a rule from the list first."
_DUPLICATE_ERROR = "Another literal rule already uses one of those aliases."
_NOTHING_SELECTED = -1
_PUNCTUATION_INSTALL_MESSAGE = (
    "Punctuation restoration requires the optional package: "
    "pip install voicetotext[punct]"
)
_TABLE_ROWS = 8


class TkDictionaryPage(ttk.Frame):
    def __init__(
        self,
        parent: tkinter.Misc,
        punctuation_available: Callable[[], bool] = sherpa_onnx_available,
    ) -> None:
        super().__init__(parent)
        self._punctuation_available = punctuation_available
        self._dictionary = DictionaryConfig()
        self._selected_index = _NOTHING_SELECTED
        self._dirty = False
        # Learned words the user removed here; the learner must not relearn them.
        self._forgotten: dict[tuple[str, str], None] = {}
        self._remove_fillers_var = tkinter.BooleanVar(master=self)
        self._spoken_commands_var = tkinter.BooleanVar(master=self)
        self._numbers_var = tkinter.BooleanVar(master=self)
        self._punctuation_model_var = tkinter.BooleanVar(master=self)
        self._rule_type_var = tkinter.StringVar(master=self, value="literal")
        self._spoken_var = tkinter.StringVar(master=self)
        self._replacement_var = tkinter.StringVar(master=self)
        self._case_sensitive_var = tkinter.BooleanVar(master=self)
        self._whole_words_var = tkinter.BooleanVar(master=self, value=True)
        self._phonetic_var = tkinter.BooleanVar(master=self)
        self._message_var = tkinter.StringVar(master=self)
        self._build()

    def _build(self) -> None:
        section_heading(self, "Cleanup", first=True)
        self._build_options()
        section_heading(self, "Words and replacements")
        page_note(self, _HINT).pack(anchor="w", pady=(0, CONTROL_GAP))
        self._build_table()
        self._build_editor()

    def _build_options(self) -> None:
        rows = RowGroup(self)
        rows.pack(fill="x")
        self._fillers_check = self._option(
            rows, "Remove fillers", "Drops um, uh, uhh, umm and erm.",
            self._remove_fillers_var, self._mark_dirty,
        )
        self._commands_check = self._option(
            rows, "Spoken commands", "Say “new line” or “period” to type them.",
            self._spoken_commands_var, self._mark_dirty,
        )
        self._numbers_check = self._option(
            rows, "Format numbers", "Writes “twenty five” as 25.",
            self._numbers_var, self._mark_dirty,
        )
        self._punctuation_check = self._option(
            rows, "Restore punctuation", "Adds punctuation with a small extra model.",
            self._punctuation_model_var, self._on_punctuation_toggle,
        )

    @staticmethod
    def _option(
        rows: RowGroup,
        title: str,
        description: str,
        variable: tkinter.BooleanVar,
        command: Callable[[], None],
    ) -> ttk.Checkbutton:
        row = rows.add(
            title, description, lambda parent: switch(parent, variable, command=command),
        )
        assert isinstance(row.control, ttk.Checkbutton)
        return row.control

    def _build_table(self) -> None:
        table = card(self, padding=1)
        table.pack(fill="x")
        self._tree = ttk.Treeview(
            table, columns=("spoken", "replacement", "kind"), show="headings",
            height=_TABLE_ROWS, selectmode="browse",
        )
        for column, heading, width, stretch in (
            ("spoken", "When I say", 240, True),
            ("replacement", "Replace with", 200, True),
            ("kind", "", 80, False),
        ):
            self._tree.heading(column, text=heading, anchor="w")
            self._tree.column(column, width=width, stretch=stretch, anchor="w")
        self._tree.pack(fill="x")
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._empty = ttk.Label(
            table, text=_EMPTY_LIST, style=CARD_MUTED_STYLE,
            wraplength=CARD_TEXT_WRAP, justify="center",
        )

    def _build_editor(self) -> None:
        editor = card(self)
        editor.pack(fill="x", pady=(CONTROL_GAP, 0))
        editor.columnconfigure(1, weight=1)

        ttk.Label(editor, text="Type", style=CARD_TEXT_STYLE).grid(row=0, column=0, sticky="w")
        type_frame = ttk.Frame(editor, style=CARD_BODY_STYLE)
        type_frame.grid(row=0, column=1, sticky="w", padx=(12, 0))
        self._literal_radio = ttk.Radiobutton(
            type_frame,
            text="Literal",
            value="literal",
            variable=self._rule_type_var,
            command=self._sync_rule_type,
            style=RADIO_STYLE,
        )
        self._literal_radio.pack(side="left")
        self._regex_radio = ttk.Radiobutton(
            type_frame,
            text="Regex",
            value="regex",
            variable=self._rule_type_var,
            command=self._sync_rule_type,
            style=RADIO_STYLE,
        )
        self._regex_radio.pack(side="left", padx=(16, 0))

        self._spoken_entry = self._field(editor, 1, "Aliases / pattern", self._spoken_var)
        self._replacement_entry = self._field(editor, 2, "Replacement", self._replacement_var)

        options = ttk.Frame(editor, style=CARD_BODY_STYLE)
        options.grid(row=3, column=1, sticky="w", padx=(12, 0), pady=(6, 0))
        self._case_check = ttk.Checkbutton(
            options, text="Case sensitive", variable=self._case_sensitive_var,
            style=CHECK_STYLE,
        )
        self._case_check.pack(side="left")
        self._whole_check = ttk.Checkbutton(
            options, text="Whole words", variable=self._whole_words_var, style=CHECK_STYLE,
        )
        self._whole_check.pack(side="left", padx=(16, 0))
        self._phonetic_check = ttk.Checkbutton(
            options, text="Phonetic", variable=self._phonetic_var, style=CHECK_STYLE,
        )
        self._phonetic_check.pack(side="left", padx=(16, 0))

        buttons = ttk.Frame(editor, style=CARD_BODY_STYLE)
        buttons.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(CARD_PADDING, 0))
        self._add_button = ttk.Button(buttons, text="Add", command=self.add_rule)
        self._add_button.pack(side="left")
        self._update_button = ttk.Button(
            buttons, text="Update", command=self.update_selected_rule,
        )
        self._update_button.pack(side="left", padx=(CONTROL_GAP, 0))
        self._delete_button = ttk.Button(
            buttons, text="Delete", style=DANGER_BUTTON_STYLE,
            command=self.delete_selected_rule,
        )
        self._delete_button.pack(side="left", padx=(CONTROL_GAP, 0))
        self._forget_learned_button = ttk.Button(
            buttons, text="Forget all learned words", style=LINK_BUTTON_STYLE,
            command=self.forget_learned_words,
        )
        self._forget_learned_button.pack(side="right")
        message = ttk.Label(
            editor, textvariable=self._message_var, style="CardDanger.TLabel",
            wraplength=CARD_TEXT_WRAP, justify="left",
        )
        message.grid(row=5, column=0, columnspan=2, sticky="w", pady=(CONTROL_GAP, 0))
        hide_while_empty(message, self._message_var)

    @staticmethod
    def _field(
        editor: ttk.Frame, row: int, label: str, variable: tkinter.StringVar,
    ) -> ttk.Entry:
        ttk.Label(editor, text=label, style=CARD_TEXT_STYLE).grid(
            row=row, column=0, sticky="w", pady=(CONTROL_GAP, 0),
        )
        entry = ttk.Entry(editor, textvariable=variable, font=BODY_FONT)
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=(CONTROL_GAP, 0))
        return entry

    def add_rule(self) -> None:
        rule = self._rule_from_editor()
        if rule is None or self._has_duplicate_alias(rule):
            return
        self._dictionary = replace(
            self._dictionary, rules=self._dictionary.rules + (rule,),
        )
        self._commit_edit()

    def update_selected_rule(self) -> None:
        if self._selected_index == _NOTHING_SELECTED:
            self._message_var.set(_NO_SELECTION_ERROR)
            return
        rule = self._rule_from_editor(self._dictionary.rules[self._selected_index])
        if rule is None or self._has_duplicate_alias(rule, self._selected_index):
            return
        rules = list(self._dictionary.rules)
        rules[self._selected_index] = rule
        self._dictionary = replace(self._dictionary, rules=tuple(rules))
        self._commit_edit()

    def delete_selected_rule(self) -> None:
        if self._selected_index == _NOTHING_SELECTED:
            self._message_var.set(_NO_SELECTION_ERROR)
            return
        rules = list(self._dictionary.rules)
        self._remember_if_learned(rules.pop(self._selected_index))
        self._dictionary = replace(self._dictionary, rules=tuple(rules))
        self._commit_edit()

    def forget_learned_words(self) -> None:
        kept: list[DictionaryRule] = []
        for rule in self._dictionary.rules:
            if isinstance(rule, LiteralRule) and rule.learned:
                self._remember_if_learned(rule)
            else:
                kept.append(rule)
        self._dictionary = replace(self._dictionary, rules=tuple(kept))
        self._commit_edit()

    def forgotten_learned_words(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._forgotten)

    def _remember_if_learned(self, rule: DictionaryRule) -> None:
        if isinstance(rule, LiteralRule) and rule.learned:
            self._forgotten[(rule.spoken[0].strip(), rule.replacement)] = None

    def _rule_from_editor(
        self, original: DictionaryRule | None = None,
    ) -> DictionaryRule | None:
        spoken = self._spoken_var.get()
        if not spoken.strip():
            self._message_var.set(_EMPTY_SPOKEN_ERROR)
            return None
        replacement = self._replacement_var.get()
        if self._rule_type_var.get() == "regex":
            try:
                compiled = re.compile(spoken)
                compiled.sub(replacement, "")
            except re.error as exc:
                self._message_var.set(_INVALID_REGEX_ERROR.format(exc))
                return None
            enabled = original.enabled if isinstance(original, RegexRule) else True
            return RegexRule(spoken, replacement, enabled)
        aliases = _split_aliases(spoken)
        if not aliases:
            self._message_var.set(_EMPTY_SPOKEN_ERROR)
            return None
        enabled = original.enabled if isinstance(original, LiteralRule) else True
        return LiteralRule(
            spoken=aliases,
            replacement=replacement,
            case_sensitive=self._case_sensitive_var.get(),
            whole_words=self._whole_words_var.get(),
            phonetic=self._phonetic_var.get(),
            enabled=enabled,
        )

    def _has_duplicate_alias(
        self, candidate: DictionaryRule, ignored_index: int = _NOTHING_SELECTED,
    ) -> bool:
        if not isinstance(candidate, LiteralRule):
            return False
        aliases = _plain_aliases(candidate)
        duplicate = any(
            index != ignored_index
            and isinstance(rule, LiteralRule)
            and bool(aliases & _plain_aliases(rule))
            for index, rule in enumerate(self._dictionary.rules)
        )
        if duplicate:
            self._message_var.set(_DUPLICATE_ERROR)
        return duplicate

    def _commit_edit(self) -> None:
        self._dirty = True
        self._clear_editor()
        self._refresh_rules()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_options()

    def _update_options(self) -> None:
        self._dictionary = replace(
            self._dictionary,
            options=DictionaryOptions(
                remove_fillers=self._remove_fillers_var.get(),
                spoken_commands=self._spoken_commands_var.get(),
                numbers=self._numbers_var.get(),
                punctuation_model=self._punctuation_model_var.get(),
            ),
        )

    def _on_punctuation_toggle(self) -> None:
        self._mark_dirty()
        if self._punctuation_model_var.get() and not self._punctuation_available():
            self._message_var.set(_PUNCTUATION_INSTALL_MESSAGE)

    def _clear_editor(self) -> None:
        self._selected_index = _NOTHING_SELECTED
        self._rule_type_var.set("literal")
        self._spoken_var.set("")
        self._replacement_var.set("")
        self._case_sensitive_var.set(False)
        self._whole_words_var.set(True)
        self._phonetic_var.set(False)
        self._message_var.set("")
        self._sync_rule_type()

    def _refresh_rules(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for index, rule in enumerate(self._dictionary.rules):
            self._tree.insert("", "end", iid=str(index), values=_rule_columns(rule))
        if self._dictionary.rules:
            self._empty.place_forget()
        else:
            self._empty.place(in_=self._tree, relx=0.5, rely=0.55, anchor="center")

    def _on_tree_select(self, _event: tkinter.Event) -> None:
        selection = self._tree.selection()
        if selection:
            self.select_rule(int(selection[0]))

    def select_rule(self, index: int) -> None:
        self._selected_index = index
        rule = self._dictionary.rules[index]
        if isinstance(rule, LiteralRule):
            self._rule_type_var.set("literal")
            self._spoken_var.set(_join_aliases(rule.spoken))
            self._case_sensitive_var.set(rule.case_sensitive)
            self._whole_words_var.set(rule.whole_words)
            self._phonetic_var.set(rule.phonetic)
        else:
            self._rule_type_var.set("regex")
            self._spoken_var.set(rule.pattern)
            self._case_sensitive_var.set(False)
            self._whole_words_var.set(False)
            self._phonetic_var.set(False)
        self._replacement_var.set(rule.replacement)
        self._message_var.set("")
        self._sync_rule_type()

    def _sync_rule_type(self) -> None:
        state = "!disabled" if self._rule_type_var.get() == "literal" else "disabled"
        self._case_check.state([state])
        self._whole_check.state([state])
        self._phonetic_check.state([state])

    def dictionary(self) -> DictionaryConfig:
        self._update_options()
        return self._dictionary

    def set_dictionary(self, dictionary: DictionaryConfig) -> None:
        self._dictionary = dictionary
        self._remove_fillers_var.set(dictionary.options.remove_fillers)
        self._spoken_commands_var.set(dictionary.options.spoken_commands)
        self._numbers_var.set(dictionary.options.numbers)
        self._punctuation_model_var.set(dictionary.options.punctuation_model)
        self._dirty = False
        self._forgotten = {}
        self._clear_editor()
        self._refresh_rules()
        if dictionary.options.punctuation_model and not self._punctuation_available():
            self._message_var.set(_PUNCTUATION_INSTALL_MESSAGE)

    def is_dirty(self) -> bool:
        return self._dirty

    def rule_labels(self) -> list[str]:
        return [_rule_label(rule) for rule in self._dictionary.rules]

    def message(self) -> str:
        return self._message_var.get()

    def spoken_entry(self) -> ttk.Entry:
        return self._spoken_entry

    def replacement_entry(self) -> ttk.Entry:
        return self._replacement_entry

    def add_button(self) -> ttk.Button:
        return self._add_button

    def update_button(self) -> ttk.Button:
        return self._update_button

    def delete_button(self) -> ttk.Button:
        return self._delete_button

    def forget_learned_button(self) -> ttk.Button:
        return self._forget_learned_button

    def fillers_check(self) -> ttk.Checkbutton:
        return self._fillers_check

    def commands_check(self) -> ttk.Checkbutton:
        return self._commands_check

    def numbers_check(self) -> ttk.Checkbutton:
        return self._numbers_check

    def punctuation_check(self) -> ttk.Checkbutton:
        return self._punctuation_check

    def literal_radio(self) -> ttk.Radiobutton:
        return self._literal_radio

    def regex_radio(self) -> ttk.Radiobutton:
        return self._regex_radio

    def case_check(self) -> ttk.Checkbutton:
        return self._case_check

    def whole_words_check(self) -> ttk.Checkbutton:
        return self._whole_check

    def phonetic_check(self) -> ttk.Checkbutton:
        return self._phonetic_check


def _rule_label(rule: DictionaryRule) -> str:
    if isinstance(rule, LiteralRule):
        source = ", ".join(_quote_if_glued(alias) for alias in rule.spoken)
        learned = "  (learned)" if rule.learned else ""
        return f"[literal] {source}  ->  {rule.replacement}{learned}"
    return f"[regex] {_quote_if_glued(rule.pattern)}  ->  {rule.replacement}"


def _rule_columns(rule: DictionaryRule) -> tuple[str, str, str]:
    if isinstance(rule, LiteralRule):
        spoken = ", ".join(_quote_if_glued(alias) for alias in rule.spoken)
        return spoken, rule.replacement, "learned" if rule.learned else ""
    return _quote_if_glued(rule.pattern), rule.replacement, "regex"


def _plain_aliases(rule: LiteralRule) -> set[str]:
    return {alias.strip() for alias in rule.spoken}


def _split_aliases(spoken: str) -> tuple[str, ...]:
    glue_left, glue_right = _edge_whitespace(spoken)
    return tuple(
        glue_left + alias.strip() + glue_right
        for alias in spoken.split(",")
        if alias.strip()
    )


def _join_aliases(aliases: tuple[str, ...]) -> str:
    glue_left, _ = _edge_whitespace(aliases[0])
    _, glue_right = _edge_whitespace(aliases[-1])
    return glue_left + ", ".join(alias.strip() for alias in aliases) + glue_right


def _edge_whitespace(text: str) -> tuple[str, str]:
    return text[: len(text) - len(text.lstrip())], text[len(text.rstrip()):]


def _quote_if_glued(source: str) -> str:
    return source if source == source.strip() else f'"{source}"'
