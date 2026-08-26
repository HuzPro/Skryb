import re
import tkinter
from dataclasses import replace
from tkinter import ttk

from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryOptions,
    DictionaryRule,
    LiteralRule,
    RegexRule,
)
from voicetotext.settings.theme import HEADING_STYLE

_HINT = (
    "Replace spoken words or phrases with typed text. Separate literal aliases "
    "with commas; regex rules use one pattern."
)
_EMPTY_SPOKEN_ERROR = "At least one spoken alias or regex pattern is required."
_INVALID_REGEX_ERROR = "The regex pattern or replacement is invalid: {}"
_NO_SELECTION_ERROR = "Select a rule from the list first."
_DUPLICATE_ERROR = "Another literal rule already uses one of those aliases."
_NOTHING_SELECTED = -1


class TkDictionaryPage(ttk.Frame):
    def __init__(self, parent: tkinter.Misc) -> None:
        super().__init__(parent)
        self._dictionary = DictionaryConfig()
        self._selected_index = _NOTHING_SELECTED
        self._dirty = False
        self._remove_fillers_var = tkinter.BooleanVar(master=self)
        self._spoken_commands_var = tkinter.BooleanVar(master=self)
        self._rule_type_var = tkinter.StringVar(master=self, value="literal")
        self._spoken_var = tkinter.StringVar(master=self)
        self._replacement_var = tkinter.StringVar(master=self)
        self._case_sensitive_var = tkinter.BooleanVar(master=self)
        self._whole_words_var = tkinter.BooleanVar(master=self, value=True)
        self._message_var = tkinter.StringVar(master=self)
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Dictionary", style=HEADING_STYLE).pack(anchor="w")
        ttk.Label(self, text=_HINT, wraplength=520, justify="left").pack(
            anchor="w", pady=(2, 6),
        )
        self._build_options()

        self._listbox = tkinter.Listbox(self, height=9, exportselection=False)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox.pack(fill="both", expand=True, pady=(6, 0))

        self._build_editor()
        ttk.Label(self, textvariable=self._message_var, wraplength=520).pack(
            anchor="w", pady=(5, 0),
        )

    def _build_options(self) -> None:
        options = ttk.Frame(self)
        options.pack(fill="x")
        self._fillers_check = ttk.Checkbutton(
            options,
            text="Remove fillers (um, uh, uhh, umm, erm)",
            variable=self._remove_fillers_var,
            command=self._mark_dirty,
        )
        self._fillers_check.pack(side="left")
        self._commands_check = ttk.Checkbutton(
            options,
            text="Spoken commands",
            variable=self._spoken_commands_var,
            command=self._mark_dirty,
        )
        self._commands_check.pack(side="left", padx=(16, 0))

    def _build_editor(self) -> None:
        editor = ttk.Frame(self)
        editor.pack(fill="x", pady=(7, 0))
        editor.columnconfigure(1, weight=1)

        ttk.Label(editor, text="Type").grid(row=0, column=0, sticky="w")
        type_frame = ttk.Frame(editor)
        type_frame.grid(row=0, column=1, sticky="w")
        self._literal_radio = ttk.Radiobutton(
            type_frame,
            text="Literal",
            value="literal",
            variable=self._rule_type_var,
            command=self._sync_rule_type,
        )
        self._literal_radio.pack(side="left")
        self._regex_radio = ttk.Radiobutton(
            type_frame,
            text="Regex",
            value="regex",
            variable=self._rule_type_var,
            command=self._sync_rule_type,
        )
        self._regex_radio.pack(side="left", padx=(8, 0))

        ttk.Label(editor, text="Aliases / pattern").grid(
            row=1, column=0, sticky="w", pady=(4, 0),
        )
        self._spoken_entry = ttk.Entry(editor, textvariable=self._spoken_var)
        self._spoken_entry.grid(
            row=1, column=1, columnspan=4, sticky="ew", pady=(4, 0),
        )

        ttk.Label(editor, text="Replacement").grid(
            row=2, column=0, sticky="w", pady=(4, 0),
        )
        self._replacement_entry = ttk.Entry(
            editor, textvariable=self._replacement_var,
        )
        self._replacement_entry.grid(
            row=2, column=1, columnspan=4, sticky="ew", pady=(4, 0),
        )

        self._case_check = ttk.Checkbutton(
            editor, text="Case sensitive", variable=self._case_sensitive_var,
        )
        self._case_check.grid(row=3, column=1, sticky="w", pady=(4, 0))
        self._whole_check = ttk.Checkbutton(
            editor, text="Whole words", variable=self._whole_words_var,
        )
        self._whole_check.grid(row=3, column=2, sticky="w", pady=(4, 0))

        self._add_button = ttk.Button(editor, text="Add", command=self.add_rule)
        self._add_button.grid(row=4, column=1, sticky="w", pady=(5, 0))
        self._update_button = ttk.Button(
            editor, text="Update", command=self.update_selected_rule,
        )
        self._update_button.grid(row=4, column=2, sticky="w", pady=(5, 0))
        self._delete_button = ttk.Button(
            editor, text="Delete", command=self.delete_selected_rule,
        )
        self._delete_button.grid(row=4, column=3, sticky="w", pady=(5, 0))

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
        del rules[self._selected_index]
        self._dictionary = replace(self._dictionary, rules=tuple(rules))
        self._commit_edit()

    def _rule_from_editor(
        self, original: DictionaryRule | None = None,
    ) -> DictionaryRule | None:
        spoken = self._spoken_var.get().strip()
        if not spoken:
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
        aliases = tuple(alias.strip() for alias in spoken.split(",") if alias.strip())
        if not aliases:
            self._message_var.set(_EMPTY_SPOKEN_ERROR)
            return None
        enabled = original.enabled if isinstance(original, LiteralRule) else True
        phonetic = original.phonetic if isinstance(original, LiteralRule) else False
        return LiteralRule(
            spoken=aliases,
            replacement=replacement,
            case_sensitive=self._case_sensitive_var.get(),
            whole_words=self._whole_words_var.get(),
            phonetic=phonetic,
            enabled=enabled,
        )

    def _has_duplicate_alias(
        self, candidate: DictionaryRule, ignored_index: int = _NOTHING_SELECTED,
    ) -> bool:
        if not isinstance(candidate, LiteralRule):
            return False
        aliases = set(candidate.spoken)
        duplicate = any(
            index != ignored_index
            and isinstance(rule, LiteralRule)
            and bool(aliases.intersection(rule.spoken))
            for index, rule in enumerate(self._dictionary.rules)
        )
        if duplicate:
            self._message_var.set(_DUPLICATE_ERROR)
        return duplicate

    def _commit_edit(self) -> None:
        self._dirty = True
        self._clear_editor()
        self._refresh_listbox()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_options()

    def _update_options(self) -> None:
        current = self._dictionary.options
        self._dictionary = replace(
            self._dictionary,
            options=DictionaryOptions(
                remove_fillers=self._remove_fillers_var.get(),
                spoken_commands=self._spoken_commands_var.get(),
                numbers=current.numbers,
                punctuation_model=current.punctuation_model,
            ),
        )

    def _clear_editor(self) -> None:
        self._selected_index = _NOTHING_SELECTED
        self._rule_type_var.set("literal")
        self._spoken_var.set("")
        self._replacement_var.set("")
        self._case_sensitive_var.set(False)
        self._whole_words_var.set(True)
        self._message_var.set("")
        self._sync_rule_type()

    def _refresh_listbox(self) -> None:
        self._listbox.delete(0, "end")
        for rule in self._dictionary.rules:
            self._listbox.insert("end", _rule_label(rule))

    def _on_listbox_select(self, _event: tkinter.Event) -> None:
        selection = self._listbox.curselection()
        if selection:
            self.select_rule(selection[0])

    def select_rule(self, index: int) -> None:
        self._selected_index = index
        rule = self._dictionary.rules[index]
        if isinstance(rule, LiteralRule):
            self._rule_type_var.set("literal")
            self._spoken_var.set(", ".join(rule.spoken))
            self._case_sensitive_var.set(rule.case_sensitive)
            self._whole_words_var.set(rule.whole_words)
        else:
            self._rule_type_var.set("regex")
            self._spoken_var.set(rule.pattern)
            self._case_sensitive_var.set(False)
            self._whole_words_var.set(False)
        self._replacement_var.set(rule.replacement)
        self._message_var.set("")
        self._sync_rule_type()

    def _sync_rule_type(self) -> None:
        state = "!disabled" if self._rule_type_var.get() == "literal" else "disabled"
        self._case_check.state([state])
        self._whole_check.state([state])

    def dictionary(self) -> DictionaryConfig:
        self._update_options()
        return self._dictionary

    def set_dictionary(self, dictionary: DictionaryConfig) -> None:
        self._dictionary = dictionary
        self._remove_fillers_var.set(dictionary.options.remove_fillers)
        self._spoken_commands_var.set(dictionary.options.spoken_commands)
        self._dirty = False
        self._clear_editor()
        self._refresh_listbox()

    def is_dirty(self) -> bool:
        return self._dirty

    def rule_labels(self) -> list[str]:
        return [self._listbox.get(i) for i in range(self._listbox.size())]

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

    def fillers_check(self) -> ttk.Checkbutton:
        return self._fillers_check

    def commands_check(self) -> ttk.Checkbutton:
        return self._commands_check

    def literal_radio(self) -> ttk.Radiobutton:
        return self._literal_radio

    def regex_radio(self) -> ttk.Radiobutton:
        return self._regex_radio

    def case_check(self) -> ttk.Checkbutton:
        return self._case_check

    def whole_words_check(self) -> ttk.Checkbutton:
        return self._whole_check


def _rule_label(rule: DictionaryRule) -> str:
    if isinstance(rule, LiteralRule):
        source = ", ".join(rule.spoken)
        return f"[literal] {source}  ->  {rule.replacement}"
    return f"[regex] {rule.pattern}  ->  {rule.replacement}"
