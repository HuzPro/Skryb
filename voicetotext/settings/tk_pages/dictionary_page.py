import tkinter
from tkinter import ttk

from voicetotext.settings.theme import HEADING_STYLE

_HINT = (
    "Replace a single spoken word with the text you want typed. "
    "Phrases are not supported."
)
_EMPTY_ERROR = "Both the spoken word and its replacement are required."
_PHRASE_ERROR = "The spoken word must be one word, without spaces."
_NO_SELECTION_ERROR = "Select a rule from the list first."
_DUPLICATE_ERROR = "There is already a rule for that spoken word."
_ARROW = "  ->  "
_NOTHING_SELECTED = ""


def _rule_error(spoken: str, replacement: str) -> str:
    """Return why the pair is unusable, or an empty string when it is valid.

    Mirrors the constraints ``load_dictionary`` enforces when reading
    ``dictionary.json``, so the UI cannot stage a rule the loader would
    later drop.
    """
    if not spoken or not replacement:
        return _EMPTY_ERROR
    if any(character.isspace() for character in spoken):
        return _PHRASE_ERROR
    return ""


class TkDictionaryPage(ttk.Frame):
    """Page for editing the personal spoken-word replacement rules.

    Edits are staged in memory and reported through :meth:`is_dirty`; the
    enclosing window persists them through the settings service when the
    user saves.
    """

    def __init__(self, parent: tkinter.Misc) -> None:
        super().__init__(parent)
        self._rules: dict[str, str] = {}
        self._selected_spoken = _NOTHING_SELECTED
        self._dirty = False
        self._spoken_var = tkinter.StringVar(master=self)
        self._replacement_var = tkinter.StringVar(master=self)
        self._message_var = tkinter.StringVar(master=self)
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Dictionary", style=HEADING_STYLE).pack(anchor="w")
        ttk.Label(self, text=_HINT, wraplength=480, justify="left").pack(
            anchor="w", pady=(2, 8),
        )

        self._listbox = tkinter.Listbox(self, height=12, exportselection=False)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox.pack(fill="both", expand=True)

        self._build_editor()

        ttk.Label(self, textvariable=self._message_var, wraplength=480).pack(
            anchor="w", pady=(6, 0),
        )

    def _build_editor(self) -> None:
        editor = ttk.Frame(self)
        editor.pack(fill="x", pady=(8, 0))

        self._spoken_entry = ttk.Entry(
            editor, textvariable=self._spoken_var, width=18,
        )
        self._spoken_entry.pack(side="left")
        ttk.Label(editor, text="->").pack(side="left", padx=6)
        self._replacement_entry = ttk.Entry(
            editor, textvariable=self._replacement_var, width=18,
        )
        self._replacement_entry.pack(side="left")

        self._add_button = ttk.Button(editor, text="Add", command=self.add_rule)
        self._add_button.pack(side="left", padx=(8, 0))
        self._update_button = ttk.Button(
            editor, text="Update", command=self.update_selected_rule,
        )
        self._update_button.pack(side="left", padx=(4, 0))
        self._delete_button = ttk.Button(
            editor, text="Delete", command=self.delete_selected_rule,
        )
        self._delete_button.pack(side="left", padx=(4, 0))

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def add_rule(self) -> None:
        spoken = self._spoken_var.get().strip()
        replacement = self._replacement_var.get().strip()
        error = _rule_error(spoken, replacement)
        if error:
            self._message_var.set(error)
            return
        self._rules[spoken] = replacement
        self._commit_edit()

    def update_selected_rule(self) -> None:
        if self._selected_spoken == _NOTHING_SELECTED:
            self._message_var.set(_NO_SELECTION_ERROR)
            return
        spoken = self._spoken_var.get().strip()
        replacement = self._replacement_var.get().strip()
        error = _rule_error(spoken, replacement)
        if error:
            self._message_var.set(error)
            return
        if spoken != self._selected_spoken and spoken in self._rules:
            self._message_var.set(_DUPLICATE_ERROR)
            return
        self._rules = self._with_replaced(
            self._selected_spoken, spoken, replacement,
        )
        self._commit_edit()

    def delete_selected_rule(self) -> None:
        if self._selected_spoken == _NOTHING_SELECTED:
            self._message_var.set(_NO_SELECTION_ERROR)
            return
        del self._rules[self._selected_spoken]
        self._commit_edit()

    def _with_replaced(
        self, old_spoken: str, spoken: str, replacement: str,
    ) -> dict[str, str]:
        updated: dict[str, str] = {}
        for key, value in self._rules.items():
            if key == old_spoken:
                updated[spoken] = replacement
            else:
                updated[key] = value
        return updated

    def _commit_edit(self) -> None:
        self._dirty = True
        self._clear_editor()
        self._refresh_listbox()

    def _clear_editor(self) -> None:
        self._selected_spoken = _NOTHING_SELECTED
        self._spoken_var.set("")
        self._replacement_var.set("")
        self._message_var.set("")

    def _refresh_listbox(self) -> None:
        self._listbox.delete(0, "end")
        for spoken, replacement in self._rules.items():
            self._listbox.insert("end", f"{spoken}{_ARROW}{replacement}")

    def _on_listbox_select(self, _event: tkinter.Event) -> None:
        selection = self._listbox.curselection()
        if selection:
            self.select_rule(selection[0])

    def select_rule(self, index: int) -> None:
        self._selected_spoken = list(self._rules)[index]
        self._spoken_var.set(self._selected_spoken)
        self._replacement_var.set(self._rules[self._selected_spoken])
        self._message_var.set("")

    # ------------------------------------------------------------------
    # Window API
    # ------------------------------------------------------------------

    def rules(self) -> dict[str, str]:
        return dict(self._rules)

    def set_rules(self, rules: dict[str, str]) -> None:
        self._rules = dict(rules)
        self._dirty = False
        self._clear_editor()
        self._refresh_listbox()

    def is_dirty(self) -> bool:
        return self._dirty

    # ------------------------------------------------------------------
    # Test accessors
    # ------------------------------------------------------------------

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
