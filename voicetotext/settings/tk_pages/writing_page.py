import tkinter
from collections.abc import Callable
from dataclasses import replace
from tkinter import filedialog, ttk

from voicetotext.settings.theme import (
    BODY_FONT,
    CARD_BODY_STYLE,
    CARD_MUTED_STYLE,
    CARD_PADDING,
    CARD_TEXT_STYLE,
    CARD_TEXT_WRAP,
    CARD_TITLE_STYLE,
    CONTROL_GAP,
    DANGER_BUTTON_STYLE,
    LINK_BUTTON_STYLE,
    PAGE_LINK_BUTTON_STYLE,
    PRIMARY_BUTTON_STYLE,
    RADIO_TITLE_STYLE,
    THIN_PROGRESS_STYLE,
    RowGroup,
    card,
    combobox,
    divider,
    hide_while_empty,
    keep_scroll_position,
    page_note,
    section_heading,
    set_card_selected,
    switch,
)
from voicetotext.writing.install import INSTALL_BYTES
from voicetotext.writing.llama_server import last_line
from voicetotext.writing.service import WritingService, WritingStatus
from voicetotext.writing.settings import (
    MAX_DEADLINE_SECONDS,
    MIN_DEADLINE_SECONDS,
    WritingSettings,
    clamp_deadline,
)
from voicetotext.writing.transforms import BUILTIN_TRANSFORMS, valid_transforms

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]

_GB = 1_000_000_000
_NO_TRANSFORM = "None"
_CLEANUP_CHOICES = (
    ("off", "Off", "Paste exactly what was recognised."),
    ("light", "Light", "Fix fillers, slips and changes of mind, keep my wording."),
    ("polished", "Polished", "Also tidy grammar and repetition."),
)
_CLEANUP_WRAP = 140
_RADIO_INDENT = 22
_STYLE_PLACEHOLDER = "e.g. British spelling, no exclamation marks, sign off with Cheers"
_LEARNING_NOTE = (
    "Learning watches edits you make to text this app pasted, for up to 15 "
    "minutes after each dictation. Everything stays on this device."
)
_DEADLINE_NOTE = "Then the raw dictation is pasted. Long dictations may skip cleanup when this is short."
_NO_SELECTION = "Select a transformation first."
_INCOMPLETE_TRANSFORM = "A transformation needs a name and an instruction."
_MODEL_BLURB = "Cleans up and formats your dictation on this computer. Nothing leaves it."
_INSTRUCTION_PREVIEW = 70
_ADVANCED_CLOSED = "Advanced ▸"
_ADVANCED_OPEN = "Advanced ▾"


def status_text(status: WritingStatus) -> str:
    if status.state == "not_installed":
        return "Not installed"
    if status.state == "downloading":
        share = status.bytes_done / status.bytes_total if status.bytes_total else 0.0
        return (
            f"Downloading {share:.0%} ({status.bytes_done / _GB:.1f} of "
            f"{status.bytes_total / _GB:.1f} GB)"
        )
    if status.state == "install_failed":
        return status.detail
    if status.state == "failed":
        return f"Failed: {last_line(status.detail) or 'no reason was given'}"
    if status.state == "starting":
        return "Starting..."
    if status.state == "ready":
        if status.last_cleanup_seconds is None:
            return "Ready"
        return f"Ready. Last cleanup {status.last_cleanup_seconds:.2f} s"
    return "Starts once dictation is ready."


def _status_pill(status: WritingStatus) -> tuple[str, str]:
    """The short state beside the model's name, and the colour it is drawn in."""
    if status.state == "downloading":
        share = status.bytes_done / status.bytes_total if status.bytes_total else 0.0
        return f"Installing {share:.0%}", "CardAccent.TLabel"
    if status.state == "starting":
        return "Starting", "CardAccent.TLabel"
    if status.state == "ready":
        return "Ready", "CardSuccess.TLabel"
    if status.state in ("failed", "install_failed"):
        return "Failed", "CardDanger.TLabel"
    if status.state == "not_installed":
        return "Not installed", "CardMuted.TLabel"
    return "Waiting", "CardMuted.TLabel"


def _status_detail(status: WritingStatus, line: str, pill: str) -> str:
    """The line under the model's name, saying what the pill beside it does not."""
    if status.state == "ready" and status.last_cleanup_seconds is not None:
        return f"Last cleanup took {status.last_cleanup_seconds:.2f} s."
    return _MODEL_BLURB if line in (pill, "Ready") else line


class TkWritingPage(ttk.Frame):
    """Settings > Writing: the local writing model and what it does.

    Settings are staged until Save & Apply like every other page; the model
    buttons (install, cancel, retry) and the learning resets act at once.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        writing: WritingService,
        schedule: ScheduleOnGuiThread,
    ) -> None:
        super().__init__(parent)
        self._writing = writing
        self._schedule = schedule
        self._cleanup = tkinter.StringVar(master=self, value="light")
        self._lists = tkinter.BooleanVar(master=self, value=True)
        self._deadline = tkinter.StringVar(master=self, value="4.0")
        self._context = tkinter.BooleanVar(master=self, value=True)
        self._learn_words = tkinter.BooleanVar(master=self, value=True)
        self._learn_style = tkinter.BooleanVar(master=self, value=True)
        self._runtime_path = tkinter.StringVar(master=self)
        self._model_path = tkinter.StringVar(master=self)
        self._transform_choice = tkinter.StringVar(master=self, value=_NO_TRANSFORM)
        self._transform_name = tkinter.StringVar(master=self)
        self._transform_text = tkinter.StringVar(master=self)
        self._message = tkinter.StringVar(master=self)
        self._status = tkinter.StringVar(master=self)
        self._learned = tkinter.StringVar(master=self)
        self._transforms: dict[str, str] = writing.transforms()
        self._transforms_dirty = False
        self._selected_transform = ""
        self._placeholder_shown = False
        self._shown_state = ""
        self._cleanup_cards: dict[str, ttk.Frame] = {}
        self._transform_rows: dict[str, ttk.Frame] = {}
        self._build()
        self._refresh_transforms()
        self._refresh_learned()
        self._refresh_status()
        self._listener = self._on_writing_changed
        writing.subscribe(self._listener)

    # -- construction ----------------------------------------------------

    def _build(self) -> None:
        self._build_model_group()
        self._build_cleanup_group()
        self._build_learning_group()
        self._build_transform_group()

    def _build_model_group(self) -> None:
        section_heading(self, "Writing model", first=True)
        self._model_card = card(self)
        self._model_card.pack(fill="x")
        self._model_card.columnconfigure(0, weight=1)
        header = ttk.Frame(self._model_card, style=CARD_BODY_STYLE)
        header.grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Local writing model", style=CARD_TITLE_STYLE).pack(side="left")
        self._pill = ttk.Label(header, text="", style=CARD_MUTED_STYLE)
        self._pill.pack(side="left", padx=(10, 0))
        self._detail = ttk.Label(
            self._model_card, text="", style=CARD_MUTED_STYLE, wraplength=360, justify="left",
        )
        self._detail.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._buttons_row = ttk.Frame(self._model_card, style=CARD_BODY_STYLE)
        self._buttons_row.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))
        self._action_button = ttk.Button(
            self._buttons_row, style=PRIMARY_BUTTON_STYLE, command=self._on_action,
        )
        self._log_button = ttk.Button(
            self._buttons_row, text="Open log", command=self._writing.open_log,
        )
        self._progress = ttk.Progressbar(self, maximum=1.0, style=THIN_PROGRESS_STYLE)

        self._advanced_button = ttk.Button(
            self, text=_ADVANCED_CLOSED, style=PAGE_LINK_BUTTON_STYLE,
            command=self._toggle_advanced,
        )
        self._advanced_button.pack(anchor="w", pady=(10, 0))
        self._advanced = card(self)
        self._advanced.columnconfigure(1, weight=1)
        self._path_row(0, "llama-server.exe", self._runtime_path, "Programs", "*.exe")
        self._path_row(1, "Model (.gguf)", self._model_path, "GGUF models", "*.gguf")
        ttk.Label(
            self._advanced, text="Leave empty to use the installed ones.",
            style=CARD_MUTED_STYLE,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _path_row(
        self,
        row: int,
        label: str,
        variable: tkinter.StringVar,
        kind: str,
        pattern: str,
    ) -> None:
        ttk.Label(self._advanced, text=label, style=CARD_TEXT_STYLE).grid(
            row=row, column=0, sticky="w", pady=4,
        )
        ttk.Entry(self._advanced, textvariable=variable, font=BODY_FONT).grid(
            row=row, column=1, sticky="ew", padx=(12, CONTROL_GAP), pady=4,
        )
        ttk.Button(
            self._advanced,
            text="Browse...",
            command=lambda: self._browse(variable, kind, pattern),
        ).grid(row=row, column=2, pady=4)

    def _build_cleanup_group(self) -> None:
        section_heading(self, "Cleanup")
        choices = ttk.Frame(self)
        choices.pack(fill="x")
        choices.columnconfigure(tuple(range(len(_CLEANUP_CHOICES))), weight=1, uniform="cleanup")
        self._cleanup_radios: dict[str, ttk.Radiobutton] = {}
        for column, (value, label, description) in enumerate(_CLEANUP_CHOICES):
            choice = card(choices)
            choice.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 10, 0))
            radio = ttk.Radiobutton(
                choice, text=label, value=value, variable=self._cleanup,
                style=RADIO_TITLE_STYLE,
            )
            radio.pack(anchor="w")
            hint = ttk.Label(
                choice, text=description, style=CARD_MUTED_STYLE,
                wraplength=_CLEANUP_WRAP, justify="left",
            )
            hint.pack(anchor="w", padx=(_RADIO_INDENT, 0), pady=(2, 0))
            for widget in (choice, hint):
                widget.bind("<Button-1>", lambda _event, r=radio: r.invoke())
            self._cleanup_radios[value] = radio
            self._cleanup_cards[value] = choice
        self._cleanup.trace_add("write", lambda *_args: self._refresh_cleanup_cards())
        self._refresh_cleanup_cards()

        rows = RowGroup(self)
        rows.pack(fill="x", pady=(12, 0))
        rows.add(
            "Turn spoken lists into lists",
            "Numbered steps and bullet points when you dictate them.",
            lambda parent: switch(parent, self._lists),
        )
        deadline = rows.add("Give up after", _DEADLINE_NOTE, self._deadline_control)
        self._deadline_box = deadline.control
        rows.add(
            "Use the text around the cursor",
            "And the window title, to spell names and terms. Never read from "
            "password fields or password managers.",
            lambda parent: switch(parent, self._context),
        )

    def _deadline_control(self, parent: tkinter.Misc) -> ttk.Frame:
        control = ttk.Frame(parent, style=CARD_BODY_STYLE)
        ttk.Spinbox(
            control,
            from_=MIN_DEADLINE_SECONDS,
            to=MAX_DEADLINE_SECONDS,
            increment=0.5,
            width=5,
            textvariable=self._deadline,
            font=BODY_FONT,
        ).pack(side="left")
        ttk.Label(control, text="seconds", style=CARD_TEXT_STYLE).pack(side="left", padx=(6, 0))
        return control

    def _build_learning_group(self) -> None:
        section_heading(self, "Learning")
        rows = RowGroup(self)
        rows.pack(fill="x")
        rows.add(
            "Learn words I correct",
            "Names and terms you fix after pasting are spelled right next time.",
            lambda parent: switch(parent, self._learn_words),
        )
        rows.add(
            "Learn my writing style",
            "How you edit what was pasted shapes the next cleanup.",
            lambda parent: switch(parent, self._learn_style),
        )
        rows.add_widget(self._style_notes_block)
        rows.add_widget(self._learned_block)
        page_note(self, _LEARNING_NOTE).pack(anchor="w", pady=(CONTROL_GAP, 0))

    def _style_notes_block(self, parent: tkinter.Misc) -> ttk.Frame:
        block = ttk.Frame(parent, style=CARD_BODY_STYLE)
        ttk.Label(block, text="Style notes", style=CARD_TITLE_STYLE).pack(anchor="w")
        ttk.Label(
            block, text="Tell the writing model how you like to write.",
            style=CARD_MUTED_STYLE,
        ).pack(anchor="w", pady=(2, CONTROL_GAP))
        self._style_notes = tkinter.Text(block, height=4, width=40, wrap="word")
        self._style_notes.pack(fill="x")
        self._style_notes.tag_configure("placeholder", foreground="#8a8a93")
        self._style_notes.bind("<FocusIn>", self._hide_placeholder)
        self._style_notes.bind("<FocusOut>", self._show_placeholder_if_empty)
        return block

    def _learned_block(self, parent: tkinter.Misc) -> ttk.Frame:
        block = ttk.Frame(parent, style=CARD_BODY_STYLE)
        block.columnconfigure(0, weight=1)
        ttk.Label(
            block, textvariable=self._learned, style=CARD_MUTED_STYLE,
            wraplength=300, justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(block, text="Forget learned style", command=self._forget_style).grid(
            row=0, column=1, padx=(16, CONTROL_GAP),
        )
        ttk.Button(
            block, text="Reset learning", style=DANGER_BUTTON_STYLE,
            command=self._reset_learning,
        ).grid(row=0, column=2)
        return block

    def _build_transform_group(self) -> None:
        section_heading(self, "Transformations")
        page_note(
            self, "Instructions the writing model can apply to a whole dictation.",
        ).pack(anchor="w", pady=(0, CONTROL_GAP))
        self._transform_list = card(self, padding=(0, 4))
        self._transform_list.pack(fill="x")

        editor = card(self)
        editor.pack(fill="x", pady=(CONTROL_GAP, 0))
        editor.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate(
            (("Name", self._transform_name), ("Instruction", self._transform_text)),
        ):
            ttk.Label(editor, text=label, style=CARD_TEXT_STYLE).grid(
                row=row, column=0, sticky="w", pady=4,
            )
            ttk.Entry(editor, textvariable=variable, font=BODY_FONT).grid(
                row=row, column=1, sticky="ew", padx=(12, 0), pady=4,
            )
        buttons = ttk.Frame(editor, style=CARD_BODY_STYLE)
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(CONTROL_GAP, 0))
        for text, command, style in (
            ("Add", self.add_transform, "TButton"),
            ("Save changes", self.edit_transform, "TButton"),
            ("Remove", self.remove_transform, DANGER_BUTTON_STYLE),
        ):
            ttk.Button(buttons, text=text, command=command, style=style).pack(
                side="left", padx=(0, CONTROL_GAP),
            )
        ttk.Button(
            buttons, text="Restore built-ins", style=LINK_BUTTON_STYLE,
            command=self.restore_builtins,
        ).pack(side="right")
        message = ttk.Label(
            editor, textvariable=self._message, style="CardDanger.TLabel",
            wraplength=CARD_TEXT_WRAP,
        )
        message.grid(row=3, column=0, columnspan=2, sticky="w", pady=(CONTROL_GAP, 0))
        hide_while_empty(message, self._message)

        rows = RowGroup(self)
        rows.pack(fill="x", pady=(CONTROL_GAP, 0))
        after = rows.add(
            "After each dictation, apply",
            "Runs after cleanup, every time.",
            lambda parent: combobox(parent, textvariable=self._transform_choice),
            fill=True,
        )
        assert isinstance(after.control, ttk.Combobox)
        self._transform_combo = after.control

    # -- staged settings -------------------------------------------------

    def settings(self) -> WritingSettings:
        try:
            deadline = clamp_deadline(float(self._deadline.get()))
        except ValueError:
            deadline = WritingSettings().deadline_seconds
        choice = self._transform_choice.get()
        return WritingSettings(
            cleanup=self._cleanup.get(),
            lists=self._lists.get(),
            deadline_seconds=deadline,
            context=self._context.get(),
            learn_words=self._learn_words.get(),
            learn_style=self._learn_style.get(),
            style_notes=self.style_notes(),
            after_dictation_transform="" if choice == _NO_TRANSFORM else choice,
            runtime_path=self._runtime_path.get().strip(),
            model_path=self._model_path.get().strip(),
        )

    def set_settings(self, settings: WritingSettings) -> None:
        self._cleanup.set(settings.cleanup)
        self._lists.set(settings.lists)
        self._deadline.set(f"{settings.deadline_seconds:g}")
        self._context.set(settings.context)
        self._learn_words.set(settings.learn_words)
        self._learn_style.set(settings.learn_style)
        self._style_notes.delete("1.0", "end")
        self._placeholder_shown = False
        if settings.style_notes:
            self._style_notes.insert("1.0", settings.style_notes)
        else:
            self._show_placeholder_if_empty()
        chosen = settings.after_dictation_transform
        self._transform_choice.set(chosen if chosen in self._transforms else _NO_TRANSFORM)
        self._runtime_path.set(settings.runtime_path)
        self._model_path.set(settings.model_path)
        if (settings.runtime_path or settings.model_path) and not self._advanced_open():
            self._toggle_advanced()

    def style_notes(self) -> str:
        if self._placeholder_shown:
            return ""
        return self._style_notes.get("1.0", "end-1c").strip()

    def transforms_dirty(self) -> bool:
        return self._transforms_dirty

    def transforms(self) -> dict[str, str]:
        return dict(self._transforms)

    # -- transformations ---------------------------------------------------

    def add_transform(self) -> None:
        name, instruction = self._transform_name.get().strip(), self._transform_text.get().strip()
        if not name or not instruction:
            self._message.set(_INCOMPLETE_TRANSFORM)
            return
        self._transforms[name] = instruction
        self._commit_transforms()

    def edit_transform(self) -> None:
        if not self._selected_transform:
            self._message.set(_NO_SELECTION)
            return
        name, instruction = self._transform_name.get().strip(), self._transform_text.get().strip()
        if not name or not instruction:
            self._message.set(_INCOMPLETE_TRANSFORM)
            return
        renamed = {
            (name if key == self._selected_transform else key): (
                instruction if key == self._selected_transform else value
            )
            for key, value in self._transforms.items()
        }
        if self._transform_choice.get() == self._selected_transform:
            self._transform_choice.set(name)
        self._transforms = renamed
        self._commit_transforms()

    def remove_transform(self) -> None:
        if not self._selected_transform:
            self._message.set(_NO_SELECTION)
            return
        self._transforms.pop(self._selected_transform, None)
        self._commit_transforms()

    def restore_builtins(self) -> None:
        self._transforms.update(BUILTIN_TRANSFORMS)
        self._commit_transforms()

    def select_transform(self, name: str) -> None:
        self._selected_transform = name
        self._transform_name.set(name)
        self._transform_text.set(self._transforms.get(name, ""))
        self._message.set("")
        for row_name, row in self._transform_rows.items():
            set_card_selected(row, row_name == name)

    def _commit_transforms(self) -> None:
        self._transforms = valid_transforms(self._transforms)
        self._transforms_dirty = True
        self._selected_transform = ""
        self._transform_name.set("")
        self._transform_text.set("")
        self._message.set("")
        keep_scroll_position(self, self._refresh_transforms)

    def _refresh_transforms(self) -> None:
        for child in self._transform_list.winfo_children():
            child.destroy()
        self._transform_rows = {}
        for index, (name, instruction) in enumerate(self._transforms.items()):
            if index:
                divider(self._transform_list).pack(fill="x")
            self._transform_rows[name] = self._transform_row(name, instruction)
        self._transform_combo["values"] = [_NO_TRANSFORM, *self._transforms]
        if self._transform_choice.get() not in self._transforms:
            self._transform_choice.set(_NO_TRANSFORM)

    def _transform_row(self, name: str, instruction: str) -> ttk.Frame:
        row = ttk.Frame(
            self._transform_list, style=CARD_BODY_STYLE, padding=(CARD_PADDING, 8),
            cursor="hand2",
        )
        row.pack(fill="x")
        ttk.Label(row, text=name, style=CARD_TITLE_STYLE).pack(anchor="w")
        preview = instruction if len(instruction) <= _INSTRUCTION_PREVIEW else (
            instruction[: _INSTRUCTION_PREVIEW - 1] + "…"
        )
        ttk.Label(row, text=preview, style=CARD_MUTED_STYLE).pack(anchor="w")
        for widget in (row, *row.winfo_children()):
            widget.bind("<Button-1>", lambda _event, chosen=name: self.select_transform(chosen))
        return row

    def _refresh_cleanup_cards(self) -> None:
        for value, choice in self._cleanup_cards.items():
            set_card_selected(choice, value == self._cleanup.get())

    # -- model status ------------------------------------------------------

    def _on_writing_changed(self) -> None:
        self._schedule(self._refresh_status)

    def _refresh_status(self) -> None:
        if not self.winfo_exists():
            return
        status = self._writing.status()
        line = status_text(status)
        pill, pill_style = _status_pill(status)
        self._status.set(line)
        self._pill.configure(text=f"● {pill}", style=pill_style)
        self._detail.configure(text=_status_detail(status, line, pill))
        if status.bytes_total:
            self._progress["value"] = status.bytes_done / status.bytes_total
        # A download reports thousands of times; only a new state moves widgets.
        if status.state == self._shown_state:
            return
        self._shown_state = status.state
        if status.state == "downloading":
            self._progress.pack(fill="x", pady=(CONTROL_GAP, 0), after=self._model_card)
        else:
            self._progress.pack_forget()
        self._action_button.pack_forget()
        self._log_button.pack_forget()
        action = _action_label(status.state)
        if action:
            self._action_button.configure(
                text=action,
                style="TButton" if status.state == "downloading" else PRIMARY_BUTTON_STYLE,
            )
            self._action_button.pack(side="right")
        if status.state in ("failed", "install_failed"):
            self._log_button.pack(side="right", padx=(0, CONTROL_GAP))

    def _on_action(self) -> None:
        state = self._writing.status().state
        if state == "not_installed":
            self._writing.install()
        elif state == "downloading":
            self._writing.cancel_install()
        elif state in ("failed", "install_failed"):
            self._writing.retry()
        self._refresh_status()

    def _advanced_open(self) -> bool:
        return bool(self._advanced.winfo_manager())

    def _toggle_advanced(self) -> None:
        if self._advanced_open():
            self._advanced.pack_forget()
            self._advanced_button.configure(text=_ADVANCED_CLOSED)
        else:
            self._advanced.pack(fill="x", pady=(4, 0), after=self._advanced_button)
            self._advanced_button.configure(text=_ADVANCED_OPEN)

    def _browse(self, variable: tkinter.StringVar, kind: str, pattern: str) -> None:
        chosen = filedialog.askopenfilename(
            parent=self, filetypes=((kind, pattern), ("All files", "*.*")),
        )
        if chosen:
            variable.set(chosen)

    # -- learning ------------------------------------------------------------

    def _refresh_learned(self) -> None:
        self._learned.set(f"Learned so far: {self._writing.learned_style_summary()}")

    def _forget_style(self) -> None:
        self._writing.forget_learned_style()
        self._refresh_learned()

    def _reset_learning(self) -> None:
        self._writing.reset_learning()
        self._refresh_learned()

    def _hide_placeholder(self, _event: tkinter.Event | None = None) -> None:
        if self._placeholder_shown:
            self._style_notes.delete("1.0", "end")
            self._placeholder_shown = False

    def _show_placeholder_if_empty(self, _event: tkinter.Event | None = None) -> None:
        if self._style_notes.get("1.0", "end-1c").strip():
            return
        self._style_notes.delete("1.0", "end")
        self._style_notes.insert("1.0", _STYLE_PLACEHOLDER, "placeholder")
        self._placeholder_shown = True

    def destroy(self) -> None:
        self._writing.unsubscribe(self._listener)
        super().destroy()

    # -- test accessors ------------------------------------------------------

    def status_line(self) -> str:
        return self._status.get()

    def action_label(self) -> str:
        return str(self._action_button.cget("text")) if self._action_button.winfo_manager() else ""

    def log_button_shown(self) -> bool:
        return bool(self._log_button.winfo_manager())

    def progress_shown(self) -> bool:
        return bool(self._progress.winfo_manager())

    def transform_names(self) -> list[str]:
        return list(self._transforms)

    def learned_line(self) -> str:
        return self._learned.get()

    def set_style_notes(self, text: str) -> None:
        self.set_settings(replace(self.settings(), style_notes=text))

    def press_action(self) -> None:
        self._on_action()


def _action_label(state: str) -> str:
    if state == "not_installed":
        return f"Install ({INSTALL_BYTES / _GB:.1f} GB)"
    if state == "downloading":
        return "Cancel"
    if state in ("failed", "install_failed"):
        return "Retry"
    return ""
