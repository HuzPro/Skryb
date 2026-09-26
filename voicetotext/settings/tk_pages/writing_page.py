import tkinter
from collections.abc import Callable
from dataclasses import replace
from tkinter import filedialog, ttk

from voicetotext.settings.theme import HEADING_STYLE
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
_WRAP = 380
_NO_TRANSFORM = "None"
_CLEANUP_CHOICES = (
    ("off", "Off", "Paste exactly what was recognised."),
    ("light", "Light", "Fix fillers, slips and changes of mind, keep my wording."),
    ("polished", "Polished", "Also tidy grammar and repetition."),
)
_STYLE_PLACEHOLDER = "e.g. British spelling, no exclamation marks, sign off with Cheers"
_LEARNING_NOTE = (
    "Learning watches edits you make to text this app pasted, for up to 15 "
    "minutes after each dictation. Everything stays on this device."
)
_DEADLINE_NOTE = "Long dictations may skip cleanup when the limit is short."
_NO_SELECTION = "Select a transformation first."
_INCOMPLETE_TRANSFORM = "A transformation needs a name and an instruction."


def status_text(status: WritingStatus) -> str:
    if status.state == "not_installed":
        return "Not installed"
    if status.state == "downloading":
        share = status.bytes_done / status.bytes_total if status.bytes_total else 0.0
        return (
            f"Downloading {share:.0%} ({status.bytes_done / _GB:.1f} of "
            f"{status.bytes_total / _GB:.1f} GB)"
        )
    if status.state in ("failed", "install_failed"):
        return f"Failed: {last_line(status.detail) or 'no reason was given'}"
    if status.state == "starting":
        return "Starting..."
    if status.state == "ready":
        if status.last_cleanup_seconds is None:
            return "Ready"
        return f"Ready. Last cleanup {status.last_cleanup_seconds:.2f} s"
    return "Starts once dictation is ready."


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
        self._advanced_open = False
        self._shown_state = ""
        self._build()
        self._refresh_transforms()
        self._refresh_learned()
        self._refresh_status()
        self._listener = self._on_writing_changed
        writing.subscribe(self._listener)

    # -- construction ----------------------------------------------------

    def _build(self) -> None:
        self.columnconfigure(0, weight=1, uniform="column")
        self.columnconfigure(1, weight=1, uniform="column")
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_model_group(left)
        self._build_cleanup_group(left)
        self._build_context_group(left)
        self._build_learning_group(right)
        self._build_transform_group(right)

    def _build_model_group(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Writing model", style=HEADING_STYLE).pack(anchor="w")
        ttk.Label(parent, textvariable=self._status, wraplength=_WRAP).pack(
            anchor="w", pady=(4, 0),
        )
        self._progress = ttk.Progressbar(parent, maximum=1.0, length=300)
        buttons = ttk.Frame(parent)
        buttons.pack(anchor="w", pady=(6, 0))
        self._buttons_row = buttons
        self._action_button = ttk.Button(buttons, command=self._on_action)
        self._log_button = ttk.Button(
            buttons, text="Open log", command=self._writing.open_log,
        )
        self._advanced_button = ttk.Button(
            parent, text="Advanced ▸", command=self._toggle_advanced,
        )
        self._advanced_button.pack(anchor="w", pady=(6, 0))
        self._advanced = ttk.Frame(parent)
        self._advanced.columnconfigure(1, weight=1)
        self._path_row(0, "llama-server.exe", self._runtime_path, "Programs", "*.exe")
        self._path_row(1, "Model (.gguf)", self._model_path, "GGUF models", "*.gguf")
        ttk.Label(
            self._advanced,
            text="Leave empty to use the installed ones.",
            wraplength=_WRAP,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

    def _path_row(
        self,
        row: int,
        label: str,
        variable: tkinter.StringVar,
        kind: str,
        pattern: str,
    ) -> None:
        ttk.Label(self._advanced, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(self._advanced, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=(6, 6), pady=2,
        )
        ttk.Button(
            self._advanced,
            text="Browse...",
            command=lambda: self._browse(variable, kind, pattern),
        ).grid(row=row, column=2, pady=2)

    def _build_cleanup_group(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Cleanup", style=HEADING_STYLE).pack(anchor="w", pady=(18, 4))
        self._cleanup_radios: dict[str, ttk.Radiobutton] = {}
        for value, label, description in _CLEANUP_CHOICES:
            radio = ttk.Radiobutton(parent, text=label, value=value, variable=self._cleanup)
            radio.pack(anchor="w", pady=(2, 0))
            ttk.Label(parent, text=description, wraplength=_WRAP - 24).pack(
                anchor="w", padx=(24, 0),
            )
            self._cleanup_radios[value] = radio
        ttk.Checkbutton(
            parent, text="Turn spoken lists and steps into lists", variable=self._lists,
        ).pack(anchor="w", pady=(6, 0))
        row = ttk.Frame(parent)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Label(row, text="Give up after").pack(side="left")
        self._deadline_box = ttk.Spinbox(
            row,
            from_=MIN_DEADLINE_SECONDS,
            to=MAX_DEADLINE_SECONDS,
            increment=0.5,
            width=5,
            textvariable=self._deadline,
        )
        self._deadline_box.pack(side="left", padx=4)
        ttk.Label(row, text="s and paste the raw dictation").pack(side="left")
        ttk.Label(parent, text=_DEADLINE_NOTE, wraplength=_WRAP).pack(anchor="w", pady=(2, 0))

    def _build_context_group(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Context", style=HEADING_STYLE).pack(anchor="w", pady=(18, 4))
        ttk.Checkbutton(
            parent,
            text="Use the text around the cursor and the window title\nto spell names and terms",
            variable=self._context,
        ).pack(anchor="w")
        ttk.Label(
            parent,
            text="Never read from password fields or password managers.",
            wraplength=_WRAP,
        ).pack(anchor="w", pady=(2, 0))

    def _build_learning_group(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Learning", style=HEADING_STYLE).pack(anchor="w")
        ttk.Checkbutton(
            parent, text="Learn words I correct after pasting", variable=self._learn_words,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(
            parent, text="Learn my writing style", variable=self._learn_style,
        ).pack(anchor="w")
        ttk.Label(parent, text="Style notes").pack(anchor="w", pady=(6, 2))
        self._style_notes = tkinter.Text(parent, height=3, width=40, wrap="word")
        self._style_notes.pack(fill="x")
        self._style_notes.tag_configure("placeholder", foreground="grey")
        self._style_notes.bind("<FocusIn>", self._hide_placeholder)
        self._style_notes.bind("<FocusOut>", self._show_placeholder_if_empty)
        ttk.Label(parent, textvariable=self._learned, wraplength=_WRAP).pack(
            anchor="w", pady=(6, 0),
        )
        buttons = ttk.Frame(parent)
        buttons.pack(anchor="w", pady=(4, 0))
        ttk.Button(buttons, text="Forget learned style", command=self._forget_style).pack(
            side="left",
        )
        ttk.Button(buttons, text="Reset learning", command=self._reset_learning).pack(
            side="left", padx=(8, 0),
        )
        ttk.Label(parent, text=_LEARNING_NOTE, wraplength=_WRAP).pack(anchor="w", pady=(4, 0))

    def _build_transform_group(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Transformations", style=HEADING_STYLE).pack(
            anchor="w", pady=(18, 4),
        )
        self._transform_list = tkinter.Listbox(parent, height=5, exportselection=False)
        self._transform_list.bind("<<ListboxSelect>>", self._on_transform_selected)
        self._transform_list.pack(fill="x")
        editor = ttk.Frame(parent)
        editor.pack(fill="x", pady=(4, 0))
        editor.columnconfigure(1, weight=1)
        ttk.Label(editor, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(editor, textvariable=self._transform_name).grid(
            row=0, column=1, sticky="ew", padx=(6, 0), pady=1,
        )
        ttk.Label(editor, text="Instruction").grid(row=1, column=0, sticky="w")
        ttk.Entry(editor, textvariable=self._transform_text).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=1,
        )
        buttons = ttk.Frame(parent)
        buttons.pack(anchor="w", pady=(4, 0))
        for text, command in (
            ("Add", self.add_transform),
            ("Edit", self.edit_transform),
            ("Remove", self.remove_transform),
            ("Restore built-ins", self.restore_builtins),
        ):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=(0, 6))
        ttk.Label(parent, textvariable=self._message, wraplength=_WRAP).pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Label(row, text="After each dictation, apply:").pack(side="left")
        self._transform_combo = ttk.Combobox(
            row, textvariable=self._transform_choice, state="readonly", width=22,
        )
        self._transform_combo.pack(side="left", padx=(6, 0))

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
        if (settings.runtime_path or settings.model_path) and not self._advanced_open:
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

    def _commit_transforms(self) -> None:
        self._transforms = valid_transforms(self._transforms)
        self._transforms_dirty = True
        self._selected_transform = ""
        self._transform_name.set("")
        self._transform_text.set("")
        self._message.set("")
        self._refresh_transforms()

    def _refresh_transforms(self) -> None:
        self._transform_list.delete(0, "end")
        for name in self._transforms:
            self._transform_list.insert("end", name)
        self._transform_combo["values"] = [_NO_TRANSFORM, *self._transforms]
        if self._transform_choice.get() not in self._transforms:
            self._transform_choice.set(_NO_TRANSFORM)

    def _on_transform_selected(self, _event: tkinter.Event) -> None:
        selection = self._transform_list.curselection()
        if selection:
            self.select_transform(self._transform_list.get(selection[0]))

    # -- model status ------------------------------------------------------

    def _on_writing_changed(self) -> None:
        self._schedule(self._refresh_status)

    def _refresh_status(self) -> None:
        if not self.winfo_exists():
            return
        status = self._writing.status()
        self._status.set(status_text(status))
        if status.bytes_total:
            self._progress["value"] = status.bytes_done / status.bytes_total
        # A download reports thousands of times; only a new state moves widgets.
        if status.state == self._shown_state:
            return
        self._shown_state = status.state
        if status.state == "downloading":
            self._progress.pack(anchor="w", pady=(4, 0), before=self._buttons_row)
        else:
            self._progress.pack_forget()
        self._action_button.pack_forget()
        self._log_button.pack_forget()
        action = _action_label(status.state)
        if action:
            self._action_button.configure(text=action)
            self._action_button.pack(side="left")
        if status.state == "failed":
            self._log_button.pack(side="left", padx=(8, 0))

    def _on_action(self) -> None:
        state = self._writing.status().state
        if state == "not_installed":
            self._writing.install()
        elif state == "downloading":
            self._writing.cancel_install()
        elif state in ("failed", "install_failed"):
            self._writing.retry()
        self._refresh_status()

    def _toggle_advanced(self) -> None:
        self._advanced_open = not self._advanced_open
        if self._advanced_open:
            self._advanced.pack(fill="x", pady=(4, 0), after=self._advanced_button)
            self._advanced_button.configure(text="Advanced ▾")
        else:
            self._advanced.pack_forget()
            self._advanced_button.configure(text="Advanced ▸")

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
