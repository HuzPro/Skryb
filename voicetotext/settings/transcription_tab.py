import threading

import customtkinter as ctk

from voicetotext.core.config import ConfigurationStore
from voicetotext.transcription.model_info import ModelInfo
from voicetotext.transcription.model_manager import ModelManager


class TranscriptionTab:

    def __init__(
        self,
        parent: ctk.CTkFrame,
        config: ConfigurationStore,
        model_manager: ModelManager,
    ):
        self._parent = parent
        self._config = config
        self._model_manager = model_manager
        self._model_rows: dict[str, _ModelRow] = {}
        self._active_model_var = ctk.StringVar(
            value=config.get("transcription.model_size", "base")
        )
        self._build(parent)

    def _build(self, parent: ctk.CTkFrame) -> None:
        models_label = ctk.CTkLabel(
            parent, text="Models", font=ctk.CTkFont(size=14, weight="bold")
        )
        models_label.pack(anchor="w", padx=10, pady=(10, 5))

        self._models_frame = ctk.CTkScrollableFrame(parent, height=200)
        self._models_frame.pack(fill="x", padx=10, pady=(0, 10))

        for model_info in self._model_manager.list_models():
            installed = self._model_manager.is_installed(model_info.name)
            row = _ModelRow(
                self._models_frame,
                model_info,
                installed=installed,
                active_var=self._active_model_var,
                on_download=self._start_download,
                on_delete=self._delete_model,
            )
            self._model_rows[model_info.name] = row

        settings_label = ctk.CTkLabel(
            parent, text="Settings", font=ctk.CTkFont(size=14, weight="bold")
        )
        settings_label.pack(anchor="w", padx=10, pady=(5, 5))

        settings_frame = ctk.CTkFrame(parent, fg_color="transparent")
        settings_frame.pack(fill="x", padx=10)

        lang_label = ctk.CTkLabel(settings_frame, text="Language:")
        lang_label.grid(row=0, column=0, sticky="w", pady=3)

        current_lang = self._config.get("transcription.language") or "auto"
        self._language_var = ctk.StringVar(value=current_lang)
        language_menu = ctk.CTkOptionMenu(
            settings_frame,
            variable=self._language_var,
            values=["auto", "en", "es", "fr", "de", "it", "pt", "nl", "ja", "ko", "zh"],
            width=150,
        )
        language_menu.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=3)

        device_label = ctk.CTkLabel(settings_frame, text="Device:")
        device_label.grid(row=1, column=0, sticky="w", pady=3)

        self._device_var = ctk.StringVar(
            value=self._config.get("transcription.device", "auto")
        )
        device_menu = ctk.CTkOptionMenu(
            settings_frame,
            variable=self._device_var,
            values=["auto", "cuda", "cpu"],
            width=150,
        )
        device_menu.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=3)

    def _start_download(self, model_name: str) -> None:
        row = self._model_rows[model_name]
        row.set_downloading()

        def do_download():
            try:
                self._model_manager.download(
                    model_name,
                    on_progress=lambda p: self._parent.after(0, row.set_progress, p),
                )
                self._parent.after(0, row.set_installed)
            except Exception as e:
                self._parent.after(0, row.set_error, str(e))

        thread = threading.Thread(target=do_download, daemon=True)
        thread.start()

    def _delete_model(self, model_name: str) -> None:
        if self._active_model_var.get() == model_name:
            return
        self._model_manager.delete(model_name)
        self._model_rows[model_name].set_not_installed()

    def apply(self) -> None:
        self._config.set("transcription.model_size", self._active_model_var.get())
        lang = self._language_var.get()
        self._config.set("transcription.language", None if lang == "auto" else lang)
        self._config.set("transcription.device", self._device_var.get())


class _ModelRow:

    def __init__(
        self,
        parent: ctk.CTkFrame,
        model_info: ModelInfo,
        installed: bool,
        active_var: ctk.StringVar,
        on_download,
        on_delete,
    ):
        self._model_info = model_info
        self._active_var = active_var
        self._on_download = on_download
        self._on_delete = on_delete

        self._frame = ctk.CTkFrame(parent)
        self._frame.pack(fill="x", pady=2)
        self._frame.grid_columnconfigure(1, weight=1)

        self._radio = ctk.CTkRadioButton(
            self._frame,
            text=f"{model_info.name}  ({model_info.display_size})",
            variable=active_var,
            value=model_info.name,
        )
        self._radio.grid(row=0, column=0, padx=(5, 10), pady=5, sticky="w")
        if not installed:
            self._radio.configure(state="disabled")

        self._desc_label = ctk.CTkLabel(
            self._frame,
            text=model_info.description,
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self._desc_label.grid(row=0, column=1, sticky="w", padx=5)

        self._action_frame = ctk.CTkFrame(self._frame, fg_color="transparent")
        self._action_frame.grid(row=0, column=2, padx=5, pady=2)

        self._progress_bar: ctk.CTkProgressBar | None = None
        self._progress_label: ctk.CTkLabel | None = None

        if installed:
            self._show_installed_buttons()
        else:
            self._show_download_button()

    def _clear_action_frame(self) -> None:
        for widget in self._action_frame.winfo_children():
            widget.destroy()

    def _show_download_button(self) -> None:
        self._clear_action_frame()
        btn = ctk.CTkButton(
            self._action_frame,
            text="Download",
            width=80,
            height=28,
            command=lambda: self._on_download(self._model_info.name),
        )
        btn.pack()

    def _show_installed_buttons(self) -> None:
        self._clear_action_frame()
        delete_btn = ctk.CTkButton(
            self._action_frame,
            text="Delete",
            width=70,
            height=28,
            fg_color="transparent",
            border_width=1,
            text_color=("gray30", "gray70"),
            command=lambda: self._on_delete(self._model_info.name),
        )
        delete_btn.pack()

    def set_downloading(self) -> None:
        self._clear_action_frame()
        self._progress_bar = ctk.CTkProgressBar(self._action_frame, width=80)
        self._progress_bar.set(0)
        self._progress_bar.pack(side="left", padx=(0, 5))
        self._progress_label = ctk.CTkLabel(
            self._action_frame, text="0%", width=35, font=ctk.CTkFont(size=11)
        )
        self._progress_label.pack(side="left")

    def set_progress(self, fraction: float) -> None:
        if self._progress_bar:
            self._progress_bar.set(fraction)
        if self._progress_label:
            self._progress_label.configure(text=f"{int(fraction * 100)}%")

    def set_installed(self) -> None:
        self._radio.configure(state="normal")
        self._show_installed_buttons()

    def set_not_installed(self) -> None:
        self._radio.configure(state="disabled")
        self._show_download_button()

    def set_error(self, message: str) -> None:
        self._clear_action_frame()
        err_label = ctk.CTkLabel(
            self._action_frame,
            text="Error",
            text_color="red",
            font=ctk.CTkFont(size=11),
        )
        err_label.pack(side="left", padx=(0, 5))
        retry_btn = ctk.CTkButton(
            self._action_frame,
            text="Retry",
            width=60,
            height=24,
            command=lambda: self._on_download(self._model_info.name),
        )
        retry_btn.pack(side="left")
