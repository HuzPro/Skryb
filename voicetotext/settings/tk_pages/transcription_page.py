import logging
import tkinter
from tkinter import ttk

from voicetotext.settings.license_consent import (
    LicenseConsentPrompt,
    ask_license_consent,
    collect_license_acceptance,
)
from voicetotext.settings.service import SettingsService
from voicetotext.settings.signal import Signal
from voicetotext.settings.theme import HEADING_STYLE
from voicetotext.settings.tk_model_list import TkModelList
from voicetotext.settings.tk_pages.mode_selector import TkModeSelector
from voicetotext.settings.transcription_mode import TranscriptionMode

_logger = logging.getLogger(__name__)

_STYLE_LABELS: dict[str, str] = {
    "intended": "Intended (clean up what I meant)",
    "verbatim": "Verbatim (exactly what I said)",
}
_VERBATIM_NOTE = (
    "Verbatim keeps fillers and skips the dictionary, "
    "spoken commands, and number formatting."
)
_UNSUPPORTED_NOTE = "Only CrisperWhisper models have speech styles."


def _style_note(supported: bool) -> str:
    return _VERBATIM_NOTE if supported else _UNSUPPORTED_NOTE

_SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ja", "ko", "zh",
]

# Last-resort defaults if the service lists no models for a mode.
_FALLBACK_MODELS: dict[TranscriptionMode, str] = {
    TranscriptionMode.BATCH: "parakeet-tdt-0.6b-v3",
    TranscriptionMode.STREAMING: "parakeet-tdt-0.6b-v3",
}

_NO_GPU_STREAMING_REASON = (
    "Streaming needs a CUDA or ROCm GPU, and none was detected."
)


class TkTranscriptionPage(ttk.Frame):
    """Page for picking the mode, model, language, and engine.

    The batch/streaming :class:`TkModeSelector` sits above the
    :class:`TkModelList` it filters. The Engine combo is hardware-aware:
    it asks the service for candidates compatible with the selected model
    and re-populates on model change. Exposes imperative getters/setters
    so the window can build a :class:`SettingsDraft` on Save.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        service: SettingsService,
        confirm_license: LicenseConsentPrompt = ask_license_consent,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._confirm_license = confirm_license
        self.download_error = Signal()
        self.delete_error = Signal()
        self.model_auto_switched = Signal()
        self._engine_ids: list[str] = ["auto"]
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        draft = self._service.current_draft()
        self._mode = draft.transcription_mode

        ttk.Label(self, text="Mode", style=HEADING_STYLE).pack(anchor="w")
        self._mode_selector = TkModeSelector(self)
        self._mode_selector.set_mode(self._mode)
        if not self._service.supports_streaming_hardware():
            self._mode_selector.disable_streaming(_NO_GPU_STREAMING_REASON)
        self._mode_selector.mode_changed.connect(self.apply_mode)
        self._mode_selector.pack(fill="x", pady=(0, 8))

        ttk.Label(self, text="Models", style=HEADING_STYLE).pack(anchor="w")
        self._model_list = TkModelList(
            self,
            self._service,
            on_download=self.request_download,
            on_cancel=self.request_cancel,
            on_delete=self.request_delete,
        )
        self._model_list.set_mode(self._mode)
        self._model_list.selection_changed.connect(self._on_model_selected)
        self._model_list.pack(fill="both", expand=True, pady=(0, 8))
        self._model_list.set_selected(draft.model_size)

        ttk.Label(self, text="Settings", style=HEADING_STYLE).pack(
            anchor="w", pady=(8, 0),
        )
        self._build_form(draft)

    def _build_form(self, draft: object) -> None:
        form = ttk.Frame(self)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        self._language_var = tkinter.StringVar(master=self, value=draft.language)
        self._language_combo = ttk.Combobox(
            form,
            textvariable=self._language_var,
            values=_SUPPORTED_LANGUAGES,
            state="readonly",
        )

        self._engine_combo = ttk.Combobox(form, state="readonly", values=[])
        self._engine_reason = ttk.Label(form, text="")

        self._speech_style_var = tkinter.StringVar(
            master=self, value=_STYLE_LABELS[draft.crisperwhisper_mode],
        )
        self._speech_style_combo = ttk.Combobox(
            form,
            textvariable=self._speech_style_var,
            values=list(_STYLE_LABELS.values()),
            state="readonly",
        )
        self._speech_style_note = ttk.Label(form, text="")

        ttk.Label(form, text="Language:").grid(row=0, column=0, sticky="w", pady=2)
        self._language_combo.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(form, text="Engine:").grid(row=1, column=0, sticky="w", pady=2)
        self._engine_combo.grid(row=1, column=1, sticky="ew", pady=2)
        self._engine_reason.grid(row=2, column=1, sticky="w")
        ttk.Label(form, text="Speech style:").grid(
            row=3, column=0, sticky="w", pady=2,
        )
        self._speech_style_combo.grid(row=3, column=1, sticky="ew", pady=2)
        self._speech_style_note.grid(row=4, column=1, sticky="w")

        self._populate_engine_combo(draft.model_size, current_engine=draft.engine)
        self._refresh_speech_style(draft.model_size)

    # ------------------------------------------------------------------
    # Accessors used by the window / tests
    # ------------------------------------------------------------------

    def mode_selector(self) -> TkModeSelector:
        return self._mode_selector

    def model_list(self) -> TkModelList:
        return self._model_list

    def current_model(self) -> str:
        return self._model_list.selected_model_name()

    def set_model(self, name: str) -> None:
        self._model_list.set_selected(name)
        self._populate_engine_combo(name)

    def current_mode(self) -> TranscriptionMode:
        return self._mode_selector.current_mode()

    def current_language(self) -> str:
        return self._language_var.get()

    def set_language(self, language: str) -> None:
        self._language_var.set(language)

    def current_engine(self) -> str:
        index = self._engine_combo.current()
        if index < 0:
            return "auto"
        return self._engine_ids[index]

    def set_engine(self, engine: str) -> None:
        self._select_engine_value(engine)

    def engine_ids(self) -> list[str]:
        return list(self._engine_ids)

    def engine_reason(self) -> str:
        return self._engine_reason.cget("text")

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def apply_mode(self, mode: TranscriptionMode) -> None:
        """React to a batch/streaming toggle.

        Re-filters the model list, auto-selects a compatible model if the
        current one cannot run in the new mode (announced via
        :attr:`model_auto_switched`).
        """
        self._mode = mode
        self._mode_selector.set_mode(mode)
        self._model_list.set_mode(mode)
        self._switch_model_if_incompatible(mode)

    def _switch_model_if_incompatible(self, mode: TranscriptionMode) -> None:
        current = self.current_model()
        if self._model_fits_mode(current, mode):
            return
        replacement = self._default_model_for(mode)
        self.set_model(replacement)
        self.model_auto_switched.emit(current, replacement)

    def _model_fits_mode(self, name: str, mode: TranscriptionMode) -> bool:
        if mode is TranscriptionMode.STREAMING:
            return self._service.model_supports_streaming(name)
        return self._service.model_supports_batch(name)

    def _default_model_for(self, mode: TranscriptionMode) -> str:
        models = self._service.list_models(mode)
        installed = [m for m in models if m.installed]
        if installed:
            return installed[0].name
        if models:
            return models[0].name
        return _FALLBACK_MODELS[mode]

    # ------------------------------------------------------------------
    # Selection and engine internals
    # ------------------------------------------------------------------

    def _on_model_selected(self, name: str) -> None:
        self._populate_engine_combo(name)
        self._refresh_speech_style(name)

    def current_crisperwhisper_mode(self) -> str:
        for mode, label in _STYLE_LABELS.items():
            if label == self._speech_style_var.get():
                return mode
        return "intended"

    def set_crisperwhisper_mode(self, mode: str) -> None:
        self._speech_style_var.set(
            _STYLE_LABELS.get(mode, _STYLE_LABELS["intended"]),
        )

    def _refresh_speech_style(self, model_name: str) -> None:
        supported = self._service.model_has_crisperwhisper_modes(model_name)
        self._speech_style_combo.configure(
            state="readonly" if supported else "disabled",
        )
        self._speech_style_note.configure(text=_style_note(supported))

    def _populate_engine_combo(
        self, model_name: str, current_engine: str = "auto",
    ) -> None:
        candidates = tuple(self._service.engine_candidates_for(model_name) or ())
        labels = ["auto"]
        ids = ["auto"]
        for candidate in candidates:
            label = candidate.display_name
            if candidate.is_recommended:
                label += "  (recommended)"
            labels.append(label)
            ids.append(candidate.engine_id)
        self._engine_ids = ids
        self._engine_combo.configure(values=labels)
        self._select_engine_value(current_engine)
        reason = candidates[0].reason if candidates else ""
        self._engine_reason.configure(text=str(reason))

    def _select_engine_value(self, engine_id: str) -> None:
        if engine_id in self._engine_ids:
            self._engine_combo.current(self._engine_ids.index(engine_id))
        else:
            self._engine_combo.current(0)

    # ------------------------------------------------------------------
    # Download/delete commands (proxy to the service, tolerate errors)
    # ------------------------------------------------------------------

    def request_download(self, name: str) -> None:
        _logger.info("tk transcription page: download requested for '%s'", name)
        # Consent precedes acquisition: the manager refuses to download a
        # gated model, and this is the only surface offering the terms.
        if not collect_license_acceptance(
            self._service, name, self, self._confirm_license,
        ):
            return
        try:
            self._service.start_download(name)
        except Exception as exc:  # noqa: BLE001 - surface anything to the user
            _logger.exception("download for '%s' failed", name)
            self.download_error.emit(name, str(exc))

    def request_pause(self, name: str) -> None:
        try:
            self._service.pause_download(name)
        except LookupError:
            pass

    def request_resume(self, name: str) -> None:
        self._service.resume_download(name)

    def request_cancel(self, name: str) -> None:
        try:
            self._service.cancel_download(name)
        except LookupError:
            pass

    def request_delete(self, name: str) -> None:
        try:
            self._service.delete_model(name)
        except ValueError as exc:
            self.delete_error.emit(name, str(exc))
