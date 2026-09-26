import logging
import tkinter
from collections.abc import Callable
from tkinter import ttk

from voicetotext.settings.license_consent import (
    LicenseConsentPrompt,
    ask_license_consent,
    collect_license_acceptance,
)
from voicetotext.settings.service import SettingsService
from voicetotext.settings.settings_draft import SettingsDraft
from voicetotext.settings.signal import Signal
from voicetotext.settings.theme import (
    PAGE_LINK_BUTTON_STYLE,
    RowGroup,
    combobox,
    section_heading,
)
from voicetotext.settings.tk_model_list import TkModelList
from voicetotext.settings.tk_pages.mode_selector import TkModeSelector
from voicetotext.settings.transcription_mode import TranscriptionMode

_logger = logging.getLogger(__name__)

_STYLE_LABELS: dict[str, str] = {
    "intended": "Intended (what I meant)",
    "verbatim": "Verbatim (what I said)",
}
_VERBATIM_NOTE = (
    "Verbatim keeps fillers and skips the dictionary, "
    "spoken commands, and number formatting."
)
_UNSUPPORTED_NOTE = "Only CrisperWhisper models have speech styles."


def _style_note(supported: bool) -> str:
    return _VERBATIM_NOTE if supported else _UNSUPPORTED_NOTE


_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}
_LANGUAGE_CODES = {name: code for code, name in _LANGUAGE_NAMES.items()}

# Last-resort defaults if the service lists no models for a mode.
_FALLBACK_MODELS: dict[TranscriptionMode, str] = {
    TranscriptionMode.BATCH: "parakeet-tdt-0.6b-v3",
    TranscriptionMode.STREAMING: "parakeet-tdt-0.6b-v3",
}

_NO_GPU_STREAMING_REASON = (
    "Needs a CUDA or ROCm GPU, and none was detected."
)
_CHECKING_FOR_GPU_REASON = "Checking for a GPU..."
_ADVANCED_CLOSED = "Advanced ▸"
_ADVANCED_OPEN = "Advanced ▾"

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]


def _run_now(action: Callable[[], None]) -> None:
    action()


class TkTranscriptionPage(ttk.Frame):
    """Page for picking the mode, model, language, and engine.

    The Batch/Live :class:`TkModeSelector` sits above the
    :class:`TkModelList` it filters. The Engine combo, under Advanced, is
    hardware-aware: it asks the service for candidates compatible with the
    selected model and re-populates on model change. Exposes imperative
    getters/setters so the window can build a :class:`SettingsDraft` on Save.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        service: SettingsService,
        confirm_license: LicenseConsentPrompt = ask_license_consent,
        schedule: ScheduleOnGuiThread = _run_now,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._confirm_license = confirm_license
        self._schedule = schedule
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

        section_heading(self, "Mode", first=True)
        self._mode_selector = TkModeSelector(self)
        self._mode_selector.set_mode(self._mode)
        self._mode_selector.mode_changed.connect(self.apply_mode)
        self._mode_selector.pack(fill="x")

        section_heading(self, "Model")
        self._model_list = TkModelList(
            self,
            self._service,
            on_download=self.request_download,
            on_cancel=self.request_cancel,
            on_delete=self.request_delete,
        )
        self._model_list.set_mode(self._mode)
        self._model_list.selection_changed.connect(self._on_model_selected)
        self._model_list.pack(fill="x")
        self._model_list.set_selected(draft.model_size)

        section_heading(self, "Speech")
        self._build_form(draft)
        self._gate_streaming_on_hardware()

    def _gate_streaming_on_hardware(self) -> None:
        """Keep Live off until a GPU is known, without waiting for one.

        The probe imports torch, which can take many seconds, so the page
        opens at once and updates itself when the probe reports back.
        """
        if self._service.hardware_probe_pending():
            self._mode_selector.disable_streaming(_CHECKING_FOR_GPU_REASON)
            self._service.request_hardware_probe(
                lambda: self._schedule(self._on_hardware_probed),
            )
        elif not self._service.supports_streaming_hardware():
            self._mode_selector.disable_streaming(_NO_GPU_STREAMING_REASON)

    def _on_hardware_probed(self) -> None:
        if not self.winfo_exists():
            return
        if self._service.supports_streaming_hardware():
            self._mode_selector.enable_streaming()
        else:
            self._mode_selector.disable_streaming(_NO_GPU_STREAMING_REASON)
        self._populate_engine_combo(
            self.current_model(), current_engine=self.current_engine(),
        )

    def _build_form(self, draft: SettingsDraft) -> None:
        self._language_var = tkinter.StringVar(
            master=self, value=_language_name(draft.language),
        )
        self._speech_style_var = tkinter.StringVar(
            master=self, value=_STYLE_LABELS[draft.crisperwhisper_mode],
        )
        rows = RowGroup(self)
        rows.pack(fill="x")
        self._language_combo = rows.add(
            "Language",
            "The language you speak in.",
            lambda parent: combobox(
                parent, textvariable=self._language_var,
                values=list(_LANGUAGE_NAMES.values()),
            ),
            fill=True,
        ).control
        self._speech_style_row = rows.add(
            "Speech style",
            "",
            lambda parent: combobox(
                parent, textvariable=self._speech_style_var,
                values=list(_STYLE_LABELS.values()),
            ),
            fill=True,
        )
        self._speech_style_combo = self._speech_style_row.control

        self._advanced_button = ttk.Button(
            self, text=_ADVANCED_CLOSED, style=PAGE_LINK_BUTTON_STYLE,
            command=self._toggle_advanced,
        )
        self._advanced_button.pack(anchor="w", pady=(16, 0))
        self._advanced = RowGroup(self)
        self._engine_row = self._advanced.add(
            "Engine",
            "",
            lambda parent: combobox(parent, values=[]),
            fill=True,
        )
        self._engine_combo = self._engine_row.control
        self._engine_reason = self._engine_row.description

        self._populate_engine_combo(draft.model_size, current_engine=draft.engine)
        self._refresh_speech_style(draft.model_size)

    def _toggle_advanced(self) -> None:
        if self._advanced.winfo_manager():
            self._advanced.pack_forget()
            self._advanced_button.configure(text=_ADVANCED_CLOSED)
        else:
            self._advanced.pack(fill="x", pady=(8, 0))
            self._advanced_button.configure(text=_ADVANCED_OPEN)

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
        shown = self._language_var.get()
        return _LANGUAGE_CODES.get(shown, shown)

    def set_language(self, language: str) -> None:
        self._language_var.set(_language_name(language))

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
        """React to a Batch/Live toggle.

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
        self._speech_style_row.set_description(_style_note(supported))

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
        self._engine_row.set_description(str(reason))

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


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)
