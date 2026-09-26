"""The UI contract's endpoints, each a thin adapter over an existing service.

One :class:`CoreApi` serves first-run setup and another the running app;
the server swaps from the first to the second without the shell noticing.
Requests that change something run one at a time.
"""

import functools
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, TypeVar

from voicetotext.core.action_dispatcher import ThreadedActionDispatcher
from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.core.state_machine import AppState
from voicetotext.history.entries import HistoryEntry
from voicetotext.history.service import HistoryService, RetryOutcome
from voicetotext.hotkeys.bindings import Binding, BindingError, validate_bindings
from voicetotext.output.dictionary_config import LiteralRule
from voicetotext.recording.input_devices import DEFAULT_LABEL, SYSTEM_DEFAULT, Microphones
from voicetotext.recording.pill import DEFAULT_PILL_POSITION
from voicetotext.settings.service import EngineEventListener, SettingsService
from voicetotext.settings.settings_draft import ChangeSet
from voicetotext.settings.signal import Signal
from voicetotext.settings.transcription_mode import TranscriptionMode
from voicetotext.transcription.model_info import ALL_MODELS
from voicetotext.ui.api_server import ApiError, ApiRequest, ApiResponse, ApiServer, Route
from voicetotext.ui.event_hub import EventHub, Json
from voicetotext.ui.shapes import (
    Fields,
    bindings_from_json,
    dictionary_from_contract,
    dictionary_json,
    draft_from_json,
    entry_id,
    history_entry_json,
    learned_pairs,
    settings_json,
    shown_combination,
    writing_status_json,
)
from voicetotext.writing.service import WritingService
from voicetotext.writing.transforms import BUILTIN_TRANSFORMS

_logger = logging.getLogger(__name__)

_ESCAPE = "esc"
# About twenty level readings a second, the contract's meter rate.
_METER_SECONDS = 0.05
_NO_CONTENT = ApiResponse(204)
_ACCEPTED = ApiResponse(202)
_OUTPUT_ONLY = ChangeSet(engine_reload_required=False, hotkey_changed=False, output_changed=True)
_T = TypeVar("_T")
_IN_BACKGROUND = ThreadedActionDispatcher()


class CaptureBridge(Protocol):
    """The native key source a shortcut is recorded from."""

    key_pressed: Signal

    def start(self) -> None: ...
    def stop(self) -> None: ...


@functools.cache
def app_version() -> str:
    try:
        return version("voicetotext")
    except PackageNotFoundError:
        return "unknown"


def open_with_shell(path: Path) -> None:
    os.startfile(path)  # type: ignore[attr-defined]  # Windows only, like the app


def _nothing() -> None:
    return None


def _ignore_changes(_changes: ChangeSet) -> None:
    return None


def _idle() -> AppState:
    return AppState.IDLE


class ShortcutCapture:
    """Records a shortcut from the keys pressed while the app's own shortcuts are paused.

    The bridge swallows every key system-wide while it runs, so it only runs
    between the shell's on and off, or until Esc. Keys arrive on the hook's
    thread, which has to answer Windows at once and cannot stop its own hook,
    so the release after Esc happens on a worker.
    """

    def __init__(
        self,
        bridge: CaptureBridge,
        publish: Callable[[Json], None],
        *,
        pause: Callable[[], None] = _nothing,
        resume: Callable[[], None] = _nothing,
        run_in_background: Callable[[Callable[[], None]], None] = _IN_BACKGROUND.dispatch,
    ) -> None:
        self._bridge = bridge
        self._publish = publish
        self._pause = pause
        self._resume = resume
        self._run_in_background = run_in_background
        self._lock = threading.Lock()
        self._active = False
        self._parts: list[str] = []
        bridge.key_pressed.connect(self._on_key)

    def start(self) -> None:
        with self._lock:
            if self._active:
                return
            self._active = True
            self._parts = []
        self._pause()
        try:
            self._bridge.start()
        except Exception:
            self._deactivate()
            self._resume()
            raise

    def stop(self) -> None:
        if self._deactivate():
            self._release()

    def _deactivate(self) -> bool:
        with self._lock:
            was_active, self._active = self._active, False
        return was_active

    def _release(self) -> None:
        try:
            self._bridge.stop()
        finally:
            self._resume()

    def _on_key(self, name: str) -> None:
        if name == _ESCAPE:
            if self._deactivate():
                self._publish({"error": "Recording the shortcut was cancelled.", "cancelled": True})
                self._run_in_background(self._release)
            return
        with self._lock:
            if not self._active or (self._parts and self._parts[-1] == name):
                return
            self._parts.append(name)
            combo = "+".join(self._parts)
        self._publish({"keys": combo, "display": shown_combination(combo)})


class _LevelThrottle:
    """The loudest level heard, passed on at most every ``_METER_SECONDS``."""

    def __init__(
        self, publish: Callable[[float], None], clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._publish = publish
        self._clock = clock
        self._peak = 0.0
        self._sent_at = 0.0

    def __call__(self, level: float) -> None:
        self._peak = max(self._peak, level)
        now = self._clock()
        if now - self._sent_at >= _METER_SECONDS:
            peak, self._peak, self._sent_at = self._peak, 0.0, now
            self._publish(peak)


class _ApplyOnceLoaded(EngineEventListener):
    """Settings Save & Apply held back until the new model is live, as the window did."""

    def __init__(
        self, service: SettingsService, changes: ChangeSet, apply: Callable[[ChangeSet], None],
    ) -> None:
        self._service = service
        self._changes = changes
        self._apply = apply

    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None:
        return None

    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        self._service.unsubscribe_engine(self)
        self._apply(self._changes)

    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None:
        self._service.unsubscribe_engine(self)


class CoreApi:
    def __init__(
        self,
        service: SettingsService,
        hub: EventHub,
        *,
        microphones: Microphones,
        capture: ShortcutCapture,
        diagnostics_dir: Path,
        state: Callable[[], AppState] = _idle,
        on_save: Callable[[ChangeSet], None] = _ignore_changes,
        writing: WritingService | None = None,
        history: HistoryService | None = None,
        first_run: bool = False,
        on_setup_done: Callable[[], None] = _nothing,
        open_path: Callable[[Path], None] = open_with_shell,
    ) -> None:
        self._service = service
        self._hub = hub
        self._microphones = microphones
        self._capture = capture
        self._diagnostics_dir = diagnostics_dir
        self._state = state
        self._on_save = on_save
        self._writing = writing
        self._history = history
        self._first_run = first_run
        self._on_setup_done = on_setup_done
        self._open_path = open_path
        self._lock = threading.Lock()

    def serve_on(self, server: ApiServer) -> None:
        """Answer every request on *server*, and its events, from this API."""
        server.bind(self.routes())
        server.hub.use(
            self,
            model=self._service.selected_model(),
            loading=self._state() is AppState.LOADING,
        )

    def routes(self) -> list[Route]:
        table: list[Route] = [
            ("GET", "/api/app", lambda _request: ApiResponse(200, self.app_info())),
            ("GET", "/api/settings", self._get_settings),
            ("PUT", "/api/settings", self._put_settings),
            ("POST", "/api/license/accept", self._accept_license),
            ("GET", "/api/models", self._list_models),
            ("POST", "/api/models/{name}/download", self._download_model),
            ("POST", "/api/models/{name}/pause", self._pause_download),
            ("POST", "/api/models/{name}/resume", self._resume_download),
            ("POST", "/api/models/{name}/cancel", self._cancel_download),
            ("POST", "/api/models/{name}/delete", self._delete_model),
            ("GET", "/api/engines", self._engines),
            ("POST", "/api/hardware/probe", self._probe_hardware),
            ("GET", "/api/microphones", self._list_microphones),
            ("POST", "/api/microphones/meter", self._meter),
            ("POST", "/api/shortcuts/capture", self._capture_shortcut),
            ("POST", "/api/shortcuts/validate", self._validate_shortcuts),
            ("GET", "/api/dictionary", self._get_dictionary),
            ("PUT", "/api/dictionary", self._put_dictionary),
            ("POST", "/api/dictionary/forget-learned", self._forget_learned_words),
            ("DELETE", "/api/dictionary/learned/{spoken}", self._forget_learned_word),
            ("GET", "/api/writing", self._writing_status),
            *(
                ("POST", f"/api/writing/{action}", functools.partial(self._writing_action, action))
                for action in _WRITING_ACTIONS
            ),
            ("GET", "/api/transforms", self._list_transforms),
            ("PUT", "/api/transforms", self._put_transforms),
            ("POST", "/api/transforms/restore-builtins", self._restore_builtin_transforms),
            ("GET", "/api/history", self._list_history),
            ("DELETE", "/api/history", self._delete_all_history),
            ("DELETE", "/api/history/{id}", self._delete_history_entry),
            ("POST", "/api/history/{id}/play", self._play_recording),
            ("POST", "/api/history/{id}/stop", self._stop_playback),
            ("POST", "/api/history/{id}/retry", self._retry_transcription),
            ("POST", "/api/history/{id}/export", self._export_audio),
            ("POST", "/api/settings/export", self._export_settings),
            ("POST", "/api/settings/import", self._import_settings),
            ("POST", "/api/logs/open", self._open_logs),
            ("PUT", "/api/logs/directory", self._set_logs_directory),
            ("POST", "/api/pill/reset-position", self._reset_pill_position),
            ("POST", "/api/setup/done", self._setup_done),
        ]
        return [
            (method, path, handler if method == "GET" else self._one_at_a_time(handler))
            for method, path, handler in table
        ]

    def _one_at_a_time(
        self, handler: Callable[[ApiRequest], ApiResponse],
    ) -> Callable[[ApiRequest], ApiResponse]:
        def serialised(request: ApiRequest) -> ApiResponse:
            with self._lock:
                return handler(request)

        return serialised

    # -- snapshots the event hub asks for ------------------------------------

    def app_info(self) -> Json:
        return {
            "version": app_version(),
            "state": self._state().value,
            "engine": self._hub.engine_status(),
            "first_run": self._first_run,
            "diagnostics_directory": str(self._diagnostics_dir),
            "history_directory": str(self._diagnostics_dir),
            "pill_default_position": list(DEFAULT_PILL_POSITION),
        }

    def writing_status(self) -> Json | None:
        writing = self._writing
        if writing is None:
            return None
        runtime, model = writing.server_paths()
        return writing_status_json(
            writing.status(), runtime=runtime, model=model,
            learned_style=writing.learned_style_summary(),
        )

    def watch_writing(self, listener: Callable[[], None]) -> Callable[[], None]:
        writing = self._writing
        if writing is None:
            return _nothing
        writing.subscribe(listener)
        return lambda: writing.unsubscribe(listener)

    def release_devices(self) -> None:
        self._microphones.stop_meter()
        self._capture.stop()

    # -- settings --------------------------------------------------------------

    def _get_dictionary(self, _request: ApiRequest) -> ApiResponse:
        return ApiResponse(200, dictionary_json(self._service.dictionary_rules()))

    def _get_settings(self, _request: ApiRequest) -> ApiResponse:
        return ApiResponse(200, settings_json(self._service.current_draft()))

    def _put_settings(self, request: ApiRequest) -> ApiResponse:
        saved = self._service.current_draft()
        draft = _read(
            lambda body: draft_from_json(body, saved), request.body,
            "Those settings could not be read.",
        )
        if self._first_run and not self._service.any_model_installed():
            raise ApiError(400, "Please download at least one model before saving.")
        _check_shortcuts(draft.hotkey_bindings)
        if not self._service.is_installed(draft.model_size):
            raise ApiError(400, f"'{draft.model_size}' is not installed. Download it first.")
        self._refuse_until_licensed(draft.model_size)
        if not self._service.can_apply_now():
            raise ApiError(
                409,
                "Settings cannot be applied while recording or transcribing. "
                "Try again in a moment.",
            )
        if self._service.is_engine_swapping():
            raise ApiError(409, "A model is still loading. Save again once it has finished.")
        changes = self._service.apply_draft(draft)
        reload = changes.engine_reload_required and not self._first_run
        if reload:
            self._apply_once_loaded(changes)
        else:
            self._on_save(changes)
        self._publish_settings()
        return ApiResponse(200, {"changes": asdict(changes), "engine_reload_started": reload})

    def _refuse_until_licensed(self, model_name: str) -> None:
        if not self._service.license_acceptance_required(model_name):
            return
        terms = self._service.license_terms(model_name)
        raise ApiError(
            409,
            f"'{model_name}' needs its licence accepted before it can be used. "
            "Nothing was changed.",
            extra={"license": {
                "id": terms.license_id,
                "title": f"Licence - {terms.license_id}",
                "text": terms.text,
                "url": terms.url,
            }},
        )

    def _apply_once_loaded(self, changes: ChangeSet) -> None:
        waiter = _ApplyOnceLoaded(self._service, changes, self._on_save)
        self._service.subscribe_engine(waiter)
        try:
            self._service.trigger_engine_swap()
        except Exception:
            self._service.unsubscribe_engine(waiter)
            raise

    def _publish_settings(self) -> None:
        self._hub.publish("settings", settings_json(self._service.current_draft()))

    def _accept_license(self, request: ApiRequest) -> ApiResponse:
        license_id = _read(_text_field("id"), request.body, "Say which licence to accept.")
        model = next((info for info in ALL_MODELS.values() if info.license_id == license_id), None)
        if model is None:
            raise ApiError(404, "No model uses that licence.")
        self._service.accept_license(model.name)
        return _NO_CONTENT

    def _export_settings(self, request: ApiRequest) -> ApiResponse:
        path = _path_from(request)
        try:
            self._service.export_settings(path)
        except OSError as exc:
            raise ApiError(400, f"Could not export settings: {exc}") from exc
        return _NO_CONTENT

    def _import_settings(self, request: ApiRequest) -> ApiResponse:
        path = _path_from(request)
        try:
            self._service.import_settings(path)
        except (OSError, ValueError) as exc:
            raise ApiError(400, f"Could not import settings: {exc}") from exc
        self._publish_settings()
        return _NO_CONTENT

    def _reset_pill_position(self, _request: ApiRequest) -> ApiResponse:
        self._service.reset_pill_position()
        return _NO_CONTENT

    def _open_logs(self, _request: ApiRequest) -> ApiResponse:
        self._open_path(self._diagnostics_dir)
        return _NO_CONTENT

    def _set_logs_directory(self, request: ApiRequest) -> ApiResponse:
        folder = _path_from(request)
        if not folder.is_absolute() or not folder.is_dir():
            raise ApiError(400, f"'{folder}' is not a folder on this PC.")
        try:
            self._service.set_diagnostics_directory(str(folder))
        except OSError as exc:
            raise ApiError(400, f"Could not save the logs folder: {exc}") from exc
        return _NO_CONTENT

    def _setup_done(self, _request: ApiRequest) -> ApiResponse:
        self._on_setup_done()
        return _NO_CONTENT

    # -- models, engines, hardware --------------------------------------------------

    def _list_models(self, _request: ApiRequest) -> ApiResponse:
        views = {
            view.name: view
            for mode in TranscriptionMode
            for view in self._service.list_models(mode)
        }
        return ApiResponse(200, [
            {
                "name": view.name,
                "display_size": view.display_size,
                "description": view.description,
                "installed": view.installed,
                "supports_batch": self._service.model_supports_batch(view.name),
                "supports_streaming": self._service.model_supports_streaming(view.name),
                "has_speech_styles": self._service.model_has_crisperwhisper_modes(view.name),
                "license_id": ALL_MODELS[view.name].license_id or None,
                "download": self._hub.download_of(view.name),
            }
            for view in views.values()
        ])

    def _download_model(self, request: ApiRequest) -> ApiResponse:
        name = _model_named(request)
        # Consent comes before the download: the manager refuses a gated model.
        self._refuse_until_licensed(name)
        try:
            self._service.start_download(name)
        except Exception as exc:
            _logger.exception("Download of '%s' did not start", name)
            raise ApiError(400, f"Could not start the download of '{name}': {exc}") from exc
        return _ACCEPTED

    def _pause_download(self, request: ApiRequest) -> ApiResponse:
        _ignoring_missing_download(self._service.pause_download, _model_named(request))
        return _ACCEPTED

    def _resume_download(self, request: ApiRequest) -> ApiResponse:
        self._service.resume_download(_model_named(request))
        return _ACCEPTED

    def _cancel_download(self, request: ApiRequest) -> ApiResponse:
        _ignoring_missing_download(self._service.cancel_download, _model_named(request))
        return _ACCEPTED

    def _delete_model(self, request: ApiRequest) -> ApiResponse:
        name = _model_named(request)
        try:
            self._service.delete_model(name)
        except ValueError as exc:
            raise ApiError(
                409, f"'{name}' is the model in use. Choose another model before deleting it.",
            ) from exc
        return _ACCEPTED

    def _engines(self, request: ApiRequest) -> ApiResponse:
        model = request.query.get("model", "")
        if not model:
            raise ApiError(400, "Say which model to list engines for.")
        return ApiResponse(200, {
            "candidates": [
                {
                    "id": candidate.engine_id,
                    "label": candidate.display_name,
                    "recommended": candidate.is_recommended,
                    "reason": candidate.reason,
                }
                for candidate in self._service.engine_candidates_for(model)
            ],
            "hardware": self._hardware(),
        })

    def _hardware(self) -> Json:
        profile = self._service.detected_hardware()
        return {
            "probe_pending": self._service.hardware_probe_pending(),
            "streaming_supported": self._service.supports_streaming_hardware(),
            "kind": "" if profile is None else profile.kind,
            "device_name": "" if profile is None else profile.device_name,
        }

    def _probe_hardware(self, _request: ApiRequest) -> ApiResponse:
        self._service.request_hardware_probe(
            lambda: self._hub.publish("hardware", self._hardware()),
        )
        return _ACCEPTED

    # -- microphones and shortcuts -------------------------------------------------

    def _list_microphones(self, request: ApiRequest) -> ApiResponse:
        names = self._microphones.names(rescan=request.query.get("rescan") == "1")
        labels = self._microphones.labels(names)
        chosen = self._service.current_draft().input_device
        microphones: list[Json] = [
            {"name": SYSTEM_DEFAULT, "label": DEFAULT_LABEL, "connected": True, "is_default": True},
            *(
                {
                    "name": name, "label": labels.get(name, name),
                    "connected": True, "is_default": False,
                }
                for name in names
            ),
        ]
        # A chosen microphone that is unplugged stays listed, so saving
        # without touching the picker does not quietly forget it.
        if chosen and chosen not in names:
            microphones.append(
                {"name": chosen, "label": chosen, "connected": False, "is_default": False},
            )
        return ApiResponse(200, microphones)

    def _meter(self, request: ApiRequest) -> ApiResponse:
        device, on = _read(
            _meter_request, request.body, "Say which microphone and whether to listen.",
        )
        if not on:
            self._microphones.stop_meter()
            return _NO_CONTENT
        level = _LevelThrottle(lambda peak: self._hub.publish("meter", {"level": peak}))
        if not self._microphones.start_meter(device, level):
            raise ApiError(400, "This microphone could not be opened.")
        return _NO_CONTENT

    def _capture_shortcut(self, request: ApiRequest) -> ApiResponse:
        on = _read(_flag_field("on"), request.body, "Say whether to record a shortcut.")
        if not on:
            self._capture.stop()
            return _NO_CONTENT
        try:
            self._capture.start()
        except Exception as exc:
            _logger.exception("Could not start recording a shortcut")
            raise ApiError(500, "Could not start recording the shortcut.", str(exc)) from exc
        return _NO_CONTENT

    def _validate_shortcuts(self, request: ApiRequest) -> ApiResponse:
        bindings = _read(
            lambda body: bindings_from_json(Fields(body, "body").items("bindings")),
            request.body, "Those shortcuts could not be read.",
        )
        _check_shortcuts(bindings)
        return ApiResponse(200, {"ok": True})

    # -- dictionary and transforms -------------------------------------------------

    def _put_dictionary(self, request: ApiRequest) -> ApiResponse:
        dictionary = _read(
            dictionary_from_contract, request.body, "That dictionary could not be read.",
        )
        forgotten = learned_pairs(self._service.dictionary_rules()) - learned_pairs(dictionary)
        self._service.save_dictionary_rules(dictionary)
        self._forget(forgotten)
        self._on_save(_OUTPUT_ONLY)
        return _NO_CONTENT

    def _forget_learned_words(self, _request: ApiRequest) -> ApiResponse:
        self._forget_learned(lambda _spoken: True)
        return _NO_CONTENT

    def _forget_learned_word(self, request: ApiRequest) -> ApiResponse:
        spoken = request.params["spoken"].strip()
        if not self._forget_learned(lambda first: first == spoken):
            raise ApiError(404, "That learned word is no longer in the dictionary.")
        return _NO_CONTENT

    def _forget_learned(self, chosen: Callable[[str], bool]) -> bool:
        """Drop the learned words whose first spoken form is *chosen*, for good."""
        current = self._service.dictionary_rules()
        forgotten = {pair for pair in learned_pairs(current) if chosen(pair[0])}
        if not forgotten:
            return False
        kept = tuple(
            rule for rule in current.rules
            if not (isinstance(rule, LiteralRule) and rule.learned and chosen(rule.spoken[0].strip()))
        )
        self._service.save_dictionary_rules(replace(current, rules=kept))
        self._forget(forgotten)
        self._on_save(_OUTPUT_ONLY)
        return True

    def _forget(self, pairs: set[tuple[str, str]]) -> None:
        if self._writing is not None:
            self._writing.forget_learned_words(tuple(sorted(pairs)))

    def _list_transforms(self, _request: ApiRequest) -> ApiResponse:
        writing = self._require_writing()
        return ApiResponse(200, [
            {"name": name, "instruction": instruction, "builtin": name in BUILTIN_TRANSFORMS}
            for name, instruction in writing.transforms().items()
        ])

    def _put_transforms(self, request: ApiRequest) -> ApiResponse:
        writing = self._require_writing()
        transforms = _read(
            _transforms_from_json, request.body, "Those transformations could not be read.",
        )
        writing.save_transforms(transforms)
        self._on_save(_OUTPUT_ONLY)
        return _NO_CONTENT

    def _restore_builtin_transforms(self, _request: ApiRequest) -> ApiResponse:
        writing = self._require_writing()
        writing.save_transforms({**writing.transforms(), **BUILTIN_TRANSFORMS})
        self._on_save(_OUTPUT_ONLY)
        return _NO_CONTENT

    # -- writing model ----------------------------------------------------------------

    def _require_writing(self) -> WritingService:
        if self._writing is None:
            raise ApiError(404, "The writing model is not available until setup has finished.")
        return self._writing

    def _writing_status(self, _request: ApiRequest) -> ApiResponse:
        self._require_writing()
        return ApiResponse(200, self.writing_status())

    def _writing_action(self, action: str, _request: ApiRequest) -> ApiResponse:
        writing = self._require_writing()
        run, asynchronous = _WRITING_ACTIONS[action]
        run(writing)
        if not asynchronous:
            self._hub.publish("writing", self.writing_status())
        return _ACCEPTED if asynchronous else _NO_CONTENT

    # -- history ------------------------------------------------------------------------

    def _require_history(self) -> HistoryService:
        if self._history is None:
            raise ApiError(404, "History is not available until setup has finished.")
        return self._history

    def _find_entry(self, request: ApiRequest) -> HistoryEntry:
        entry = self._require_history().find(request.params["id"])
        if entry is None:
            raise ApiError(404, "That dictation is no longer in history.")
        return entry

    def _list_history(self, request: ApiRequest) -> ApiResponse:
        entries = self._require_history().entries(request.query.get("query", ""))
        return ApiResponse(200, [history_entry_json(entry) for entry in entries])

    def _delete_history_entry(self, request: ApiRequest) -> ApiResponse:
        self._require_history().delete(self._find_entry(request))
        self._hub.publish("history", {"event": "changed"})
        return _NO_CONTENT

    def _delete_all_history(self, _request: ApiRequest) -> ApiResponse:
        self._require_history().delete_all(self._report_all_deleted)
        return _ACCEPTED

    def _report_all_deleted(self, deleted: bool) -> None:
        if deleted:
            self._hub.publish("history", {"event": "deleted_all"})
        else:
            self._hub.publish(
                "history", {"event": "changed", "error": "Could not delete all history."},
            )

    def _play_recording(self, request: ApiRequest) -> ApiResponse:
        entry = self._find_entry(request)
        try:
            self._require_history().play(entry)
        except FileNotFoundError as exc:
            raise ApiError(404, str(exc)) from exc
        return _NO_CONTENT

    def _stop_playback(self, _request: ApiRequest) -> ApiResponse:
        self._require_history().stop()
        return _NO_CONTENT

    def _retry_transcription(self, request: ApiRequest) -> ApiResponse:
        entry = self._find_entry(request)
        try:
            self._require_history().retry(
                entry, functools.partial(self._report_retry, entry_id(entry)),
            )
        except FileNotFoundError as exc:
            raise ApiError(404, str(exc)) from exc
        return _ACCEPTED

    def _export_audio(self, request: ApiRequest) -> ApiResponse:
        entry = self._find_entry(request)
        destination = _path_from(request)
        try:
            self._require_history().export(entry, destination)
        except (OSError, ValueError) as exc:
            raise ApiError(400, f"Could not save the audio: {exc}") from exc
        return _NO_CONTENT

    def _report_retry(self, entry: str, outcome: RetryOutcome) -> None:
        if isinstance(outcome, Exception):
            self._hub.publish(
                "history", {"event": "retry_failed", "id": entry, "error": str(outcome)},
            )
        else:
            self._hub.publish("history", {"event": "retry_done", "id": entry, "text": outcome})


# Each writing action, and whether it only starts something the status
# reports on later (202) or is done when it returns (204).
_WRITING_ACTIONS: dict[str, tuple[Callable[[WritingService], None], bool]] = {
    "install": (WritingService.install, True),
    "cancel": (WritingService.cancel_install, True),
    "retry": (WritingService.retry, True),
    "restart": (WritingService.restart_server, True),
    "open-log": (WritingService.open_log, False),
    "forget-style": (WritingService.forget_learned_style, False),
    "reset-learning": (WritingService.reset_learning, False),
}


def _read(parse: Callable[[object], _T], body: object, problem: str) -> _T:
    try:
        return parse(body)
    except ValueError as exc:
        raise ApiError(400, problem, str(exc)) from exc


def _check_shortcuts(bindings: Sequence[Binding]) -> None:
    try:
        validate_bindings(bindings)
    except BindingError as exc:
        raise ApiError(400, str(exc)) from exc


def _path_from(request: ApiRequest) -> Path:
    text = _read(_text_field("path"), request.body, "Say which file to use.")
    if not text.strip():
        raise ApiError(400, "Say which file to use.")
    return Path(text)


def _model_named(request: ApiRequest) -> str:
    name = request.params["name"]
    if name not in ALL_MODELS:
        raise ApiError(404, f"There is no model called '{name}'.")
    return name


def _text_field(key: str) -> Callable[[object], str]:
    return lambda body: Fields(body, "body").text(key)


def _flag_field(key: str) -> Callable[[object], bool]:
    return lambda body: Fields(body, "body").flag(key)


def _meter_request(body: object) -> tuple[str, bool]:
    fields = Fields(body, "body")
    return fields.text("device"), fields.flag("on")


def _ignoring_missing_download(command: Callable[[str], None], name: str) -> None:
    try:
        command(name)
    except LookupError:
        _logger.info("No download of '%s' to change.", name)


def _transforms_from_json(document: object) -> dict[str, str]:
    if not isinstance(document, list):
        raise ValueError("Transformations must be a list.")
    transforms: dict[str, str] = {}
    for index, row in enumerate(document):
        fields = Fields(row, f"transforms[{index}]")
        transforms[fields.text("name")] = fields.text("instruction")
    return transforms

