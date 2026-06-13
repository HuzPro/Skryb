from abc import ABC, abstractmethod
from collections.abc import Callable

from voicetotext.core.config import ConfigurationStore
from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
    ModelDeletedEvent,
    StateChangedEvent,
)
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import EngineHotSwap
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.transcription_mode import TranscriptionMode
from voicetotext.transcription.engine_registry import EngineRegistry
from voicetotext.transcription.engine_selector import (
    model_supports_batch as _model_supports_batch,
    model_supports_streaming as _model_supports_streaming,
    rank_engines,
)
from voicetotext.transcription.engine_types import EngineCandidate, HardwareProfile
from voicetotext.transcription.model_info import (
    ALL_MODELS,
    ModelInfo,
    allows_device_selection as _allows_device_selection,
)
from voicetotext.transcription.model_manager import ModelManager


class DownloadEventListener(ABC):
    """Callback interface for download lifecycle updates."""

    @abstractmethod
    def on_download_progress(self, event: DownloadProgressEvent) -> None: ...

    @abstractmethod
    def on_download_state(self, event: DownloadStateChangedEvent) -> None: ...

    @abstractmethod
    def on_download_completed(self, event: DownloadCompletedEvent) -> None: ...

    @abstractmethod
    def on_download_failed(self, event: DownloadFailedEvent) -> None: ...


class EngineEventListener(ABC):
    """Callback interface for engine load lifecycle updates."""

    @abstractmethod
    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None: ...

    @abstractmethod
    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None: ...

    @abstractmethod
    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None: ...


class StateChangeListener(ABC):
    """Callback interface for app state transitions (IDLE/RECORDING/...)."""

    @abstractmethod
    def on_state_changed(self, event: StateChangedEvent) -> None: ...


class SettingsService:
    """Facade (GoF) the settings UI talks to instead of domain objects.

    Coordinates :class:`ModelInstallationCache`,
    :class:`DownloadCoordinator`, and :class:`EngineHotSwapCoordinator`
    on behalf of the UI. Owns config read/write, draft assembly, and
    event fan-out to :class:`DownloadEventListener` and
    :class:`EngineEventListener` subscribers.

    The UI depends on this class alone — no imports from
    :mod:`voicetotext.core`, :mod:`voicetotext.transcription`, or
    :mod:`voicetotext.downloads` are needed in ``voicetotext/settings/``
    beyond this file and its collaborators.
    """

    def __init__(
        self,
        config: ConfigurationStore,
        model_manager: ModelManager,
        event_bus: EventBus,
        install_cache: ModelInstallationCache,
        downloads: DownloadCoordinator,
        hot_swap: EngineHotSwap,
        state_machine: AppStateMachine,
        registry: EngineRegistry | None = None,
        hardware: HardwareProfile | None = None,
    ) -> None:
        self._config = config
        self._model_manager = model_manager
        self._event_bus = event_bus
        self._install_cache = install_cache
        self._downloads = downloads
        self._hot_swap = hot_swap
        self._state_machine = state_machine
        self._registry = registry
        self._hardware = hardware
        self._download_listeners: list[DownloadEventListener] = []
        self._engine_listeners: list[EngineEventListener] = []
        self._state_listeners: list[StateChangeListener] = []

        self._event_bus.subscribe(DownloadProgressEvent, self._fanout_progress)
        self._event_bus.subscribe(DownloadStateChangedEvent, self._fanout_state)
        self._event_bus.subscribe(DownloadCompletedEvent, self._fanout_completed)
        self._event_bus.subscribe(DownloadFailedEvent, self._fanout_failed)
        self._event_bus.subscribe(EngineLoadStartedEvent, self._fanout_load_started)
        self._event_bus.subscribe(EngineLoadCompletedEvent, self._fanout_load_completed)
        self._event_bus.subscribe(EngineLoadFailedEvent, self._fanout_load_failed)
        self._event_bus.subscribe(StateChangedEvent, self._fanout_state_change)

    # ------------------------------------------------------------------
    # Model queries
    # ------------------------------------------------------------------

    def list_models(
        self, mode: TranscriptionMode | None = None,
    ) -> list[ModelView]:
        """List models, filtered to streaming-capable ones in streaming mode.

        *mode* lets the UI preview a not-yet-saved toggle; when omitted the
        persisted ``transcription.mode`` is used.
        """
        effective_mode = mode if mode is not None else self.transcription_mode()
        infos = self._model_manager.list_models()
        if effective_mode is TranscriptionMode.STREAMING:
            infos = [info for info in infos if self._is_streaming_model(info)]
        else:
            infos = [info for info in infos if self._is_batch_model(info)]
        return [
            ModelView.from_info(info, installed=self._install_cache.is_installed(info.name))
            for info in infos
        ]

    def transcription_mode(self) -> TranscriptionMode:
        return TranscriptionMode.migrate(self._config.get("transcription.mode"))

    def model_supports_streaming(self, model_name: str) -> bool:
        """Whether *model_name* can be driven by a streaming engine."""
        info = ALL_MODELS.get(model_name)
        return info is not None and self._is_streaming_model(info)

    def model_supports_batch(self, model_name: str) -> bool:
        """Whether *model_name* can be driven by a batch engine."""
        info = ALL_MODELS.get(model_name)
        return info is not None and self._is_batch_model(info)

    def _is_streaming_model(self, info: ModelInfo) -> bool:
        if self._registry is None:
            return False
        return _model_supports_streaming(info, self._registry.all_descriptors())

    def _is_batch_model(self, info: ModelInfo) -> bool:
        if self._registry is None:
            return True
        return _model_supports_batch(info, self._registry.all_descriptors())

    def any_model_installed(self) -> bool:
        return any(
            self._install_cache.is_installed(info.name)
            for info in self._model_manager.list_models()
        )

    def is_installed(self, name: str) -> bool:
        return self._install_cache.is_installed(name)

    def selected_model(self) -> str:
        return self._config.get("transcription.model_size", "base")

    # ------------------------------------------------------------------
    # Download commands
    # ------------------------------------------------------------------

    def start_download(self, name: str) -> None:
        self._downloads.start(name)

    def pause_download(self, name: str) -> None:
        self._downloads.pause(name)

    def resume_download(self, name: str) -> None:
        self._downloads.resume(name)

    def cancel_download(self, name: str) -> None:
        self._downloads.cancel(name)

    def delete_model(self, name: str) -> None:
        self._reject_if_active(name)
        self._model_manager.delete(name)
        self._install_cache.invalidate(name)
        self._event_bus.publish(ModelDeletedEvent(name=name))

    def _reject_if_active(self, name: str) -> None:
        if name == self.selected_model():
            raise ValueError(f"cannot delete active model '{name}'")

    # ------------------------------------------------------------------
    # Draft lifecycle
    # ------------------------------------------------------------------

    def current_draft(self) -> SettingsDraft:
        return SettingsDraft(
            model_size=self._config.get("transcription.model_size", "base"),
            language=self._config.get("transcription.language", "en"),
            device=self._config.get("transcription.device", "auto"),
            engine=self._config.get("transcription.engine", "auto"),
            hotkey_combination=self._config.get("hotkey.combination", "ctrl+alt+m"),
            hotkey_mode=self._config.get("hotkey.mode", "toggle"),
            output_mode=OutputMode.migrate(self._config.get("output.mode")),
            transcription_mode=TranscriptionMode.migrate(
                self._config.get("transcription.mode")
            ),
        )

    def apply_draft(self, draft: SettingsDraft) -> ChangeSet:
        original = self.current_draft()
        self._config.set("transcription.model_size", draft.model_size)
        self._config.set("transcription.language", draft.language)
        self._config.set("transcription.device", draft.device)
        self._config.set("transcription.engine", draft.engine)
        self._config.set("hotkey.combination", draft.hotkey_combination)
        self._config.set("hotkey.mode", draft.hotkey_mode)
        self._config.set("output.mode", draft.output_mode.value)
        self._config.set("transcription.mode", draft.transcription_mode.value)
        self._config.save()
        self._invalidate_cache_on_engine_change(original, draft)
        return ChangeSet.between(original, draft)

    def _invalidate_cache_on_engine_change(
        self, original: SettingsDraft, updated: SettingsDraft,
    ) -> None:
        """Drop cached ``is_installed`` results when the engine changed.

        Cache keys are model names — engine-agnostic — but the truth
        they cache (``CompositeModelManager.is_installed``) now routes
        to the active engine's repo. A stale ``True`` from before the
        swap would tell the loader a model is ready when its files live
        in a sibling engine's cache, surfacing as the original
        ``LocalEntryNotFoundError``.
        """
        if original.engine != updated.engine:
            self._install_cache.clear()

    # ------------------------------------------------------------------
    # Engine + device queries (hardware-aware)
    # ------------------------------------------------------------------

    def engine_candidates_for(
        self, model_name: str,
    ) -> tuple[EngineCandidate, ...]:
        """Return engines that can drive *model_name* on this hardware.

        Empty when registry/hardware aren't wired (first-run path).
        Otherwise a ranked tuple — the first entry has
        ``is_recommended=True`` and the reason string says why.
        """
        if self._registry is None or self._hardware is None:
            return ()
        info = ALL_MODELS.get(model_name)
        if info is None:
            return ()
        return rank_engines(info, self._hardware, self._registry.all_descriptors())

    def available_devices(self) -> list[str]:
        """Return device choices applicable to the detected hardware.

        Always includes ``auto`` and ``cpu``; adds ``cuda`` only when a
        GPU was detected, so a CPU-only box doesn't show a non-functional
        option.
        """
        if self._hardware is None or self._hardware.kind == "cpu":
            return ["auto", "cpu"]
        return ["auto", "cuda", "cpu"]

    def supports_streaming_hardware(self) -> bool:
        """Whether the detected hardware can run the GPU-only streaming engine.

        Streaming requires a CUDA or ROCm GPU; a CPU-only box (or one whose
        hardware hasn't been probed yet) cannot load a streaming model, so the
        UI uses this to keep the Streaming mode from being selected there.
        """
        return self._hardware is not None and self._hardware.kind != "cpu"

    def allows_device_selection(self, model_name: str) -> bool:
        """Whether the device picker should be enabled for *model_name*.

        Delegates to the model catalog so the settings UI never imports
        the transcription domain directly (see class docstring).
        """
        return _allows_device_selection(model_name)

    # ------------------------------------------------------------------
    # Engine hot-swap
    # ------------------------------------------------------------------

    def trigger_engine_swap(self) -> None:
        if self._hot_swap.is_busy():
            return
        self._hot_swap.start_swap(self.selected_model())

    def is_engine_swapping(self) -> bool:
        return self._hot_swap.is_busy()

    # ------------------------------------------------------------------
    # App-state guard
    # ------------------------------------------------------------------

    def can_apply_now(self) -> bool:
        """Return ``True`` iff it is safe to apply settings right now.

        Applying settings during :attr:`AppState.RECORDING` or
        :attr:`AppState.TRANSCRIBING` would either steal audio from an
        in-flight capture or be rejected by the engine hot-swap (which
        requires :attr:`AppState.IDLE`). Save-button UX consults this
        before issuing commands.
        """
        state = self._state_machine.state
        return state not in (AppState.RECORDING, AppState.TRANSCRIBING)

    # ------------------------------------------------------------------
    # Listener fan-out
    # ------------------------------------------------------------------

    def subscribe_downloads(self, listener: DownloadEventListener) -> None:
        self._download_listeners.append(listener)

    def unsubscribe_downloads(self, listener: DownloadEventListener) -> None:
        if listener in self._download_listeners:
            self._download_listeners.remove(listener)

    def subscribe_engine(self, listener: EngineEventListener) -> None:
        self._engine_listeners.append(listener)

    def unsubscribe_engine(self, listener: EngineEventListener) -> None:
        if listener in self._engine_listeners:
            self._engine_listeners.remove(listener)

    def subscribe_state_changes(self, listener: StateChangeListener) -> None:
        self._state_listeners.append(listener)

    def unsubscribe_state_changes(self, listener: StateChangeListener) -> None:
        if listener in self._state_listeners:
            self._state_listeners.remove(listener)

    def subscribe_model_deleted(
        self, handler: Callable[[ModelDeletedEvent], None],
    ) -> None:
        """Register *handler* to be invoked on every :class:`ModelDeletedEvent`.

        The settings window uses this to refresh the affected row when a
        model is removed. Subscriptions are reference-counted by the
        underlying :class:`EventBus`, so the window must call
        :meth:`unsubscribe_model_deleted` on close.
        """
        self._event_bus.subscribe(ModelDeletedEvent, handler)

    def unsubscribe_model_deleted(
        self, handler: Callable[[ModelDeletedEvent], None],
    ) -> None:
        self._event_bus.unsubscribe(ModelDeletedEvent, handler)

    def _fanout_progress(self, event: DownloadProgressEvent) -> None:
        for listener in list(self._download_listeners):
            listener.on_download_progress(event)

    def _fanout_state(self, event: DownloadStateChangedEvent) -> None:
        for listener in list(self._download_listeners):
            listener.on_download_state(event)

    def _fanout_completed(self, event: DownloadCompletedEvent) -> None:
        for listener in list(self._download_listeners):
            listener.on_download_completed(event)

    def _fanout_failed(self, event: DownloadFailedEvent) -> None:
        for listener in list(self._download_listeners):
            listener.on_download_failed(event)

    def _fanout_load_started(self, event: EngineLoadStartedEvent) -> None:
        for listener in list(self._engine_listeners):
            listener.on_engine_load_started(event)

    def _fanout_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        for listener in list(self._engine_listeners):
            listener.on_engine_load_completed(event)

    def _fanout_load_failed(self, event: EngineLoadFailedEvent) -> None:
        for listener in list(self._engine_listeners):
            listener.on_engine_load_failed(event)

    def _fanout_state_change(self, event: StateChangedEvent) -> None:
        for listener in list(self._state_listeners):
            listener.on_state_changed(event)
