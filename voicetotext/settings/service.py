from abc import ABC, abstractmethod

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
)
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import EngineHotSwap
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
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


_AUTO_LANGUAGE = "auto"


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
    ) -> None:
        self._config = config
        self._model_manager = model_manager
        self._event_bus = event_bus
        self._install_cache = install_cache
        self._downloads = downloads
        self._hot_swap = hot_swap
        self._download_listeners: list[DownloadEventListener] = []
        self._engine_listeners: list[EngineEventListener] = []

        self._event_bus.subscribe(DownloadProgressEvent, self._fanout_progress)
        self._event_bus.subscribe(DownloadStateChangedEvent, self._fanout_state)
        self._event_bus.subscribe(DownloadCompletedEvent, self._fanout_completed)
        self._event_bus.subscribe(DownloadFailedEvent, self._fanout_failed)
        self._event_bus.subscribe(EngineLoadStartedEvent, self._fanout_load_started)
        self._event_bus.subscribe(EngineLoadCompletedEvent, self._fanout_load_completed)
        self._event_bus.subscribe(EngineLoadFailedEvent, self._fanout_load_failed)

    # ------------------------------------------------------------------
    # Model queries
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelView]:
        return [
            ModelView.from_info(info, installed=self._install_cache.is_installed(info.name))
            for info in self._model_manager.list_models()
        ]

    def any_model_installed(self) -> bool:
        return any(
            self._install_cache.is_installed(info.name)
            for info in self._model_manager.list_models()
        )

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
        if name == self.selected_model():
            raise ValueError(f"cannot delete active model '{name}'")
        self._model_manager.delete(name)
        self._install_cache.invalidate(name)

    # ------------------------------------------------------------------
    # Draft lifecycle
    # ------------------------------------------------------------------

    def current_draft(self) -> SettingsDraft:
        language = self._config.get("transcription.language")
        return SettingsDraft(
            model_size=self._config.get("transcription.model_size", "base"),
            language=language if language else _AUTO_LANGUAGE,
            device=self._config.get("transcription.device", "auto"),
            hotkey_combination=self._config.get("hotkey.combination", "ctrl+alt+m"),
            hotkey_mode=self._config.get("hotkey.mode", "toggle"),
            output_mode=self._config.get("output.mode", "auto_paste"),
        )

    def apply_draft(self, draft: SettingsDraft) -> ChangeSet:
        original = self.current_draft()
        self._config.set("transcription.model_size", draft.model_size)
        self._config.set(
            "transcription.language",
            None if draft.language == _AUTO_LANGUAGE else draft.language,
        )
        self._config.set("transcription.device", draft.device)
        self._config.set("hotkey.combination", draft.hotkey_combination)
        self._config.set("hotkey.mode", draft.hotkey_mode)
        self._config.set("output.mode", draft.output_mode)
        self._config.save()
        return ChangeSet.between(original, draft)

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
