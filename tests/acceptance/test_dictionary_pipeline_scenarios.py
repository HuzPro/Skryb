from pathlib import Path

from tests.fakes.config import FakeConfig
from tests.fakes.model_manager import FakeModelManager
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.state_machine import AppStateMachine
from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryOptions,
    LiteralRule,
)
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.text_filter_decorator import (
    TextFilterDecorator,
    load_dictionary,
)
from voicetotext.output.text_pipeline import TextPipeline
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import NullEngineHotSwap
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.service import SettingsService


class CapturingOutput(TextOutputHandler):
    def __init__(self) -> None:
        self.values: list[str] = []

    def output(self, text: str) -> None:
        self.values.append(text)


def _settings_service(dictionary_path: Path) -> SettingsService:
    event_bus = SimpleEventBus()
    model_manager = FakeModelManager()
    return SettingsService(
        config=FakeConfig(),
        model_manager=model_manager,
        event_bus=event_bus,
        install_cache=ModelInstallationCache(model_manager, event_bus),
        downloads=DownloadCoordinator(model_manager),
        hot_swap=NullEngineHotSwap(),
        state_machine=AppStateMachine(event_bus),
        dictionary_path=dictionary_path,
    )


def test_should_apply_saved_dictionary_after_output_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "dictionary.json"
    service = _settings_service(path)
    service.save_dictionary_rules(DictionaryConfig(
        options=DictionaryOptions(spoken_commands=True),
        rules=(LiteralRule(("git hub", "get hub"), "GitHub"),),
    ))
    output = CapturingOutput()
    rebuilt = TextFilterDecorator(
        output, TextPipeline.for_batch(load_dictionary(path)),
    )

    rebuilt.output("get hub comma next")

    assert output.values == ["GitHub, next"]
