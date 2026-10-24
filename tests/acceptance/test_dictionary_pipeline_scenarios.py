from pathlib import Path

from tests.fakes.config import FakeConfig
from tests.fakes.licensing import AllowAllLicensePolicy
from tests.fakes.model_manager import FakeModelManager
from skryb.core.event_bus import SimpleEventBus
from skryb.core.state_machine import AppStateMachine
from skryb.output.dictionary_config import (
    DictionaryConfig,
    DictionaryOptions,
    LiteralRule,
)
from skryb.output.handler import TextOutputHandler
from skryb.output.text_filter_decorator import (
    TextFilterDecorator,
    load_dictionary,
)
from skryb.output.text_pipeline import TextPipeline
from skryb.settings.download_coordinator import DownloadCoordinator
from skryb.settings.engine_hot_swap import NullEngineHotSwap
from skryb.settings.model_installation_cache import ModelInstallationCache
from skryb.settings.service import SettingsService


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
        license_policy=AllowAllLicensePolicy(),
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
