from tests.fakes.config import FakeConfig
from skryb.core.config import ConfigValue


_DEFAULTS: dict[str, ConfigValue] = {
    "transcription.model_size": "parakeet-tdt-0.6b-v3",
    "transcription.language": "en",
    "transcription.device": "auto",
    "hotkey.bindings": [
        {"keys": "ctrl+alt+m", "mode": "toggle", "action": "dictate"},
    ],
    "output.mode": "direct_paste",
    "audio.sample_rate": 16000,
}


class _ConfigBuilder:
    """Fluent builder that produces a pre-populated :class:`FakeConfig`.

    Defaults mirror a healthy first-run config so individual tests
    only override the keys they care about.
    """

    def __init__(self) -> None:
        self._overrides: dict[str, ConfigValue] = {}

    def with_setting(self, key: str, value: ConfigValue) -> "_ConfigBuilder":
        self._overrides[key] = value
        return self

    def build(self) -> FakeConfig:
        merged: dict[str, ConfigValue] = dict(_DEFAULTS)
        merged.update(self._overrides)
        return FakeConfig(merged)


def a_config() -> _ConfigBuilder:
    return _ConfigBuilder()
