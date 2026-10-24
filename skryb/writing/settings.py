from dataclasses import dataclass

from skryb.core.config import ConfigurationStore

CLEANUP_LEVELS = ("off", "light", "polished")
MIN_DEADLINE_SECONDS = 0.5
MAX_DEADLINE_SECONDS = 30.0


@dataclass(frozen=True)
class WritingSettings:
    """Everything under ``writing.*`` in the config, as the Writing page edits it."""

    cleanup: str = "light"
    lists: bool = True
    deadline_seconds: float = 4.0
    context: bool = True
    learn_words: bool = True
    learn_style: bool = True
    style_notes: str = ""
    after_dictation_transform: str = ""
    runtime_path: str = ""
    model_path: str = ""

    @property
    def learns(self) -> bool:
        return self.learn_words or self.learn_style


_DEFAULTS = WritingSettings()


def read_writing_settings(config: ConfigurationStore) -> WritingSettings:
    cleanup = config.get("writing.cleanup", _DEFAULTS.cleanup)
    return WritingSettings(
        cleanup=cleanup if cleanup in CLEANUP_LEVELS else _DEFAULTS.cleanup,
        lists=_flag(config, "writing.lists", _DEFAULTS.lists),
        deadline_seconds=clamp_deadline(
            config.get("writing.deadline_seconds", _DEFAULTS.deadline_seconds),
        ),
        context=_flag(config, "writing.context", _DEFAULTS.context),
        learn_words=_flag(config, "writing.learn_words", _DEFAULTS.learn_words),
        learn_style=_flag(config, "writing.learn_style", _DEFAULTS.learn_style),
        style_notes=_text(config, "writing.style_notes"),
        after_dictation_transform=_text(config, "writing.after_dictation_transform"),
        runtime_path=_text(config, "writing.runtime_path").strip(),
        model_path=_text(config, "writing.model_path").strip(),
    )


def write_writing_settings(
    config: ConfigurationStore, settings: WritingSettings,
) -> None:
    config.set("writing.cleanup", settings.cleanup)
    config.set("writing.lists", settings.lists)
    config.set("writing.deadline_seconds", clamp_deadline(settings.deadline_seconds))
    config.set("writing.context", settings.context)
    config.set("writing.learn_words", settings.learn_words)
    config.set("writing.learn_style", settings.learn_style)
    config.set("writing.style_notes", settings.style_notes)
    config.set("writing.after_dictation_transform", settings.after_dictation_transform)
    config.set("writing.runtime_path", settings.runtime_path.strip())
    config.set("writing.model_path", settings.model_path.strip())


def clamp_deadline(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _DEFAULTS.deadline_seconds
    return min(max(float(value), MIN_DEADLINE_SECONDS), MAX_DEADLINE_SECONDS)


def _flag(config: ConfigurationStore, key: str, default: bool) -> bool:
    value = config.get(key, default)
    return value if isinstance(value, bool) else default


def _text(config: ConfigurationStore, key: str) -> str:
    value = config.get(key, "")
    return value if isinstance(value, str) else ""
