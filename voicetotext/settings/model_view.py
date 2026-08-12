from dataclasses import dataclass

from voicetotext.transcription.model_info import ModelInfo


@dataclass(frozen=True)
class ModelView:
    """UI-friendly projection of :class:`ModelInfo` + installation state.

    The settings UI never touches :class:`ModelInfo` directly - it renders
    :class:`ModelView` instances. Keeping a dedicated view object lets the
    facade pre-compute installation status once and hand the UI a snapshot
    that doesn't require any further domain calls.
    """

    name: str
    display_size: str
    description: str
    installed: bool

    @classmethod
    def from_info(cls, info: ModelInfo, installed: bool) -> "ModelView":
        return cls(
            name=info.name,
            display_size=info.display_size,
            description=info.description,
            installed=installed,
        )
