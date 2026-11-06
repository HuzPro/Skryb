"""The recording settings people change, as read from and written to config.

Each is read again at the moment it is used, so a change saved in Settings
applies from the next recording, sound or plugged-in microphone on.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from skryb.core.color_scheme import ColorScheme, detect_app_color_scheme, transparency_effects_on
from skryb.core.config import ConfigurationStore, ConfigValue
from skryb.recording.pill import (
    COLOURWAYS,
    DEFAULT_COLOURWAYS,
    DEFAULT_PILL_POSITION,
    PillPosition,
    PillSkin,
    pill_skin,
    without_glass,
)

_logger = logging.getLogger(__name__)

DUCK_MODES = ("off", "lower", "mute")


@dataclass(frozen=True)
class RecordingPreferences:
    warn_when_silent: bool = True
    # Full microphone names, most wanted first.
    preferred_devices: tuple[str, ...] = ()
    cues_enabled: bool = True
    cue_volume: float = 0.6
    duck_other_audio: str = "off"
    # The share of its volume another app keeps while it is lowered.
    duck_level: float = 0.2
    max_duration_seconds: float = 600.0
    warn_before_cap_seconds: float = 30.0

    @classmethod
    def from_config(cls, config: ConfigurationStore) -> "RecordingPreferences":
        default = cls()
        duck_mode = config.get("audio.duck_other_audio", default.duck_other_audio)
        return cls(
            warn_when_silent=config.get("audio.warn_when_silent", True) is not False,
            preferred_devices=_names(config.get("audio.preferred_devices", [])),
            cues_enabled=config.get("audio.cues_enabled", True) is not False,
            cue_volume=_number(
                config.get("audio.cue_volume"), default.cue_volume, 0.0, 1.0,
            ),
            duck_other_audio=(
                duck_mode if isinstance(duck_mode, str) and duck_mode in DUCK_MODES
                else default.duck_other_audio
            ),
            duck_level=_number(
                config.get("audio.duck_level"), default.duck_level, 0.0, 1.0,
            ),
            max_duration_seconds=_number(
                config.get("recording.max_duration_seconds"),
                default.max_duration_seconds, 1.0, math.inf,
            ),
            warn_before_cap_seconds=_number(
                config.get("recording.warn_before_cap_seconds"),
                default.warn_before_cap_seconds, 0.0, math.inf,
            ),
        )

    def save_to(self, config: ConfigurationStore) -> None:
        config.set("audio.warn_when_silent", self.warn_when_silent)
        config.set("audio.preferred_devices", list(self.preferred_devices))
        config.set("audio.cues_enabled", self.cues_enabled)
        config.set("audio.cue_volume", self.cue_volume)
        config.set("audio.duck_other_audio", self.duck_other_audio)
        config.set("audio.duck_level", self.duck_level)
        config.set("recording.max_duration_seconds", self.max_duration_seconds)
        config.set("recording.warn_before_cap_seconds", self.warn_before_cap_seconds)


def load_pill_position(config: ConfigurationStore) -> PillPosition:
    across, down = DEFAULT_PILL_POSITION
    return (
        _number(config.get("overlay.position.x"), across, 0.0, 1.0),
        _number(config.get("overlay.position.y"), down, 0.0, 1.0),
    )


def load_pill_skin(
    config: ConfigurationStore,
    scheme: Callable[[], ColorScheme] = detect_app_color_scheme,
    transparency: Callable[[], bool] = transparency_effects_on,
) -> PillSkin:
    """The pill in the window's theme and its colourway, and Windows' light or dark setting for apps.

    With Windows' transparency effects off, glass is solid, as Windows' own materials are.
    """
    theme = config.get("ui.theme", "default")
    colourway = load_colourways(config).get(theme) if isinstance(theme, str) else None
    skin = pill_skin(theme, scheme() is ColorScheme.DARK, colourway)
    return skin if transparency() else without_glass(skin)


def load_colourways(config: ConfigurationStore) -> dict[str, str]:
    """Each theme's colourway; a missing or unknown one is the theme's own."""
    return {
        theme: picked if isinstance(picked := config.get(f"ui.colourways.{theme}"), str) and picked in COLOURWAYS else default
        for theme, default in DEFAULT_COLOURWAYS.items()
    }


def active_colourway(config: ConfigurationStore) -> str:
    """The colourway of the theme in use; an unknown theme counts as the default one."""
    colourways = load_colourways(config)
    theme = config.get("ui.theme", "default")
    return colourways[theme if isinstance(theme, str) and theme in colourways else "default"]


def fold_paper_colourway(config: ConfigurationStore) -> bool:
    """Move a saved ``ui.paper_colourway`` into ``ui.colourways`` and drop it. True when config changed."""
    document = config.get_all()
    ui = document.get("ui")
    if not isinstance(ui, dict) or "paper_colourway" not in ui:
        return False
    settings = cast(dict[str, object], ui)
    paper = settings.pop("paper_colourway")
    saved = settings.get("colourways")
    colourways = cast(dict[str, object], saved) if isinstance(saved, dict) else {}
    if isinstance(paper, str) and paper in COLOURWAYS:
        colourways["paper"] = paper
    settings["colourways"] = colourways
    config.replace_all(document)
    return True


def save_pill_position(config: ConfigurationStore, position: PillPosition) -> None:
    try:
        _write_pill_position(config, position)
    except Exception:  # a pill that cannot remember its place still works
        _logger.warning("Saving the pill position failed.", exc_info=True)


def _write_pill_position(config: ConfigurationStore, position: PillPosition) -> None:
    config.set("overlay.position.x", position[0])
    config.set("overlay.position.y", position[1])
    config.save()


def _number(value: ConfigValue, default: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    if not math.isfinite(value):
        return default
    return min(max(float(value), low), high)


def _names(value: ConfigValue) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names = (name for name in value if isinstance(name, str) and name.strip())
    return tuple(dict.fromkeys(names))
