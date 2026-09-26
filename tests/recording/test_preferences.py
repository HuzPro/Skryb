import json
import math

import pytest

from tests.fakes.config import FakeConfig
from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.recording.pill import DEFAULT_PILL_POSITION
from voicetotext.recording.preferences import (
    RecordingPreferences,
    load_pill_position,
    save_pill_position,
)


class TestRecordingPreferences:
    def test_a_fresh_config_gives_the_documented_defaults(self):
        preferences = RecordingPreferences.from_config(FakeConfig())

        assert preferences == RecordingPreferences(
            warn_when_silent=True,
            preferred_devices=(),
            cues_enabled=True,
            cue_volume=0.6,
            duck_other_audio="off",
            duck_level=0.2,
            max_duration_seconds=600.0,
            warn_before_cap_seconds=30.0,
        )

    def test_a_saved_set_reads_back_the_same(self):
        config = FakeConfig()
        chosen = RecordingPreferences(
            warn_when_silent=False,
            preferred_devices=("Microphone (Maono PD300X Microphone)", "Gerät für Café ✓"),
            cues_enabled=False,
            cue_volume=0.35,
            duck_other_audio="mute",
            duck_level=0.1,
            max_duration_seconds=90.0,
            warn_before_cap_seconds=0.0,
        )

        chosen.save_to(config)

        assert RecordingPreferences.from_config(config) == chosen

    def test_they_land_under_the_documented_keys_in_the_real_file(self, tmp_path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()

        RecordingPreferences(preferred_devices=("Blue Yeti",)).save_to(config)
        config.save()

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["audio"]["preferred_devices"] == ["Blue Yeti"]
        assert saved["audio"]["duck_other_audio"] == "off"
        assert saved["recording"]["warn_before_cap_seconds"] == 30.0
        assert saved["overlay"]["position"] == {"x": 0.5, "y": 1.0}

    def test_the_preferred_list_drops_blanks_duplicates_and_non_names(self):
        config = FakeConfig()
        config.set("audio.preferred_devices", [
            "Blue Yeti", "", "   ", "Blue Yeti", "Mikrofon (Gerät für Café ✓)",
        ])

        assert RecordingPreferences.from_config(config).preferred_devices == (
            "Blue Yeti", "Mikrofon (Gerät für Café ✓)",
        )

    @pytest.mark.parametrize(
        ("key", "value", "field", "expected"),
        [
            ("audio.cue_volume", "loud", "cue_volume", 0.6),
            ("audio.cue_volume", 7.0, "cue_volume", 1.0),
            ("audio.cue_volume", -1, "cue_volume", 0.0),
            ("audio.cue_volume", True, "cue_volume", 0.6),
            ("audio.duck_level", math.nan, "duck_level", 0.2),
            ("audio.duck_other_audio", "LOUDER", "duck_other_audio", "off"),
            ("audio.duck_other_audio", 3, "duck_other_audio", "off"),
            ("recording.max_duration_seconds", 0, "max_duration_seconds", 1.0),
            ("recording.max_duration_seconds", math.inf, "max_duration_seconds", 600.0),
            ("recording.max_duration_seconds", 7200, "max_duration_seconds", 7200.0),
            ("recording.warn_before_cap_seconds", -30, "warn_before_cap_seconds", 0.0),
            ("audio.preferred_devices", "Blue Yeti", "preferred_devices", ()),
        ],
    )
    def test_hand_edited_nonsense_falls_back_to_something_safe(
        self, key, value, field, expected,
    ):
        config = FakeConfig()
        config.set(key, value)

        assert getattr(RecordingPreferences.from_config(config), field) == expected


class TestPillPositionStorage:
    def test_an_unset_position_is_the_bottom_centre(self):
        assert load_pill_position(FakeConfig()) == DEFAULT_PILL_POSITION

    def test_a_saved_position_is_written_straight_away(self):
        config = FakeConfig()

        save_pill_position(config, (0.25, 0.75))

        assert load_pill_position(config) == (0.25, 0.75)
        assert config.save_count == 1

    @pytest.mark.parametrize("value", ["left", None, math.nan, 12.0, -0.5])
    def test_a_garbled_position_stays_on_screen(self, value):
        config = FakeConfig()
        config.set("overlay.position.x", value)

        across, down = load_pill_position(config)

        assert 0.0 <= across <= 1.0
        assert down == 1.0

    def test_a_config_that_cannot_be_written_does_not_break_the_pill(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"overlay": {"position": "bottom"}}), encoding="utf-8")
        config = JsonConfigAdapter(path)
        config.load()

        save_pill_position(config, (0.3, 0.3))

        assert load_pill_position(config) == DEFAULT_PILL_POSITION
