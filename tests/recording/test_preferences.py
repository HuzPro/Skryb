import json
import math

import pytest

from tests.fakes.config import FakeConfig
from skryb.core.color_scheme import ColorScheme
from skryb.core.json_config_adapter import JsonConfigAdapter
from skryb.recording.pill import DEFAULT_PILL_POSITION, DEFAULT_SKIN, pill_skin, without_glass
from skryb.recording.preferences import (
    RecordingPreferences,
    active_colourway,
    fold_paper_colourway,
    load_colourways,
    load_pill_position,
    load_pill_skin,
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


class TestPillSkin:
    def test_the_pill_follows_the_saved_theme_and_windows_light_or_dark_setting(self):
        for theme in ("default", "glass", "material", "paper"):
            config = FakeConfig({"ui.theme": theme})

            assert load_pill_skin(config, lambda: ColorScheme.DARK, lambda: True) == pill_skin(theme, True)
            assert load_pill_skin(config, lambda: ColorScheme.LIGHT, lambda: True) == pill_skin(theme, False)

    def test_with_windows_transparency_effects_off_glass_is_solid_and_the_rest_are_unchanged(self):
        for theme in ("default", "glass", "material", "paper"):
            config = FakeConfig({"ui.theme": theme})

            skin = load_pill_skin(config, lambda: ColorScheme.DARK, lambda: False)

            assert skin == without_glass(pill_skin(theme, True))
            assert not skin.glass

    def test_the_pill_follows_the_saved_colourway_of_the_theme_in_use(self):
        colourways = {"default": "plum", "glass": "terracotta", "material": "green", "paper": "ink"}
        for theme, colourway in colourways.items():
            config = FakeConfig({"ui.theme": theme, "ui.colourways": colourways})

            assert load_pill_skin(config, lambda: ColorScheme.DARK) == pill_skin(theme, True, colourway)
            assert load_pill_skin(config, lambda: ColorScheme.LIGHT) == pill_skin(theme, False, colourway)

    def test_a_missing_or_hand_edited_colourway_is_the_themes_own(self):
        for saved in (None, 7, "plum", ["ink"], {"paper": "Ink blue 🖋"}, {"paper": 3}):
            values = {"ui.theme": "paper"} if saved is None else {"ui.theme": "paper", "ui.colourways": saved}

            assert load_pill_skin(FakeConfig(values), lambda: ColorScheme.LIGHT) == pill_skin("paper", False, "green")

    def test_a_fresh_or_hand_edited_config_gets_the_default_pill_in_ink_blue(self):
        for values in ({}, {"ui.theme": 42}, {"ui.theme": "Glass ✨"}, {"ui.theme": ["glass"]}):
            assert load_pill_skin(FakeConfig(values), lambda: ColorScheme.DARK) == pill_skin("default", True, "ink")


class TestColourways:
    def test_a_fresh_config_gives_each_theme_its_own_colourway(self):
        assert load_colourways(FakeConfig()) == {"default": "ink", "glass": "ink", "material": "ink", "paper": "green"}

    def test_saved_colourways_come_back_and_unknown_ones_or_themes_are_left_out(self):
        saved = {"glass": "plum", "material": "Teal", "paper": "graphite", "neon": "green", "default": None}

        assert load_colourways(FakeConfig({"ui.colourways": saved})) == {
            "default": "ink", "glass": "plum", "material": "ink", "paper": "graphite",
        }

    def test_the_active_colourway_is_the_one_saved_for_the_theme_in_use(self):
        saved = {"default": "plum", "glass": "terracotta", "material": "graphite", "paper": "ink"}
        for theme, colourway in saved.items():
            assert active_colourway(FakeConfig({"ui.theme": theme, "ui.colourways": saved})) == colourway

    def test_a_fresh_or_hand_edited_theme_takes_the_default_themes_colourway(self):
        for values in ({}, {"ui.theme": 42}, {"ui.theme": "Glass ✨"}, {"ui.theme": ["glass"]}, {"ui.theme": ""}):
            assert active_colourway(FakeConfig(values)) == "ink"
            assert active_colourway(FakeConfig({**values, "ui.colourways": {"default": "plum"}})) == "plum"

    def test_an_old_paper_colourway_moves_into_the_colourways_and_the_old_key_goes(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"ui": {"theme": "paper", "paper_colourway": "plum"}}), encoding="utf-8")
        config = JsonConfigAdapter(path)
        config.load()

        assert fold_paper_colourway(config) is True

        assert config.get("ui.paper_colourway") is None
        assert config.get("ui.colourways") == {"default": "ink", "glass": "ink", "material": "ink", "paper": "plum"}
        assert load_colourways(config)["paper"] == "plum"
        assert config.get("ui.theme") == "paper"

    def test_an_old_paper_colourway_this_version_does_not_know_is_dropped_and_paper_stays_green(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"ui": {"paper_colourway": "Neon pink 🌸", "colourways": {"glass": "plum"}}}), encoding="utf-8")
        config = JsonConfigAdapter(path)
        config.load()

        assert fold_paper_colourway(config) is True

        assert config.get("ui.paper_colourway") is None
        assert load_colourways(config) == {"default": "ink", "glass": "plum", "material": "ink", "paper": "green"}

    def test_a_config_already_on_colourways_is_left_alone(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()
        before = config.get_all()

        assert fold_paper_colourway(config) is False
        assert config.get_all() == before

    def test_a_theme_saved_later_is_read_the_next_time_the_pill_asks(self):
        config = FakeConfig({"ui.theme": "default"})
        assert load_pill_skin(config, lambda: ColorScheme.LIGHT) == pill_skin("default", False)

        config.set("ui.theme", "paper")

        assert load_pill_skin(config, lambda: ColorScheme.LIGHT) == pill_skin("paper", False)
