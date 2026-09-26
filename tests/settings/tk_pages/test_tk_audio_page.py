import pytest

from voicetotext.recording.preferences import RecordingPreferences
from voicetotext.settings.tk_pages.audio_page import TkAudioPage


class FakeMicrophones:
    def __init__(self, names: list[str], labels: dict[str, str] | None = None) -> None:
        self.connected = list(names)
        self.full = labels or {}
        self.rescans = 0
        self.meters: list[str] = []
        self.meter_open = False
        self.on_level = None
        self.opens = True

    def names(self, rescan: bool = False) -> list[str]:
        if rescan:
            self.rescans += 1
        return list(self.connected)

    def labels(self, names):
        return {name: self.full.get(name, name) for name in names}

    def start_meter(self, name, on_level) -> bool:
        self.meters.append(name)
        self.meter_open = self.opens
        self.on_level = on_level
        return self.opens

    def stop_meter(self) -> None:
        self.meter_open = False


_MAONO = "Microphone (Maono PD300X Microp"


@pytest.fixture
def microphones():
    return FakeMicrophones(
        [_MAONO, "Mikrofon (Gerät für Café ✓)"],
        {_MAONO: "Microphone (Maono PD300X Microphone)"},
    )


@pytest.fixture
def page(tk_root, microphones):
    widget = TkAudioPage(tk_root, microphones)
    yield widget
    widget.destroy()


def _pick(page, label: str) -> None:
    page._choose(label)
    page.update()


class TestChoosing:
    def test_offers_the_default_first_then_full_microphone_names(self, page):
        assert page.microphone_labels() == [
            "Windows default",
            "Microphone (Maono PD300X Microphone)",
            "Mikrofon (Gerät für Café ✓)",
        ]

    def test_starts_on_the_windows_default(self, page):
        assert page.current_device() == ""

    def test_saves_the_name_the_recorder_looks_up_not_the_display_name(self, page):
        _pick(page, "Microphone (Maono PD300X Microphone)")

        assert page.current_device() == _MAONO

    def test_shows_the_saved_microphone_when_opened(self, page):
        page.set_device(_MAONO)

        assert page._choice.get() == "Microphone (Maono PD300X Microphone)"

    def test_keeps_an_unplugged_choice_so_saving_does_not_forget_it(self, page):
        page.set_device("USB Mic")

        assert page._choice.get() == "USB Mic (not connected)"
        assert page.current_device() == "USB Mic"

    def test_refresh_looks_for_newly_plugged_microphones(self, page, microphones):
        microphones.connected.append("USB Mic")

        page._fill(rescan=True)

        assert microphones.rescans == 1
        assert "USB Mic" in page.microphone_labels()


class TestLevelMeter:
    def test_only_listens_while_the_page_is_on_screen(self, page, microphones):
        assert microphones.meters == []

        page.on_shown()
        assert microphones.meter_open
        page.on_hidden()

        assert not microphones.meter_open

    def test_listens_to_the_newly_picked_microphone(self, page, microphones):
        page.on_shown()

        _pick(page, "Microphone (Maono PD300X Microphone)")

        assert microphones.meters == ["", _MAONO]
        page.on_hidden()

    def test_moves_with_the_voice_and_falls_back_when_it_stops(self, page, microphones):
        page.on_shown()

        microphones.on_level(0.8)
        page._poll()
        loud = float(page._meter["value"])
        for _ in range(20):
            page._poll()

        assert loud == pytest.approx(0.8)
        assert float(page._meter["value"]) == pytest.approx(0.0)
        page.on_hidden()

    def test_says_so_when_a_microphone_will_not_open(self, page, microphones):
        microphones.opens = False

        page.on_shown()

        assert "could not be opened" in page._meter_status.cget("text")
        assert page._no_signal.winfo_manager() == "grid"
        assert page._no_signal.cget("text") == "No signal"
        assert not page._meter.winfo_manager()
        page.on_hidden()

    def test_the_meter_comes_back_when_another_microphone_opens(self, page, microphones):
        microphones.opens = False
        page.on_shown()

        microphones.opens = True
        _pick(page, "Mikrofon (Gerät für Café ✓)")

        assert page._meter.winfo_manager() == "grid"
        assert not page._no_signal.winfo_manager()
        page.on_hidden()

    def test_closing_the_page_stops_the_meter(self, tk_root, microphones):
        widget = TkAudioPage(tk_root, microphones)
        widget.on_shown()

        widget.destroy()

        assert not microphones.meter_open


class TestBoost:
    def test_is_on_by_default(self, page):
        assert page.boosts_quiet_speech() is True

    def test_round_trips(self, page):
        page.set_boost_quiet_speech(False)

        assert page.boosts_quiet_speech() is False


def _listed(page) -> list[str]:
    return page.preferred_labels()


class TestPreferredMicrophones:
    def test_add_selected_adds_the_picked_microphone_by_its_full_name(self, page):
        _pick(page, "Microphone (Maono PD300X Microphone)")

        page._add_selected()
        page._add_selected()

        assert page.recording_preferences().preferred_devices == (
            "Microphone (Maono PD300X Microphone)",
        )

    def test_the_windows_default_is_not_a_microphone_to_prefer(self, page):
        page._add_selected()

        assert page.recording_preferences().preferred_devices == ()

    def test_up_down_and_remove_reorder_the_list(self, page):
        page.set_recording_preferences(RecordingPreferences(
            preferred_devices=("Blue Yeti", "Gerät für Café ✓", "Jabra Evolve"),
        ))

        page._move_preferred(2, -1)
        page._move_preferred(1, -1)
        page._move_preferred(0, -1)
        page._remove_preferred(1)
        page._move_preferred(0, 1)
        page._move_preferred(1, 1)

        assert page.recording_preferences().preferred_devices == (
            "Gerät für Café ✓", "Jabra Evolve",
        )

    def test_moving_past_either_end_or_removing_nothing_changes_nothing(self, page):
        page.set_recording_preferences(RecordingPreferences(
            preferred_devices=("Blue Yeti", "Jabra Evolve"),
        ))

        page._move_preferred(0, -1)
        page._move_preferred(1, 1)
        page._move_preferred(5, -1)
        page._remove_preferred(7)

        assert page.recording_preferences().preferred_devices == (
            "Blue Yeti", "Jabra Evolve",
        )

    def test_microphones_that_are_not_plugged_in_say_so(self, page):
        page.set_recording_preferences(RecordingPreferences(
            preferred_devices=(
                "Microphone (Maono PD300X Microphone)",
                "Mikrofon (Gerät für Café ✓)",
                "Blue Yeti",
            ),
        ))

        assert _listed(page) == [
            "Microphone (Maono PD300X Microphone)",
            "Mikrofon (Gerät für Café ✓)",
            "Blue Yeti (not connected)",
        ]

    def test_refreshing_updates_which_ones_are_connected(self, page, microphones):
        page.set_recording_preferences(RecordingPreferences(preferred_devices=("Blue Yeti",)))
        microphones.connected.append("Blue Yeti")

        page._fill(rescan=True)

        assert _listed(page) == ["Blue Yeti"]


class TestSoundsAndLimit:
    def test_every_setting_round_trips_through_the_page(self, page):
        chosen = RecordingPreferences(
            warn_when_silent=False,
            preferred_devices=("Blue Yeti",),
            cues_enabled=False,
            cue_volume=0.25,
            duck_other_audio="lower",
            duck_level=0.35,
            max_duration_seconds=1200.0,
            warn_before_cap_seconds=45.0,
        )

        page.set_recording_preferences(chosen)

        assert page.recording_preferences() == chosen

    def test_the_limit_is_shown_in_minutes_and_saved_in_seconds(self, page):
        page.set_recording_preferences(RecordingPreferences(max_duration_seconds=90.0))
        assert page._limit_minutes.get() == "1.5"

        page._limit_minutes.set("25")

        assert page.recording_preferences().max_duration_seconds == 1500.0

    @pytest.mark.parametrize(
        ("minutes", "warn", "expected"),
        [
            ("ten", "", (600.0, 30.0)),
            ("0", "-5", (60.0, 0.0)),
            ("999", "9999", (3600.0, 300.0)),
            ("nan", "inf", (600.0, 30.0)),
            (" 12 ", "15", (720.0, 15.0)),
        ],
    )
    def test_typed_nonsense_keeps_or_clamps_the_value(self, page, minutes, warn, expected):
        page.set_recording_preferences(RecordingPreferences())

        page._limit_minutes.set(minutes)
        page._warn_seconds.set(warn)
        preferences = page.recording_preferences()

        assert (
            preferences.max_duration_seconds, preferences.warn_before_cap_seconds,
        ) == expected

    def test_a_setting_the_page_does_not_show_is_kept_as_it_was(self, page):
        page.set_recording_preferences(RecordingPreferences(duck_level=0.05))

        page._duck.set("mute")

        assert page.recording_preferences().duck_level == 0.05
        assert page.recording_preferences().duck_other_audio == "mute"
