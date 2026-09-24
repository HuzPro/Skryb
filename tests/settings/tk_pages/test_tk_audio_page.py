import pytest

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
    page._choice.set(label)
    page._picker.event_generate("<<ComboboxSelected>>")
    page.update()


class TestChoosing:
    def test_offers_the_default_first_then_full_microphone_names(self, page):
        assert list(page._picker["values"]) == [
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
        assert "USB Mic" in page._picker["values"]


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
