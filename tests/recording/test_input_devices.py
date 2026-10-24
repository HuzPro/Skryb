import math

import pytest

from skryb.recording.input_devices import (
    DEFAULT_LABEL,
    SYSTEM_DEFAULT,
    Microphones,
    connected_preferred_microphone,
    full_names,
    input_device_index,
    input_device_names,
    loudness,
    preferred_microphone,
    short_microphone_name,
)

_MME = 0
_WASAPI = 2


def _device(name, hostapi=_MME, inputs=1, outputs=0) -> dict:
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


# Shaped like the real list on this machine: MME cuts names at 31
# characters, WASAPI repeats the same hardware in full.
_DEVICES = [
    _device("Microsoft Sound Mapper - Input"),
    _device("Microphone (Maono PD300X Microp"),
    _device("Microphone (DroidCam Audio)"),
    _device("Speakers (Realtek(R) Audio)", inputs=0, outputs=2),
    _device("Microphone (Maono PD300X Microp"),
    _device("   "),
    _device("Mikrofon (Gerät für Café ✓)"),
    _device("Microphone (Maono PD300X Microphone)", hostapi=_WASAPI),
    _device("Microphone (DroidCam Audio)", hostapi=_WASAPI),
]


def _query():
    return _DEVICES, _MME


class TestListing:
    def test_lists_each_microphone_once_in_device_order(self):
        assert input_device_names(_query) == [
            "Microphone (Maono PD300X Microp",
            "Microphone (DroidCam Audio)",
            "Mikrofon (Gerät für Café ✓)",
        ]

    def test_leaves_out_speakers_blank_names_and_the_default_alias(self):
        names = input_device_names(_query)

        assert "Speakers (Realtek(R) Audio)" not in names
        assert "Microsoft Sound Mapper - Input" not in names
        assert all(name.strip() for name in names)

    def test_lists_only_the_default_host_api(self):
        assert "Microphone (Maono PD300X Microphone)" not in input_device_names(_query)

    def test_copes_with_thousands_of_devices(self):
        devices = [_device(f"Mic {index % 700}") for index in range(5000)]

        names = input_device_names(lambda: (devices, _MME))

        assert len(names) == 700


class TestFullNames:
    def test_restores_a_name_the_host_api_cut_short(self):
        labels = full_names(["Microphone (Maono PD300X Microp"], _query)

        assert labels == {
            "Microphone (Maono PD300X Microp": "Microphone (Maono PD300X Microphone)",
        }

    def test_keeps_a_name_that_was_never_cut(self):
        assert full_names(["Mikrofon (Gerät für Café ✓)"], _query) == {
            "Mikrofon (Gerät für Café ✓)": "Mikrofon (Gerät für Café ✓)",
        }


class TestFindingTheDevice:
    def test_finds_the_first_matching_microphone_on_the_default_api(self):
        assert input_device_index("Microphone (Maono PD300X Microp", _query) == 1

    def test_the_default_is_never_a_specific_device(self):
        assert input_device_index(SYSTEM_DEFAULT, _query) is None

    @pytest.mark.parametrize(
        "name",
        ["Unplugged USB Mic", "Speakers (Realtek(R) Audio)", "Microphone (Maono PD300X Microphone)"],
    )
    def test_reports_none_for_a_microphone_it_cannot_record_from(self, name):
        assert input_device_index(name, _query) is None


class TestLoudness:
    @pytest.mark.parametrize(
        ("rms", "expected"),
        [(1.0, 1.0), (0.1, 2 / 3), (0.001, 0.0), (0.0, 0.0), (-1.0, 0.0), (math.nan, 0.0)],
    )
    def test_maps_level_onto_a_sixty_decibel_scale(self, rms, expected):
        assert loudness(rms) == pytest.approx(expected)


class TestRescanning:
    def _microphones(self, busy: bool) -> tuple[Microphones, list[str]]:
        rescans: list[str] = []
        microphones = Microphones(
            is_busy=lambda: busy, query=_query, refresh=lambda: rescans.append("rescan"),
        )
        return microphones, rescans

    def test_rescans_for_new_hardware_when_asked_and_idle(self):
        microphones, rescans = self._microphones(busy=False)

        microphones.names(rescan=True)

        assert rescans == ["rescan"]

    def test_never_rescans_mid_recording(self):
        microphones, rescans = self._microphones(busy=True)

        names = microphones.names(rescan=True)

        assert rescans == []
        assert names

    def test_lists_nothing_rather_than_failing_when_audio_is_broken(self):
        def broken():
            raise OSError("PortAudio not initialized")

        assert Microphones(query=broken).names() == []

    def test_labels_fall_back_to_the_raw_names_when_audio_is_broken(self):
        def broken():
            raise OSError("PortAudio not initialized")

        assert Microphones(query=broken).labels(["Mic A"]) == {"Mic A": "Mic A"}


def crowded_device_query():
    """A thousand-odd inputs: MME cuts, WASAPI full names, duplicates, blanks."""
    devices = [_device("Microsoft Sound Mapper - Input")]
    for index in range(1000):
        name = f"Microphone {index % 400} (Generic USB Audio Device)"
        devices.append(_device(name[:31]))
        devices.append(_device(name, hostapi=_WASAPI))
    devices += [
        _device(""),
        _device("   "),
        _device("Mikrofon (Gerät für Café ✓)"),
        _device("Microphone (Maono PD300X Microp"),
        _device("Microphone (Maono PD300X Microphone)", hostapi=_WASAPI),
        _device("Headset (Jabra Evolve 20 MS Ste"),
        _device("Headset (Jabra Evolve 20 MS Stereo)", hostapi=_WASAPI),
    ]
    return devices, _MME


class TestPreferredMicrophones:
    def test_the_first_connected_one_in_the_list_wins_by_its_full_name(self):
        preferred = [
            "Unplugged Blue Yeti",
            "Headset (Jabra Evolve 20 MS Stereo)",
            "Microphone (Maono PD300X Microphone)",
        ]

        chosen = connected_preferred_microphone(preferred, crowded_device_query)

        assert chosen == "Headset (Jabra Evolve 20 MS Ste"

    def test_a_name_saved_the_way_windows_cut_it_still_matches(self):
        chosen = connected_preferred_microphone(
            ["Microphone (Maono PD300X Microp"], crowded_device_query,
        )

        assert chosen == "Microphone (Maono PD300X Microp"

    def test_unicode_names_match_exactly(self):
        assert connected_preferred_microphone(
            ["Mikrofon (Gerät für Café ✓)"], crowded_device_query,
        ) == "Mikrofon (Gerät für Café ✓)"

    def test_nothing_connected_from_the_list_means_no_preference(self):
        assert connected_preferred_microphone(
            ["Unplugged Blue Yeti", "", "   "], crowded_device_query,
        ) is None

    def test_an_empty_list_never_touches_the_audio_stack(self):
        def untouchable():
            raise AssertionError("queried PortAudio for nothing")

        assert connected_preferred_microphone([], untouchable) is None

    def test_duplicate_devices_resolve_to_the_first_of_them(self):
        connected = ["Mic", "Mic", "Mic 2"]

        assert preferred_microphone(["Mic 2", "Mic"], connected, {}) == "Mic 2"
        assert preferred_microphone(["Mic"], connected, {}) == "Mic"

    def test_a_short_name_is_never_mistaken_for_a_longer_one(self):
        assert preferred_microphone(["Mic 2 (USB)"], ["Mic"], {"Mic": "Mic"}) is None


class TestShortNames:
    @pytest.mark.parametrize(
        ("name", "short"),
        [
            ("Microphone (Maono PD300X Microphone)", "Maono PD300X Microphone"),
            ("Microphone (Maono PD300X Microp", "Maono PD300X Microp"),
            ("Microphone (Realtek(R) Audio)", "Realtek(R) Audio"),
            ("Mikrofon (Gerät für Café ✓)", "Gerät für Café ✓"),
            ("Blue Yeti", "Blue Yeti"),
            ("Microphone ()", "Microphone ()"),
            (DEFAULT_LABEL, DEFAULT_LABEL),
        ],
    )
    def test_keeps_the_part_that_names_the_hardware(self, name, short):
        assert short_microphone_name(name) == short


class TestLabels:
    def test_the_windows_default_reads_as_such(self):
        assert Microphones(query=_query).label(SYSTEM_DEFAULT) == "Windows default"

    def test_a_cut_name_reads_in_full(self):
        assert Microphones(query=_query).label("Microphone (Maono PD300X Microp") == (
            "Microphone (Maono PD300X Microphone)"
        )
