import math

import pytest

from voicetotext.recording.input_devices import (
    SYSTEM_DEFAULT,
    Microphones,
    full_names,
    input_device_index,
    input_device_names,
    loudness,
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
