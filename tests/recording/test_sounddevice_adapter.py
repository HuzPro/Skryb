"""Tests for :class:`SoundDeviceAdapter`.

The adapter is driven through its public port - ``start_recording``,
``stop_recording``, ``is_recording``, ``dispose`` - and chunks are
fed via the same callback that ``sd.InputStream`` would invoke in
production.  No private-attribute poking.
"""

from unittest.mock import patch

import numpy as np
import pytest

from voicetotext.core.exceptions import AudioRecorderError
from tests.recording.test_input_devices import crowded_device_query
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.input_devices import (
    connected_preferred_microphone,
    input_device_index,
)
from voicetotext.recording.recorder import AudioRecorder
from voicetotext.recording.sounddevice_adapter import SoundDeviceAdapter


def _feed(mock_sd, *chunks: np.ndarray) -> None:
    """Fire the callback that the adapter registered with ``sd.InputStream``
    with each of the given chunks - the same path ``sounddevice`` uses
    in production when audio data arrives from the device."""
    callback = mock_sd.InputStream.call_args.kwargs["callback"]
    for chunk in chunks:
        callback(chunk, len(chunk), None, None)


class TestSoundDeviceAdapterStartRecording:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_open_input_stream_on_start(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)

        adapter.start_recording()

        mock_sd.InputStream.assert_called_once()
        mock_sd.InputStream.return_value.start.assert_called_once()

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_report_is_recording_after_start(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)

        adapter.start_recording()

        assert adapter.is_recording() is True

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_raise_when_already_recording(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()

        with pytest.raises(AudioRecorderError, match="already recording"):
            adapter.start_recording()


class TestSoundDeviceAdapterStopRecording:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_return_audio_segment_when_audio_was_captured(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()
        _feed(mock_sd, np.zeros((1024, 1), dtype=np.float32))

        result = adapter.stop_recording()

        assert isinstance(result, AudioSegment)
        assert result.sample_rate == 16000

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_stop_and_close_stream_on_stop(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()

        adapter.stop_recording()

        mock_sd.InputStream.return_value.stop.assert_called_once()
        mock_sd.InputStream.return_value.close.assert_called_once()

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_report_not_recording_after_stop(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()

        adapter.stop_recording()

        assert adapter.is_recording() is False

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_assemble_captured_chunks_into_a_single_segment(
        self, mock_sd
    ):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()
        _feed(
            mock_sd,
            np.ones((1024, 1), dtype=np.float32),
            np.ones((512, 1), dtype=np.float32) * 0.5,
        )

        result = adapter.stop_recording()

        assert result is not None
        assert result.sample_count == 1536

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_preserve_captured_samples_in_result(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()
        chunk_a = np.full((4, 1), 0.25, dtype=np.float32)
        chunk_b = np.full((4, 1), 0.75, dtype=np.float32)
        _feed(mock_sd, chunk_a, chunk_b)

        result = adapter.stop_recording()

        assert result is not None
        np.testing.assert_array_equal(
            result.data,
            np.array(
                [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75],
                dtype=np.float32,
            ),
        )

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_raise_when_not_recording(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)

        with pytest.raises(AudioRecorderError, match="not recording"):
            adapter.stop_recording()


class TestSoundDeviceAdapterEmptyRecording:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_return_none_when_no_chunks_were_captured(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()

        result = adapter.stop_recording()

        assert result is None


class TestSoundDeviceAdapterLiveFrames:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_publish_latest_chunk_rms_without_buffering(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()

        _feed(mock_sd, np.full((4, 1), 0.5, dtype=np.float32))

        assert adapter.current_rms == pytest.approx(0.5)

        adapter.stop_recording()
        assert adapter.current_rms == 0.0

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_forward_mono_one_dimensional_frame_to_on_frame(self, mock_sd):
        # The recorder contract promises "float32, mono" frames. sounddevice
        # delivers (frames, 1); the adapter must flatten to (frames,) so the
        # streaming model's torch.stft does not reflect-pad a size-1 axis.
        adapter = SoundDeviceAdapter(sample_rate=16000)
        received: list[np.ndarray] = []

        adapter.start_recording(on_frame=received.append)
        _feed(
            mock_sd,
            np.ones((1024, 1), dtype=np.float32),
            np.zeros((512, 1), dtype=np.float32),
        )

        assert len(received) == 2
        assert received[0].shape == (1024,)
        assert received[1].shape == (512,)

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_still_assemble_segment_when_on_frame_set(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)

        adapter.start_recording(on_frame=lambda chunk: None)
        _feed(mock_sd, np.ones((1024, 1), dtype=np.float32))
        result = adapter.stop_recording()

        assert result is not None
        assert result.sample_count == 1024

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_pass_a_copy_to_on_frame(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        received: list[np.ndarray] = []
        adapter.start_recording(on_frame=received.append)

        chunk = np.ones((4, 1), dtype=np.float32)
        _feed(mock_sd, chunk)
        chunk[0, 0] = 9.0  # mutate the source after the callback ran

        assert received[0][0] == 1.0  # on_frame got an independent (1-D) copy

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_clear_on_frame_after_stop(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        received: list[np.ndarray] = []
        adapter.start_recording(on_frame=received.append)
        adapter.stop_recording()

        callback = mock_sd.InputStream.call_args.kwargs["callback"]
        callback(np.ones((4, 1), dtype=np.float32), 4, None, None)

        assert received == []


class TestSoundDeviceAdapterDispose:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_stop_recording_on_dispose_if_active(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()

        adapter.dispose()

        assert adapter.is_recording() is False

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_not_raise_on_dispose_when_idle(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)

        adapter.dispose()


class TestSoundDeviceAdapterIsAnAudioRecorder:
    def test_should_implement_audio_recorder_interface(self):
        with patch("voicetotext.recording.sounddevice_adapter.sd"):
            adapter = SoundDeviceAdapter(sample_rate=16000)
            assert isinstance(adapter, AudioRecorder)


class ChosenMicrophone:
    """Stands in for the saved choice and PortAudio's device list."""

    def __init__(self, name: str, connected: dict[str, int]) -> None:
        self.name = name
        self.connected = dict(connected)
        self.missing: list[str] = []
        self.rescans = 0
        self.plug_in_on_rescan: dict[str, int] = {}

    def adapter(self) -> SoundDeviceAdapter:
        return SoundDeviceAdapter(
            device_name=lambda: self.name,
            on_device_missing=self.missing.append,
            device_index=self.connected.get,
            refresh_devices=self._rescan,
        )

    def _rescan(self) -> None:
        self.rescans += 1
        self.connected.update(self.plug_in_on_rescan)


def _opened_device(mock_sd) -> object:
    return mock_sd.InputStream.call_args.kwargs["device"]


class TestChoosingTheMicrophone:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_records_from_the_windows_default_unless_told_otherwise(self, mock_sd):
        SoundDeviceAdapter().start_recording()

        assert _opened_device(mock_sd) is None

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_records_from_the_chosen_microphone(self, mock_sd):
        name = "Microphone (Maono PD300X Microp"
        mic = ChosenMicrophone(name, {name: 7})

        mic.adapter().start_recording()

        assert _opened_device(mock_sd) == 7
        assert mic.missing == []

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_looks_for_a_newly_plugged_microphone_before_giving_up(self, mock_sd):
        mic = ChosenMicrophone("USB Mic", {})
        mic.plug_in_on_rescan = {"USB Mic": 3}

        mic.adapter().start_recording()

        assert _opened_device(mock_sd) == 3
        assert mic.missing == []

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_falls_back_to_the_default_and_says_so_when_it_is_unplugged(self, mock_sd):
        mic = ChosenMicrophone("USB Mic", {})

        mic.adapter().start_recording()

        assert _opened_device(mock_sd) is None
        assert mic.missing == ["USB Mic"]
        assert mic.rescans == 1

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_says_so_once_per_absence_not_once_per_recording(self, mock_sd):
        mic = ChosenMicrophone("USB Mic", {})
        adapter = mic.adapter()

        for _ in range(5):
            adapter.start_recording()
            adapter.stop_recording()

        assert mic.missing == ["USB Mic"]

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_goes_back_to_the_microphone_once_it_returns_and_warns_again_next_time(
        self, mock_sd,
    ):
        mic = ChosenMicrophone("USB Mic", {})
        adapter = mic.adapter()
        adapter.start_recording()
        adapter.stop_recording()

        mic.connected["USB Mic"] = 4
        adapter.start_recording()
        adapter.stop_recording()
        assert _opened_device(mock_sd) == 4

        del mic.connected["USB Mic"]
        adapter.start_recording()

        assert mic.missing == ["USB Mic", "USB Mic"]

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_falls_back_when_the_chosen_microphone_refuses_to_open(self, mock_sd):
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        opened = []

        def open_stream(**kwargs):
            opened.append(kwargs["device"])
            if kwargs["device"] is not None:
                raise mock_sd.PortAudioError("Device unavailable")
            return mock_sd.stream

        mock_sd.InputStream.side_effect = open_stream
        mic = ChosenMicrophone("Busy Mic", {"Busy Mic": 2})
        adapter = mic.adapter()

        adapter.start_recording()

        assert opened == [2, None]
        assert mic.missing == ["Busy Mic"]
        assert adapter.is_recording()

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_a_broken_default_microphone_is_still_an_error(self, mock_sd):
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.InputStream.side_effect = mock_sd.PortAudioError("No default input")

        with pytest.raises(mock_sd.PortAudioError):
            SoundDeviceAdapter().start_recording()

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_a_failing_notice_never_stops_the_recording(self, mock_sd):
        def broken_notice(_name):
            raise RuntimeError("tray gone")

        adapter = SoundDeviceAdapter(
            device_name=lambda: "USB Mic",
            on_device_missing=broken_notice,
            device_index=lambda _name: None,
            refresh_devices=lambda: None,
        )

        adapter.start_recording()

        assert adapter.is_recording()


class TestPausing:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_audio_that_arrives_while_paused_is_dropped(self, mock_sd):
        adapter = SoundDeviceAdapter()
        adapter.start_recording()
        _feed(mock_sd, np.full((1024, 1), 0.1, dtype=np.float32))

        adapter.set_paused(True)
        _feed(mock_sd, np.full((1024, 1), 0.9, dtype=np.float32))
        adapter.set_paused(False)
        _feed(mock_sd, np.full((1024, 1), 0.2, dtype=np.float32))

        audio = adapter.stop_recording()
        assert audio.sample_count == 2048
        assert float(audio.data.max()) == pytest.approx(0.2)

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_a_new_recording_never_starts_paused(self, mock_sd):
        adapter = SoundDeviceAdapter()
        adapter.start_recording()
        adapter.set_paused(True)
        adapter.stop_recording()

        adapter.start_recording()
        _feed(mock_sd, np.full((1024, 1), 0.1, dtype=np.float32))

        assert adapter.stop_recording() is not None


class PreferredMicrophones(ChosenMicrophone):
    """A chosen microphone plus an ordered wish list, matched on connection."""

    def __init__(self, name, connected, preferred) -> None:
        super().__init__(name, connected)
        self.preferred = list(preferred)
        self.broken = False

    def adapter(self) -> SoundDeviceAdapter:
        return SoundDeviceAdapter(
            device_name=lambda: self.name,
            on_device_missing=self.missing.append,
            device_index=self.connected.get,
            refresh_devices=self._rescan,
            preferred_names=lambda: self.preferred,
            find_preferred=self._find,
        )

    def _find(self, preferred):
        if self.broken:
            raise OSError("PortAudio not initialized")
        return next((name for name in preferred if name in self.connected), None)


class TestPreferredMicrophonesAtRecordingStart:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_the_first_connected_preferred_microphone_beats_the_chosen_one(self, mock_sd):
        mics = PreferredMicrophones(
            "Built-in Mic",
            {"Built-in Mic": 1, "Jabra": 5, "Blue Yeti": 9},
            ["Unplugged Rode", "Blue Yeti", "Jabra"],
        )
        adapter = mics.adapter()

        adapter.start_recording()

        assert _opened_device(mock_sd) == 9
        assert adapter.device_name == "Blue Yeti"

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_with_none_of_them_connected_the_chosen_microphone_is_used(self, mock_sd):
        mics = PreferredMicrophones(
            "Built-in Mic", {"Built-in Mic": 1}, ["Unplugged Rode", "Blue Yeti"],
        )
        adapter = mics.adapter()

        adapter.start_recording()

        assert _opened_device(mock_sd) == 1
        assert adapter.device_name == "Built-in Mic"
        assert mics.missing == []

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_with_nothing_connected_it_records_from_the_windows_default(self, mock_sd):
        mics = PreferredMicrophones("USB Mic", {}, ["Blue Yeti"])
        adapter = mics.adapter()

        adapter.start_recording()

        assert _opened_device(mock_sd) is None
        assert adapter.device_name == ""
        assert mics.missing == ["USB Mic"]

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_the_choice_is_made_again_at_every_start(self, mock_sd):
        mics = PreferredMicrophones("Built-in Mic", {"Built-in Mic": 1}, ["Blue Yeti"])
        adapter = mics.adapter()
        adapter.start_recording()
        adapter.stop_recording()

        mics.connected["Blue Yeti"] = 9
        adapter.start_recording()

        assert _opened_device(mock_sd) == 9

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_a_broken_device_list_falls_back_to_the_chosen_microphone(self, mock_sd):
        mics = PreferredMicrophones("Built-in Mic", {"Built-in Mic": 1}, ["Blue Yeti"])
        mics.broken = True

        mics.adapter().start_recording()

        assert _opened_device(mock_sd) == 1

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_a_preferred_microphone_that_will_not_open_falls_back_to_the_default(
        self, mock_sd,
    ):
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})

        def open_stream(**kwargs):
            if kwargs["device"] is not None:
                raise mock_sd.PortAudioError("Device unavailable")
            return mock_sd.stream

        mock_sd.InputStream.side_effect = open_stream
        mics = PreferredMicrophones("", {"Blue Yeti": 9}, ["Blue Yeti"])
        adapter = mics.adapter()

        adapter.start_recording()

        assert adapter.device_name == ""
        assert mics.missing == ["Blue Yeti"]


class TestChoosingAmongAThousandDevices:
    @pytest.mark.parametrize(
        ("preferred", "chosen", "opened"),
        [
            (
                ["Unplugged Rode", "Headset (Jabra Evolve 20 MS Stereo)"],
                "Microphone (Maono PD300X Microp",
                "Headset (Jabra Evolve 20 MS Ste",
            ),
            (
                ["Unplugged Rode", ""],
                "Mikrofon (Gerät für Café ✓)",
                "Mikrofon (Gerät für Café ✓)",
            ),
            (["Unplugged Rode"], "Unplugged Blue Yeti", ""),
            ([], "", ""),
        ],
    )
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_first_connected_preference_then_the_chosen_one_then_the_default(
        self, mock_sd, preferred, chosen, opened,
    ):
        adapter = SoundDeviceAdapter(
            device_name=lambda: chosen,
            device_index=lambda name: input_device_index(name, crowded_device_query),
            refresh_devices=lambda: None,
            preferred_names=lambda: preferred,
            find_preferred=lambda wishes: connected_preferred_microphone(
                wishes, crowded_device_query,
            ),
        )

        adapter.start_recording()

        assert adapter.device_name == opened
        devices, _host_api = crowded_device_query()
        device = _opened_device(mock_sd)
        assert (devices[device]["name"] if device is not None else "") == opened
