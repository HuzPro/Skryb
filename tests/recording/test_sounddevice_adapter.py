"""Tests for :class:`SoundDeviceAdapter`.

The adapter is driven through its public port — ``start_recording``,
``stop_recording``, ``is_recording``, ``dispose`` — and chunks are
fed via the same callback that ``sd.InputStream`` would invoke in
production.  No private-attribute poking.
"""

from unittest.mock import patch

import numpy as np
import pytest

from voicetotext.core.exceptions import AudioRecorderError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.recorder import AudioRecorder
from voicetotext.recording.sounddevice_adapter import SoundDeviceAdapter


def _feed(mock_sd, *chunks: np.ndarray) -> None:
    """Fire the callback that the adapter registered with ``sd.InputStream``
    with each of the given chunks — the same path ``sounddevice`` uses
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
    def test_should_forward_each_frame_to_on_frame(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        received: list[np.ndarray] = []

        adapter.start_recording(on_frame=received.append)
        _feed(
            mock_sd,
            np.ones((1024, 1), dtype=np.float32),
            np.zeros((512, 1), dtype=np.float32),
        )

        assert len(received) == 2
        assert received[0].shape == (1024, 1)

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

        assert received[0][0, 0] == 1.0  # on_frame got an independent copy

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
