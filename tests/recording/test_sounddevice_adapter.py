from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import AudioRecorderError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.recorder import AudioRecorder
from voicetotext.recording.sounddevice_adapter import SoundDeviceAdapter


class TestSoundDeviceAdapterStartRecording:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_open_input_stream_on_start(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)

        adapter.start_recording()

        mock_sd.InputStream.assert_called_once()
        mock_sd.InputStream.return_value.start.assert_called_once()

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_set_recording_flag(self, mock_sd):
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
    def test_should_return_audio_segment_on_stop(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()
        adapter._chunks = [np.zeros((1024, 1), dtype=np.float32)]

        result = adapter.stop_recording()

        assert isinstance(result, AudioSegment)
        assert result.sample_rate == 16000

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_stop_stream_on_stop(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()
        adapter._chunks = [np.zeros((1024, 1), dtype=np.float32)]

        adapter.stop_recording()

        mock_sd.InputStream.return_value.stop.assert_called_once()
        mock_sd.InputStream.return_value.close.assert_called_once()

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_clear_recording_flag_on_stop(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()
        adapter._chunks = [np.zeros((1024, 1), dtype=np.float32)]

        adapter.stop_recording()

        assert adapter.is_recording() is False

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_raise_when_not_recording(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)

        with pytest.raises(AudioRecorderError, match="not recording"):
            adapter.stop_recording()

    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_concatenate_audio_chunks(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()
        chunk1 = np.ones((1024, 1), dtype=np.float32)
        chunk2 = np.ones((512, 1), dtype=np.float32) * 0.5
        adapter._chunks = [chunk1, chunk2]

        result = adapter.stop_recording()

        assert result.sample_count == 1536


class TestSoundDeviceAdapterDispose:
    @patch("voicetotext.recording.sounddevice_adapter.sd")
    def test_should_stop_recording_on_dispose_if_active(self, mock_sd):
        adapter = SoundDeviceAdapter(sample_rate=16000)
        adapter.start_recording()
        adapter._chunks = [np.zeros((1024, 1), dtype=np.float32)]

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
