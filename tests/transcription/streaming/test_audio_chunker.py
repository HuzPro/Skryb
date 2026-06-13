import numpy as np

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.streaming.audio_chunker import iter_chunks


def _ramp(samples: int, sample_rate: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.arange(samples, dtype=np.float32), sample_rate=sample_rate,
    )


class TestIterChunks:
    def test_should_reconstruct_original_when_concatenated(self):
        audio = _ramp(16000)

        chunks = list(iter_chunks(audio, chunk_seconds=0.1))

        np.testing.assert_array_equal(np.concatenate(chunks), audio.data)

    def test_should_emit_ceil_count_with_short_final_chunk(self):
        audio = _ramp(16050)  # 1600 samples/chunk -> 11 chunks, last is 50

        chunks = list(iter_chunks(audio, chunk_seconds=0.1))

        assert len(chunks) == 11
        assert len(chunks[-1]) == 50

    def test_should_preserve_float32_dtype(self):
        audio = _ramp(3200)

        for chunk in iter_chunks(audio, chunk_seconds=0.1):
            assert chunk.dtype == np.float32

    def test_should_yield_single_chunk_when_chunk_exceeds_audio(self):
        audio = _ramp(800)

        chunks = list(iter_chunks(audio, chunk_seconds=1.0))

        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], audio.data)

    def test_should_yield_one_chunk_for_single_sample_audio(self):
        audio = _ramp(1)

        chunks = list(iter_chunks(audio, chunk_seconds=0.1))

        assert len(chunks) == 1
        assert len(chunks[0]) == 1
