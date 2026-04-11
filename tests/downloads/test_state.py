import pytest

from voicetotext.downloads.state import (
    ChunkState,
    DownloadState,
    DownloadStats,
)


class TestDownloadState:
    def test_should_have_all_lifecycle_states(self):
        assert DownloadState.QUEUED.value == "queued"
        assert DownloadState.DOWNLOADING.value == "downloading"
        assert DownloadState.PAUSED.value == "paused"
        assert DownloadState.COMPLETED.value == "completed"
        assert DownloadState.CANCELLED.value == "cancelled"
        assert DownloadState.FAILED.value == "failed"


class TestChunkState:
    def test_should_store_range_and_progress(self):
        state = ChunkState(index=0, start=0, end=999, bytes_done=500)

        assert state.index == 0
        assert state.start == 0
        assert state.end == 999
        assert state.bytes_done == 500

    def test_should_report_length_including_both_endpoints(self):
        state = ChunkState(index=0, start=0, end=999, bytes_done=0)

        assert state.length == 1000

    def test_should_report_remaining_bytes(self):
        state = ChunkState(index=0, start=0, end=999, bytes_done=300)

        assert state.remaining == 700

    def test_is_complete_when_bytes_done_equals_length(self):
        state = ChunkState(index=0, start=0, end=999, bytes_done=1000)

        assert state.is_complete

    def test_is_not_complete_when_bytes_done_less_than_length(self):
        state = ChunkState(index=0, start=0, end=999, bytes_done=999)

        assert not state.is_complete

    def test_should_be_frozen(self):
        state = ChunkState(index=0, start=0, end=999, bytes_done=0)

        with pytest.raises((AttributeError, Exception)):
            state.bytes_done = 500  # type: ignore[misc]


class TestDownloadStats:
    def test_should_store_aggregate_numbers(self):
        stats = DownloadStats(
            bytes_done=500,
            bytes_total=1000,
            speed_bps=1024.0,
            eta_seconds=0.5,
        )

        assert stats.bytes_done == 500
        assert stats.bytes_total == 1000
        assert stats.speed_bps == 1024.0
        assert stats.eta_seconds == 0.5

    def test_fraction_should_be_zero_before_any_bytes_arrive(self):
        stats = DownloadStats(bytes_done=0, bytes_total=1000, speed_bps=0.0, eta_seconds=None)

        assert stats.fraction == 0.0

    def test_fraction_should_equal_ratio_when_partially_done(self):
        stats = DownloadStats(bytes_done=250, bytes_total=1000, speed_bps=0.0, eta_seconds=None)

        assert stats.fraction == 0.25

    def test_fraction_should_be_zero_when_total_unknown(self):
        stats = DownloadStats(bytes_done=100, bytes_total=0, speed_bps=0.0, eta_seconds=None)

        assert stats.fraction == 0.0

    def test_fraction_should_cap_at_one(self):
        stats = DownloadStats(bytes_done=2000, bytes_total=1000, speed_bps=0.0, eta_seconds=None)

        assert stats.fraction == 1.0
