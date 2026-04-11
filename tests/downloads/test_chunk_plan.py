import pytest

from voicetotext.downloads.chunk_plan import plan_chunks


class TestPlanChunks:
    def test_should_split_evenly_when_size_divides_cleanly(self):
        ranges = plan_chunks(total_size=1000, n_chunks=4)

        assert len(ranges) == 4
        assert ranges[0].start == 0 and ranges[0].end == 249
        assert ranges[1].start == 250 and ranges[1].end == 499
        assert ranges[2].start == 500 and ranges[2].end == 749
        assert ranges[3].start == 750 and ranges[3].end == 999

    def test_last_chunk_absorbs_remainder_on_uneven_split(self):
        ranges = plan_chunks(total_size=1003, n_chunks=4)

        assert len(ranges) == 4
        assert ranges[0].end == 249
        assert ranges[1].end == 499
        assert ranges[2].end == 749
        assert ranges[3].start == 750 and ranges[3].end == 1002

    def test_chunks_should_have_sequential_indices(self):
        ranges = plan_chunks(total_size=1000, n_chunks=4)

        assert [r.index for r in ranges] == [0, 1, 2, 3]

    def test_single_chunk_covers_entire_file(self):
        ranges = plan_chunks(total_size=500, n_chunks=1)

        assert len(ranges) == 1
        assert ranges[0].start == 0
        assert ranges[0].end == 499

    def test_should_collapse_to_one_chunk_when_file_smaller_than_requested_chunks(self):
        ranges = plan_chunks(total_size=5, n_chunks=8)

        assert len(ranges) == 1
        assert ranges[0].start == 0
        assert ranges[0].end == 4

    def test_should_reject_zero_or_negative_size(self):
        with pytest.raises(ValueError):
            plan_chunks(total_size=0, n_chunks=4)

    def test_should_reject_zero_or_negative_chunk_count(self):
        with pytest.raises(ValueError):
            plan_chunks(total_size=1000, n_chunks=0)

    def test_ranges_should_cover_file_without_gaps_or_overlaps(self):
        ranges = plan_chunks(total_size=10_000_000, n_chunks=8)

        assert ranges[0].start == 0
        assert ranges[-1].end == 9_999_999
        for earlier, later in zip(ranges, ranges[1:]):
            assert later.start == earlier.end + 1

    def test_byte_count_per_range_equals_total(self):
        ranges = plan_chunks(total_size=9_999_999, n_chunks=7)

        total = sum(r.end - r.start + 1 for r in ranges)
        assert total == 9_999_999
