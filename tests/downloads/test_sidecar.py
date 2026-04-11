import json

import pytest

from voicetotext.downloads.sidecar import (
    Sidecar,
    SidecarCorruptError,
    SidecarVersionError,
)
from voicetotext.downloads.state import ChunkState


class TestSidecarRoundTrip:
    def test_should_write_and_read_back_identical_content(self, tmp_path):
        path = tmp_path / "model.safetensors.part.meta"
        original = Sidecar(
            url="https://example.com/model.safetensors",
            etag='"abc123"',
            content_length=1000,
            chunks=[
                ChunkState(index=0, start=0, end=499, bytes_done=499),
                ChunkState(index=1, start=500, end=999, bytes_done=200),
            ],
        )

        original.save(path)
        loaded = Sidecar.load(path)

        assert loaded == original

    def test_save_should_create_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "file.meta"
        sidecar = Sidecar(
            url="https://example.com/f",
            etag=None,
            content_length=10,
            chunks=[ChunkState(index=0, start=0, end=9, bytes_done=0)],
        )

        sidecar.save(path)

        assert path.exists()


class TestSidecarLoad:
    def test_should_raise_file_not_found_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Sidecar.load(tmp_path / "nope.meta")

    def test_should_raise_corrupt_when_json_invalid(self, tmp_path):
        path = tmp_path / "broken.meta"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(SidecarCorruptError):
            Sidecar.load(path)

    def test_should_raise_version_error_when_version_unknown(self, tmp_path):
        path = tmp_path / "future.meta"
        path.write_text(
            json.dumps({"version": 999, "url": "x", "content_length": 1, "chunks": []}),
            encoding="utf-8",
        )

        with pytest.raises(SidecarVersionError):
            Sidecar.load(path)

    def test_should_raise_corrupt_when_required_fields_missing(self, tmp_path):
        path = tmp_path / "partial.meta"
        path.write_text(json.dumps({"version": 1}), encoding="utf-8")

        with pytest.raises(SidecarCorruptError):
            Sidecar.load(path)


class TestSidecarBytesDone:
    def test_should_sum_bytes_done_across_chunks(self):
        sidecar = Sidecar(
            url="u",
            etag=None,
            content_length=1000,
            chunks=[
                ChunkState(index=0, start=0, end=499, bytes_done=499),
                ChunkState(index=1, start=500, end=999, bytes_done=200),
            ],
        )

        assert sidecar.bytes_done == 699

    def test_bytes_done_is_zero_for_fresh_sidecar(self):
        sidecar = Sidecar(
            url="u",
            etag=None,
            content_length=1000,
            chunks=[
                ChunkState(index=0, start=0, end=499, bytes_done=0),
                ChunkState(index=1, start=500, end=999, bytes_done=0),
            ],
        )

        assert sidecar.bytes_done == 0

    def test_is_complete_when_all_chunks_complete(self):
        sidecar = Sidecar(
            url="u",
            etag=None,
            content_length=1000,
            chunks=[
                ChunkState(index=0, start=0, end=499, bytes_done=500),
                ChunkState(index=1, start=500, end=999, bytes_done=500),
            ],
        )

        assert sidecar.is_complete

    def test_is_not_complete_when_any_chunk_incomplete(self):
        sidecar = Sidecar(
            url="u",
            etag=None,
            content_length=1000,
            chunks=[
                ChunkState(index=0, start=0, end=499, bytes_done=500),
                ChunkState(index=1, start=500, end=999, bytes_done=400),
            ],
        )

        assert not sidecar.is_complete


class TestSidecarUpdateChunk:
    def test_should_return_new_sidecar_with_updated_chunk(self):
        sidecar = Sidecar(
            url="u",
            etag=None,
            content_length=1000,
            chunks=[
                ChunkState(index=0, start=0, end=499, bytes_done=0),
                ChunkState(index=1, start=500, end=999, bytes_done=0),
            ],
        )

        updated = sidecar.with_chunk_progress(index=1, bytes_done=300)

        assert updated.chunks[0].bytes_done == 0
        assert updated.chunks[1].bytes_done == 300
        # Original stays immutable
        assert sidecar.chunks[1].bytes_done == 0
